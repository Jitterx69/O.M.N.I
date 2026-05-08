include("../../src/dvs_loader.jl")

const SURROGATE_SCALE = 5.0
const V_THRESHOLD = 0.85
const V_LEAK = 0.85
const SNN_T_STEPS = DVS_TIME_BINS
const SPIKE_DROPOUT = 0.25

lrelu(x) = x > 0 ? x : 0.01x
lrelu_d(x) = x > 0 ? 1.0 : 0.01

function surrogate_grad(x)
    1.0 / (1.0 + SURROGATE_SCALE * abs(x))^2
end

function softmax_s(X)
    mx = maximum(X, dims=1)
    e = exp.(X .- mx)
    e ./ sum(e, dims=1)
end

# Spiking Convolutional Layer
mutable struct SpikeConv
    in_ch::Int; out_ch::Int
    k::Int; pad::Int
    W::Array{Float64, 4} # (out, in, k, k)
    b::Vector{Float64}
    dW::Array{Float64, 4}; db::Vector{Float64}
    mW::Array{Float64, 4}; vW::Array{Float64, 4}
    mb::Vector{Float64}; vb::Vector{Float64}
    inp::Array{Float64, 5} # Store for BWD
end

function SpikeConv(in_ch, out_ch, k; pad=1)
    sc = sqrt(2.0 / (in_ch * k * k))
    SpikeConv(in_ch, out_ch, k, pad,
        randn(out_ch, in_ch, k, k) .* sc, zeros(out_ch),
        zeros(out_ch, in_ch, k, k), zeros(out_ch),
        zeros(out_ch, in_ch, k, k), zeros(out_ch, in_ch, k, k),
        zeros(out_ch), zeros(out_ch),
        zeros(0,0,0,0,0))
end

function conv2d(X::Array{Float64, 4}, W::Array{Float64, 4}, b::Vector{Float64}, pad::Int)
    B, C, H, W_in = size(X)
    OC, IC, k, _ = size(W)
    H_out = H + 2*pad - k + 1
    W_out = W_in + 2*pad - k + 1
    
    out = zeros(B, OC, H_out, W_out)
    X_pad = zeros(B, C, H + 2*pad, W_in + 2*pad)
    X_pad[:, :, pad+1:H+pad, pad+1:W_in+pad] = X
    
    @inbounds for oc in 1:OC
        for ic in 1:IC
            for i in 1:k, j in 1:k
                w_val = W[oc, ic, i, j]
                @simd for row in 1:H_out
                    for b in 1:B
                        @views out[b, oc, row, :] .+= w_val .* X_pad[b, ic, i+row-1, j:j+W_out-1]
                    end
                end
            end
        end
        @views out[:, oc, :, :] .+= b[oc]
    end
    return out
end

function conv2d_grad(X::Array{Float64, 4}, W::Array{Float64, 4}, d_out::Array{Float64, 4}, pad::Int)
    B, C, H, W_in = size(X)
    OC, IC, k, _ = size(W)
    H_out, W_out = size(d_out, 3), size(d_out, 4)
    
    dW = zeros(size(W))
    db = vec(sum(d_out, dims=(1,3,4)))
    dX = zeros(size(X))
    
    X_pad = zeros(B, C, H + 2*pad, W_in + 2*pad)
    X_pad[:, :, pad+1:H+pad, pad+1:W_in+pad] = X
    dX_pad = zeros(size(X_pad))
    
    @inbounds for oc in 1:OC
        d_out_oc = d_out[:, oc, :, :]
        for ic in 1:IC
            for i in 1:k, j in 1:k
                # dW
                slice = X_pad[:, ic, i:i+H_out-1, j:j+W_out-1]
                dW[oc, ic, i, j] = sum(d_out_oc .* slice)
                # dX
                dX_pad[:, ic, i:i+H_out-1, j:j+W_out-1] .+= W[oc, ic, i, j] .* d_out_oc
            end
        end
    end
    dX = dX_pad[:, :, pad+1:H+pad, pad+1:W_in+pad]
    return dW, db, dX
end

function bwd_conv!(l::SpikeConv, d_out_seq::Array{Float64, 5}, lambda::Float64, clip::Float64)
    B, OC, H, W, T = size(d_out_seq)
    l.dW .= 0.0; l.db .= 0.0
    
    for t in 1:T
        dw, db, dx = conv2d_grad(l.inp[:, :, :, :, t], l.W, d_out_seq[:, :, :, :, t], l.pad)
        l.dW .+= dw ./ B
        l.db .+= db ./ B
    end
    
    l.dW .+= lambda .* l.W
    gnorm = sqrt(sum(l.dW.^2) + sum(l.db.^2))
    if gnorm > clip
        sc = clip / gnorm; l.dW .*= sc; l.db .*= sc
    end
end

mutable struct SpikeDense
    ni::Int; no::Int
    W::Matrix{Float64}; b::Vector{Float64}
    dW::Matrix{Float64}; db::Vector{Float64}
    mW::Matrix{Float64}; vW::Matrix{Float64}
    mb::Vector{Float64}; vb::Vector{Float64}
end

function SpikeDense(ni, no)
    sc = sqrt(2.0 / ni)
    SpikeDense(ni, no, randn(no, ni) .* sc, zeros(no),
               zeros(no, ni), zeros(no),
               zeros(no, ni), zeros(no, ni),
               zeros(no), zeros(no))
end

mutable struct ConvLIFBlock
    conv::SpikeConv
    U::Array{Float64, 5} # (B, C, H, W, T+1)
    S::Array{Float64, 5} # (B, C, H, W, T)
    inp::Array{Float64, 5}
    v_th::Float64
end

function ConvLIFBlock(in_ch, out_ch, k; pad=1)
    ConvLIFBlock(SpikeConv(in_ch, out_ch, k; pad=pad), 
                 zeros(0,0,0,0,0), zeros(0,0,0,0,0), zeros(0,0,0,0,0), V_THRESHOLD)
end

function fwd_conv_lif!(blk::ConvLIFBlock, X_seq::Array{Float64, 5}; training=false)
    B, C, H, W, T = size(X_seq)
    OC = blk.conv.out_ch
    blk.conv.inp = X_seq # Store for BWD
    
    blk.U = zeros(B, OC, H, W, T + 1)
    blk.S = zeros(B, OC, H, W, T)
    
    for t in 1:T
        X_t = X_seq[:, :, :, :, t]
        curr = conv2d(X_t, blk.conv.W, blk.conv.b, blk.conv.pad)
        blk.U[:, :, :, :, t+1] = V_LEAK .* blk.U[:, :, :, :, t] .+ (1.0 - V_LEAK) .* curr
        spikes = Float64.(blk.U[:, :, :, :, t+1] .> blk.v_th)
        if training
            mask = rand(size(spikes)...) .> SPIKE_DROPOUT
            spikes .*= mask
        end
        blk.S[:, :, :, :, t] = spikes
        blk.U[:, :, :, :, t+1] .-= spikes .* blk.v_th
    end
    
    if training
        rate = mean(blk.S)
        if rate < 0.01; blk.v_th *= 0.98; elseif rate > 0.1; blk.v_th *= 1.02; end
        blk.v_th = clamp(blk.v_th, 0.3, 2.0)
    end
    return blk.S
end

function bwd_conv_lif!(blk::ConvLIFBlock, dS::Array{Float64, 5}, lambda::Float64, clip::Float64)
    B, OC, H, W, T = size(dS)
    dU = zeros(B, OC, H, W)
    dI_seq = zeros(B, OC, H, W, T)
    
    for t in T:-1:1
        grad_s = surrogate_grad.(blk.U[:, :, :, :, t+1] .- blk.v_th)
        dU_curr = dS[:, :, :, :, t] .+ dU .* V_LEAK
        dU_fire = dU_curr .* grad_s
        dI_seq[:, :, :, :, t] = dU_fire .* (1.0 - V_LEAK)
        dU = dU_curr .- dU_fire .* blk.v_th
    end
    
    return bwd_conv!(blk.conv, dI_seq, lambda, clip)
end

mutable struct LIFBlock
    layer::SpikeDense
    U::Array{Float64, 3}; S::Array{Float64, 3}; inp::Array{Float64, 3}
    v_th::Float64
end

function LIFBlock(ni, no)
    LIFBlock(SpikeDense(ni, no), zeros(0,0,0), zeros(0,0,0), zeros(0,0,0), V_THRESHOLD)
end

function fwd_lif!(blk::LIFBlock, X_seq::Array{Float64, 3}; training=false)
    no = blk.layer.no; B = size(X_seq, 2); T = size(X_seq, 3)
    blk.inp = X_seq
    blk.U = zeros(no, B, T+1); blk.S = zeros(no, B, T)
    for t in 1:T
        curr = blk.layer.W * X_seq[:, :, t] .+ blk.layer.b
        blk.U[:, :, t+1] = V_LEAK .* blk.U[:, :, t] .+ (1.0 - V_LEAK) .* curr
        sp = Float64.(blk.U[:, :, t+1] .> blk.v_th)
        if training; sp .*= (rand(size(sp)...) .> SPIKE_DROPOUT); end
        blk.S[:, :, t] = sp
        blk.U[:, :, t+1] .-= sp .* blk.v_th
    end
    if training
        rate = mean(blk.S)
        if rate < 0.02; blk.v_th *= 0.95; elseif rate > 0.15; blk.v_th *= 1.05; end
        blk.v_th = clamp(blk.v_th, 0.2, 2.0)
    end
    return blk.S
end

function bwd_lif!(blk::LIFBlock, dS::Array{Float64, 3}, lambda::Float64, clip::Float64)
    no, B, T = size(dS)
    l = blk.layer
    l.dW .= 0.0; l.db .= 0.0
    dU = zeros(no, B)
    dX = zeros(size(blk.inp))
    for t in T:-1:1
        grad_s = surrogate_grad.(blk.U[:, :, t+1] .- blk.v_th)
        dU_curr = dS[:, :, t] .+ dU .* V_LEAK
        dU_fire = dU_curr .* grad_s
        dI = dU_fire .* (1.0 - V_LEAK)
        l.dW .+= (dI * blk.inp[:, :, t]') ./ B
        l.db .+= vec(sum(dI, dims=2)) ./ B
        dX[:, :, t] = l.W' * dI
        dU = dU_curr .- dU_fire .* blk.v_th
    end
    l.dW .+= lambda .* l.W
    gnorm = sqrt(sum(l.dW.^2) + sum(l.db.^2))
    if gnorm > clip; s = clip/gnorm; l.dW .*= s; l.db .*= s; end
    return dX
end

mutable struct TemporalAttention
    w::Vector{Float64}; dw::Vector{Float64}
    mw::Vector{Float64}; vw::Vector{Float64}
end

function TemporalAttention(T)
    TemporalAttention(ones(T) ./ T, zeros(T), zeros(T), zeros(T))
end

function apply_attention(attn::TemporalAttention, S::Array{Float64, 3})
    no, B, T = size(S)
    alpha = exp.(attn.w) ./ sum(exp.(attn.w))
    out = zeros(no, B)
    for t in 1:T
        out .+= S[:, :, t] .* alpha[t]
    end
    return out, alpha
end

mutable struct GestureSNNv4
    estc::ConvLIFBlock
    lif1::LIFBlock
    lif2::LIFBlock
    attn::TemporalAttention
    head::SpikeDense
    training::Bool
end

function GestureSNNv4()
    estc = ConvLIFBlock(2, 16, 3; pad=1) # 2x32x32 -> 16x32x32
    # We pool 2x2 manually in fwd
    # 16x16x16 = 4096
    lif1 = LIFBlock(4096, 256)
    lif2 = LIFBlock(256, 128)
    GestureSNNv4(estc, lif1, lif2, TemporalAttention(SNN_T_STEPS), SpikeDense(128, 11), true)
end

function forward_v4!(net::GestureSNNv4, X::Matrix{Float64})
    B = size(X, 2)
    # X is (2048*20, B). Reshape to (B, 2, 32, 32, 20)
    X_seq = reshape(X, 2, DVS_RES_DOWN, DVS_RES_DOWN, SNN_T_STEPS, B)
    X_seq = permutedims(X_seq, (5, 1, 2, 3, 4)) .* 5.0 # (B, C, H, W, T)
    
    s_conv = fwd_conv_lif!(net.estc, X_seq; training=net.training)
    
    # Spiking Max Pool 2x2
    B, C, H, W, T = size(s_conv)
    s_pool = zeros(B, C, H÷2, W÷2, T)
    for t in 1:T, c in 1:C
        for i in 1:H÷2, j in 1:W÷2
            s_pool[:, c, i, j, t] = maximum(s_conv[:, c, 2i-1:2i, 2j-1:2j, t], dims=(2,3))
        end
    end
    
    h_flat = reshape(permutedims(s_pool, (2,3,4,1,5)), C*(H÷2)*(W÷2), B, T)
    
    h1 = fwd_lif!(net.lif1, h_flat; training=net.training)
    h2 = fwd_lif!(net.lif2, h1; training=net.training)
    
    readout, _ = apply_attention(net.attn, h2)
    return softmax_s(net.head.W * readout .+ net.head.b)
end

function backward_v4!(net::GestureSNNv4, y_oh, y_pred, weights; lr=1e-3, l2=1e-5, clip=1.0)
    B = size(y_oh, 2)
    d = (y_pred .- y_oh)
    # Focal Loss Gradient Approximation (gamma = 2.0)
    for i in 1:B
        target_idx = argmax(y_oh[:, i])
        p_t = y_pred[target_idx, i]
        focal_weight = (1.0 - p_t)^2.0
        d[:, i] .*= weights[target_idx] * focal_weight
    end
    
    readout, alpha = apply_attention(net.attn, net.lif2.S)
    net.head.dW = d * readout' ./ B .+ l2 .* net.head.W
    net.head.db = vec(sum(d, dims=2)) ./ B
    
    d_readout = net.head.W' * d
    
    # Attention Grad
    no, _, T = size(net.lif2.S)
    for t in 1:T
        net.attn.dw[t] = sum(d_readout .* net.lif2.S[:, :, t]) / B
    end
    
    # LIF Stack Backprop
    dS2 = zeros(no, B, T)
    for t in 1:T; dS2[:, :, t] = d_readout .* alpha[t]; end
    
    dS1 = bwd_lif!(net.lif2, dS2, l2, clip)
    dS_pool = bwd_lif!(net.lif1, dS1, l2, clip)
    
    # Pool Backward (Unpooling)
    B_sz = B; OC = net.estc.conv.out_ch; H = DVS_RES_DOWN; W = DVS_RES_DOWN
    dS_conv = zeros(B_sz, OC, H, W, T)
    # Simple broadcast unpooling for spikes
    for t in 1:T, c in 1:OC, i in 1:H÷2, j in 1:W÷2
        val = dS_pool[(c-1)*(H÷2)*(W÷2) + (i-1)*(W÷2) + j, :, t]
        dS_conv[:, c, 2i-1, 2j-1, t] .= val ./ 4.0
        dS_conv[:, c, 2i, 2j-1, t] .= val ./ 4.0
        dS_conv[:, c, 2i-1, 2j, t] .= val ./ 4.0
        dS_conv[:, c, 2i, 2j, t] .= val ./ 4.0
    end
    
    bwd_conv_lif!(net.estc, dS_conv, l2, clip)
end

function adam_step_v4!(net::GestureSNNv4, lr, t)
    bc1, bc2 = 1.0-0.9^t, 1.0-0.999^t
    function upd!(l, dW, db, mW, vW, mb, vb)
        @. mW = 0.9*mW + 0.1*dW; @. vW = 0.999*vW + 0.001*dW^2
        @. l.W -= lr*(mW/bc1)/(sqrt(vW/bc2)+1e-8)
        @. mb = 0.9*mb + 0.1*db; @. vb = 0.999*vb + 0.001*db^2
        @. l.b -= lr*(mb/bc1)/(sqrt(vb/bc2)+1e-8)
    end
    
    upd!(net.head, net.head.dW, net.head.db, net.head.mW, net.head.vW, net.head.mb, net.head.vb)
    upd!(net.lif2.layer, net.lif2.layer.dW, net.lif2.layer.db, net.lif2.layer.mW, net.lif2.layer.vW, net.lif2.layer.mb, net.lif2.layer.vb)
    upd!(net.lif1.layer, net.lif1.layer.dW, net.lif1.layer.db, net.lif1.layer.mW, net.lif1.layer.vW, net.lif1.layer.mb, net.lif1.layer.vb)
    
    c = net.estc.conv
    @. c.mW = 0.9*c.mW + 0.1*c.dW; @. c.vW = 0.999*c.vW + 0.001*c.dW^2
    @. c.W -= lr*(c.mW/bc1)/(sqrt(c.vW/bc2)+1e-8)
    @. c.mb = 0.9*c.mb + 0.1*c.db; @. c.vb = 0.999*c.vb + 0.001*c.db^2
    @. c.b -= lr*(c.mb/bc1)/(sqrt(c.vb/bc2)+1e-8)
end

function run_dvs_v4()
    println("\n" * "█"^80)
    println("   NEUROMORPHIC GESTURE RECOGNITION — SNN v4 (ESTC Architecture)")
    println("   Spatio-Temporal Convolutions | Spiking Max Pool | Temporal Attention")
    println("█"^80)

    train_list = read_trial_list(joinpath(DVS_DATA_DIR, "trials_to_train.txt"))
    test_list = read_trial_list(joinpath(DVS_DATA_DIR, "trials_to_test.txt"))

    println("\n  Loading training trials ($(length(train_list)) files)...")
    X_train, y_train = load_split(train_list, SNN_T_STEPS)
    println("  Loading test trials ($(length(test_list)) files)...")
    X_test, y_test = load_split(test_list, SNN_T_STEPS)
    
    # Weights for classes
    counts = zeros(11)
    for l in y_train; counts[l] += 1; end
    weights = sum(counts) ./ (11 .* max.(counts, 1.0))

    epochs = 120
    batch_size = 16
    n = length(y_train)
    net = GestureSNNv4()
    
    println("\n  Dataset: $n train, $(length(y_test)) test")
    println("  Architecture: ESTC(2->16) -> Pool(2x2) -> Dense(256) -> Dense(128) -> Head(11)")
    println("  " * "-"^70)

    best_acc = 0.0
    step = 0

    for ep in 1:epochs
        lr = 1e-3 * 0.5 * (1 + cos(pi * ep / epochs))
        perm = randperm(n)
        net.training = true
        
        for i in 1:batch_size:n
            step += 1
            idx = perm[i:min(i+batch_size-1, n)]
            X_b = X_train[:, idx]
            y_b = y_train[idx]
            
            y_oh = zeros(11, length(y_b))
            for (j, l) in enumerate(y_b); y_oh[l, j] = 1.0; end
            
            y_pred = forward_v4!(net, X_b)
            backward_v4!(net, y_oh, y_pred, weights; lr=lr)
            adam_step_v4!(net, lr, step)
        end
        
        # Eval
        if ep == 1 || ep % 5 == 0 || ep == epochs
            net.training = false
            y_pred_te = forward_v4!(net, X_test)
            preds_te = [argmax(y_pred_te[:, i]) for i in 1:size(y_pred_te, 2)]
            acc = mean(preds_te .== y_test)
            
            improved = ""
            if acc > best_acc
                best_acc = acc
                improved = " *"
            end
            @printf("  Epoch %2d | Test Acc: %5.2f%%%s\n", ep, acc*100, improved)
        end
    end
end

run_dvs_v4()
