using Serialization
include("../../src/dvs_loader.jl")

const SURROGATE_SCALE = 8.0 # Softened for precision refinement
const V_THRESHOLD = 0.75
const V_LEAK = 0.93 # Increased for better temporal memory
const SNN_T_STEPS = DVS_TIME_BINS
const SPIKE_DROPOUT = 0.10 # Reduced for final stage stability

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
    B, IC, H, W_in = size(X)
    OC, _, k, _ = size(W)
    H_out = H + 2pad - k + 1
    W_out = W_in + 2pad - k + 1
    
    X_pad = zeros(B, IC, H + 2pad, W_in + 2pad)
    X_pad[:, :, pad+1:H+pad, pad+1:W_in+pad] .= X
    
    out = zeros(B, OC, H_out, W_out)
    
    # Reordered for better cache locality (B and OC last)
    for row in 1:H_out, col in 1:W_out
        for i in 1:k, j in 1:k
            @inbounds for ic in 1:IC
                x_val = @view X_pad[:, ic, row+i-1, col+j-1]
                for oc in 1:OC
                    w_val = W[oc, ic, i, j]
                    @simd for b_idx in 1:B
                        out[b_idx, oc, row, col] += x_val[b_idx] * w_val
                    end
                end
            end
        end
    end
    
    @inbounds for oc in 1:OC
        val = b[oc]
        for row in 1:H_out, col in 1:W_out, b_idx in 1:B
            out[b_idx, oc, row, col] += val
        end
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
    
    # We need to return dX sequence
    dX_seq = zeros(size(l.inp))
    
    for t in 1:T
        dw, db, dx = conv2d_grad(l.inp[:, :, :, :, t], l.W, d_out_seq[:, :, :, :, t], l.pad)
        l.dW .+= dw ./ B
        l.db .+= db ./ B
        dX_seq[:, :, :, :, t] = dx
    end
    
    l.dW .+= lambda .* l.W
    gnorm = sqrt(sum(l.dW.^2) + sum(l.db.^2))
    if gnorm > clip
        sc = clip / gnorm; l.dW .*= sc; l.db .*= sc
    end
    return dX_seq
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
    
    # Generate current by running conv
    conv_out = zeros(B, OC, H, W, T)
    for t in 1:T
        conv_out[:, :, :, :, t] = conv2d(X_seq[:, :, :, :, t], blk.conv.W, blk.conv.b, blk.conv.pad)
    end
    
    blk.U = zeros(B, OC, H, W, T + 1)
    blk.S = zeros(B, OC, H, W, T)
    
    H, W = size(conv_out, 3), size(conv_out, 4)
    OC = blk.conv.out_ch
    
    # Optimized LIF with Soft Reset and Vectorized Dropout
    v_th = blk.v_th
    @inbounds for t in 1:T
        # Potential accumulation
        for b in 1:B, c in 1:OC, i in 1:H, j in 1:W
            blk.U[b, c, i, j, t+1] = blk.U[b, c, i, j, t] * V_LEAK + conv_out[b, c, i, j, t] * (1.0 - V_LEAK)
        end
        
        # Thresholding
        spikes = Float64.(blk.U[:, :, :, :, t+1] .> v_th)
        
        # Apply Dropout during training
        if training
            mask = rand(B, OC, H, W) .> SPIKE_DROPOUT
            spikes .*= mask
        end
        
        blk.S[:, :, :, :, t] .= spikes
        
        # Soft Reset (Subtract threshold)
        for b in 1:B, c in 1:OC, i in 1:H, j in 1:W
            blk.U[b, c, i, j, t+1] -= spikes[b, c, i, j] * v_th
        end
        
        # Adaptive Threshold (Heaviside Approximation)
        if training
            rate = sum(@view blk.S[:, :, :, :, t]) / (B * OC * H * W)
            if rate < 0.01; blk.v_th *= 0.98; elseif rate > 0.1; blk.v_th *= 1.02; end
            blk.v_th = clamp(blk.v_th, 0.2, 2.0)
        end
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
    LIFBlock(SpikeDense(ni, no), zeros(0,0,0), zeros(0,0,0), V_THRESHOLD)
end

function fwd_lif!(blk::LIFBlock, X_seq::Array{Float64, 3}; training=false)
    no, B, T = size(X_seq)
    blk.layer.inp = X_seq
    blk.U = zeros(no, B, T+1)
    blk.S = zeros(no, B, T)
    
    v_th = blk.v_th
    @inbounds for t in 1:T
        # Potential accumulation
        for b in 1:B, n in 1:no
            blk.U[n, b, t+1] = blk.U[n, b, t] * V_LEAK + X_seq[n, b, t] * (1.0 - V_LEAK)
        end
        
        # Thresholding
        spikes = Float64.(blk.U[:, :, t+1] .> v_th)
        
        if training
            mask = rand(no, B) .> SPIKE_DROPOUT
            spikes .*= mask
        end
        
        blk.S[:, :, t] .= spikes
        
        # Soft Reset
        for b in 1:B, n in 1:no
            blk.U[n, b, t+1] -= spikes[n, b] * v_th
        end
        
        if training
            rate = sum(@view blk.S[:, :, t]) / (no * B)
            if rate < 0.01; blk.v_th *= 0.98; elseif rate > 0.1; blk.v_th *= 1.02; end
            blk.v_th = clamp(blk.v_th, 0.2, 2.0)
        end
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

mutable struct GestureSNNv6
    estc1::ConvLIFBlock
    estc2::ConvLIFBlock
    lif1::LIFBlock
    lif2::LIFBlock
    attn::TemporalAttention
    head::SpikeDense
    training::Bool
end

function GestureSNNv6()
    # Path 1: High-Res Spatial (2 -> 32)
    estc1 = ConvLIFBlock(2, 32, 3; pad=1)
    # Path 2: Deep Feature (32 -> 64)
    estc2 = ConvLIFBlock(32, 64, 3; pad=1)
    
    # After two 2x2 pools, 32x32 -> 8x8
    # 64 * 8 * 8 = 4096
    lif1 = LIFBlock(4096, 512)
    lif2 = LIFBlock(512, 256)
    
    GestureSNNv6(estc1, estc2, lif1, lif2, TemporalAttention(SNN_T_STEPS), SpikeDense(256, 11), true)
end

function forward_v6!(net::GestureSNNv6, X::Matrix{Float64})
    B = size(X, 2)
    X_seq = reshape(X, 2, DVS_RES_DOWN, DVS_RES_DOWN, SNN_T_STEPS, B)
    X_seq = permutedims(X_seq, (5, 1, 2, 3, 4)) .* 10.0 # Signal Boost
    
    # Stage 1
    s_conv1 = fwd_conv_lif!(net.estc1, X_seq; training=net.training)
    # Max Pool 1
    B, C1, H1, W1, T = size(s_conv1)
    s_pool1 = zeros(B, C1, H1÷2, W1÷2, T)
    for t in 1:T, c in 1:C1, i in 1:H1÷2, j in 1:W1÷2
        s_pool1[:, c, i, j, t] = maximum(s_conv1[:, c, 2i-1:2i, 2j-1:2j, t], dims=(2,3))
    end
    
    # Stage 2
    s_conv2 = fwd_conv_lif!(net.estc2, s_pool1; training=net.training)
    # Max Pool 2
    B, C2, H2, W2, T = size(s_conv2)
    s_pool2 = zeros(B, C2, H2÷2, W2÷2, T)
    for t in 1:T, c in 1:C2, i in 1:H2÷2, j in 1:W2÷2
        s_pool2[:, c, i, j, t] = maximum(s_conv2[:, c, 2i-1:2i, 2j-1:2j, t], dims=(2,3))
    end
    
    h_flat = reshape(permutedims(s_pool2, (2,3,4,1,5)), C2*(H2÷2)*(W2÷2), B, T)
    
    h1 = fwd_lif!(net.lif1, h_flat; training=net.training)
    h2 = fwd_lif!(net.lif2, h1; training=net.training)
    
    readout, _ = apply_attention(net.attn, h2)
    return softmax_s(net.head.W * readout .+ net.head.b)
end

function backward_v6!(net::GestureSNNv6, y_oh, y_pred, weights; lr=1e-3, l2=1e-5, clip=1.0)
    B = size(y_oh, 2)
    d = (y_pred .- y_oh)
    # Focal Loss (gamma=2.0)
    for i in 1:B
        target_idx = argmax(y_oh[:, i])
        p_t = y_pred[target_idx, i]
        d[:, i] .*= weights[target_idx] * (1.0 - p_t)^2.0
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
    
    # LIF Stack
    dS2 = zeros(no, B, T)
    for t in 1:T; dS2[:, :, t] = d_readout .* alpha[t]; end
    
    dS1 = bwd_lif!(net.lif2, dS2, l2, clip)
    dS_pool2 = bwd_lif!(net.lif1, dS1, l2, clip)
    
    # Pool 2 Backward
    B_sz, OC2, H2, W2 = B, net.estc2.conv.out_ch, DVS_RES_DOWN÷2, DVS_RES_DOWN÷2
    dS_conv2 = zeros(B_sz, OC2, H2, W2, T)
    for t in 1:T, c in 1:OC2, i in 1:H2÷2, j in 1:W2÷2
        val = dS_pool2[(c-1)*(H2÷2)*(W2÷2) + (i-1)*(W2÷2) + j, :, t]
        dS_conv2[:, c, 2i-1, 2j-1, t] .= val ./ 4.0
        dS_conv2[:, c, 2i, 2j-1, t] .= val ./ 4.0
        dS_conv2[:, c, 2i-1, 2j, t] .= val ./ 4.0
        dS_conv2[:, c, 2i, 2j, t] .= val ./ 4.0
    end
    dS_pool1 = bwd_conv_lif!(net.estc2, dS_conv2, l2, clip)
    
    # Pool 1 Backward
    H1, W1 = DVS_RES_DOWN, DVS_RES_DOWN
    OC1 = net.estc1.conv.out_ch
    dS_conv1 = zeros(B_sz, OC1, H1, W1, T)
    for t in 1:T, c in 1:OC1, i in 1:H1÷2, j in 1:W1÷2
        val = dS_pool1[:, c, i, j, t]
        dS_conv1[:, c, 2i-1, 2j-1, t] .= val ./ 4.0
        dS_conv1[:, c, 2i, 2j-1, t] .= val ./ 4.0
        dS_conv1[:, c, 2i-1, 2j, t] .= val ./ 4.0
        dS_conv1[:, c, 2i, 2j, t] .= val ./ 4.0
    end
    bwd_conv_lif!(net.estc1, dS_conv1, l2, clip)
end

function adam_step_v6!(net::GestureSNNv6, lr, t)
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
    
    # Conv Layers
    for (blk, l) in [(net.estc1, net.estc1.conv), (net.estc2, net.estc2.conv)]
        @. l.mW = 0.9*l.mW + 0.1*l.dW; @. l.vW = 0.999*l.vW + 0.001*l.dW^2
        @. l.W -= lr*(l.mW/bc1)/(sqrt(l.vW/bc2)+1e-8)
        @. l.mb = 0.9*l.mb + 0.1*l.db; @. l.vb = 0.999*l.vb + 0.001*l.db^2
        @. l.b -= lr*(l.mb/bc1)/(sqrt(l.vb/bc2)+1e-8)
    end
end

function run_dvs_omega_v6()
    println("\n" * "█"^80)
    println("   NEUROMORPHIC GESTURE RECOGNITION — OMEGA v6 (DSTE Architecture)")
    println("   Dual-Conv Path (32/64) | Spiking Max Pool | Multi-Layer LIF | Attention")
    println("█"^80)

    train_list = read_trial_list(joinpath(DVS_DATA_DIR, "trials_to_train.txt"))
    test_list = read_trial_list(joinpath(DVS_DATA_DIR, "trials_to_test.txt"))

    println("\n  Loading training trials ($(length(train_list)) files)...")
    X_train, y_train = load_split(train_list, SNN_T_STEPS)
    println("  Loading test trials ($(length(test_list)) files)...")
    X_test, y_test = load_split(test_list, SNN_T_STEPS)
    
    counts = zeros(11)
    for l in y_train; counts[l] += 1; end
    weights = sum(counts) ./ (11 .* max.(counts, 1.0))

    epochs = 120
    batch_size = 12 
    n = length(y_train)
    net = GestureSNNv6()
    
    best_acc = 0.0
    step = 0
    start_epoch = 1
    checkpoint_path = "dvs_omega_v6_checkpoint.jls"

    if isfile(checkpoint_path)
        println("  [CHECKPOINT] Found existing state. Resuming from disk...")
        try
            checkpoint = deserialize(checkpoint_path)
            net = checkpoint.net
            best_acc = checkpoint.best_acc
            step = checkpoint.step
            start_epoch = checkpoint.epoch + 1
            println("  [CHECKPOINT] Resumed at Epoch $start_epoch | Best Acc: $(round(best_acc*100, digits=2))%")
        catch e
            println("  [WARNING] Failed to load checkpoint: $e. Starting fresh.")
        end
    end

    println("\n  Dataset: $n train, $(length(y_test)) test")
    println("  Architecture: Conv(2->32) -> Pool -> Conv(32->64) -> Pool -> Dense(512) -> Dense(256) -> Head(11)")
    println("  " * "-"^70)

    for ep in start_epoch:epochs
        lr = 0.0001 + 0.5 * (2e-3 - 0.0001) * (1 + cos(pi * ep / epochs))
        perm = randperm(n)
        net.training = true
        
        for i in 1:batch_size:n
            step += 1
            idx = perm[i:min(i+batch_size-1, n)]
            X_b = X_train[:, idx]
            y_b = y_train[idx]
            
            y_oh = zeros(11, length(y_b))
            for (j, l) in enumerate(y_b); y_oh[l, j] = 1.0; end
            
            y_pred = forward_v6!(net, X_b)
            backward_v6!(net, y_oh, y_pred, weights; lr=lr)
            adam_step_v6!(net, lr, step)
        end
        
        if ep == 1 || ep % 2 == 0 || ep == epochs
            net.training = false
            y_pred_te = forward_v6!(net, X_test)
            preds_te = [argmax(y_pred_te[:, i]) for i in 1:size(y_pred_te, 2)]
            acc = mean(preds_te .== y_test)
            
            improved = ""
            if acc > best_acc
                best_acc = acc
                improved = " *"
            end
            @printf("  Epoch %2d | Test Acc: %5.2f%%%s | LR: %7.5f\n", ep, acc*100, improved, lr)
            
            # Save Checkpoint
            serialize(checkpoint_path, (net=net, best_acc=best_acc, step=step, epoch=ep))
        end
    end
end

run_dvs_omega_v6()
