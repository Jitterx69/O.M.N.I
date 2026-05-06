include("../src/dvs_loader.jl")

const SURROGATE_SCALE = 5.0
const V_THRESHOLD = 1.0
const V_LEAK = 0.9
const SNN_T_STEPS = DVS_TIME_BINS
const SPIKE_DROPOUT = 0.2

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
end

function SpikeConv(in_ch, out_ch, k; pad=1)
    sc = sqrt(2.0 / (in_ch * k * k))
    SpikeConv(in_ch, out_ch, k, pad,
        randn(out_ch, in_ch, k, k) .* sc, zeros(out_ch),
        zeros(out_ch, in_ch, k, k), zeros(out_ch),
        zeros(out_ch, in_ch, k, k), zeros(out_ch, in_ch, k, k),
        zeros(out_ch), zeros(out_ch))
end

function conv2d(X::Array{Float64, 4}, W::Array{Float64, 4}, b::Vector{Float64}, pad::Int)
    B, C, H, W_in = size(X)
    OC, IC, k, _ = size(W)
    H_out = H + 2*pad - k + 1
    W_out = W_in + 2*pad - k + 1
    
    out = zeros(B, OC, H_out, W_out)
    
    # Fast im2col-like approach or padded loop
    X_pad = zeros(B, C, H + 2*pad, W_in + 2*pad)
    X_pad[:, :, pad+1:H+pad, pad+1:W_in+pad] = X
    
    for oc in 1:OC
        for ic in 1:IC
            for i in 1:k, j in 1:k
                @views out[:, oc, :, :] .+= W[oc, ic, i, j] .* X_pad[:, ic, i:i+H_out-1, j:j+W_out-1]
            end
        end
        out[:, oc, :, :] .+= b[oc]
    end
    return out
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
    blk.inp = X_seq
    
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

function backward_v4!(net::GestureSNNv4, y_oh, y_pred, weights; lr=1e-3)
    # Simplified BPTT for Conv-SNN v4
    # Just updating Head for now to verify flow, full BWD in production
    B = size(y_oh, 2)
    d = (y_pred .- y_oh)
    for i in 1:B; d[:, i] .*= weights[argmax(y_oh[:, i])]; end
    
    readout, _ = apply_attention(net.attn, net.lif2.S)
    net.head.dW = d * readout' ./ B
    net.head.db = vec(sum(d, dims=2)) ./ B
    
    # Adam Head
    @. net.head.mW = 0.9*net.head.mW + 0.1*net.head.dW
    @. net.head.vW = 0.999*net.head.vW + 0.001*net.head.dW^2
    @. net.head.W -= lr * (net.head.mW/0.9) / (sqrt(net.head.vW/0.999) + 1e-8)
end

function run_dvs_v4()
    println("\n" * "█"^80)
    println("   NEUROMORPHIC GESTURE RECOGNITION — SNN v4 (ESTC Architecture)")
    println("   Spatio-Temporal Convolutions | Spiking Max Pool | Temporal Attention")
    println("█"^80)

    train_data, test_data = load_dvs_gestures()
    net = GestureSNNv4()
    
    # Weights for classes
    counts = zeros(11)
    for (_, l) in train_data; counts[l] += 1; end
    weights = sum(counts) ./ (11 .* counts)

    epochs = 100
    batch_size = 16
    
    for ep in 1:epochs
        shuffle!(train_data)
        for i in 1:batch_size:length(train_data)
            batch = train_data[i:min(i+batch_size-1, end)]
            X_b = hcat([b[1] for b in batch]...)
            y_b = [b[2] for b in batch]
            y_oh = zeros(11, length(y_b))
            for (j, l) in enumerate(y_b); y_oh[l, j] = 1.0; end
            
            net.training = true
            y_pred = forward_v4!(net, X_b)
            backward_v4!(net, y_oh, y_pred, weights)
        end
        
        # Eval
        if ep % 5 == 0
            net.training = false
            correct = 0
            for (x, l) in test_data
                y_p = forward_v4!(net, reshape(x, :, 1))
                if argmax(y_p) == l; correct += 1; end
            end
            @printf("  Epoch %2d | Test Acc: %.2f%%\n", ep, correct/length(test_data)*100)
        end
    end
end

run_dvs_v4()
