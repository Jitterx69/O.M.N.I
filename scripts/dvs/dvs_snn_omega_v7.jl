using Serialization, Statistics, Random, Printf, Dates, LinearAlgebra
include("../../src/dvs_loader.jl")

# --- Constants & Hyperparameters ---
const SURROGATE_SCALE = 10.0
const V_THRESHOLD = 1.0
const V_LEAK = 0.9
const SNN_T_STEPS = 20
const SPIKE_DROPOUT = 0.05  # Only for Dense LIF
const LABEL_SMOOTHING = 0.1
const DVS_RES_DOWN = 32

# --- Model Components ---

mutable struct tdBN
    gamma::Vector{Float64}; beta::Vector{Float64}
    d_gamma::Vector{Float64}; d_beta::Vector{Float64}
    m_gamma::Vector{Float64}; v_gamma::Vector{Float64}
    m_beta::Vector{Float64}; v_beta::Vector{Float64}
    running_mu::Vector{Float64}; running_var::Vector{Float64}
    # Saved for backward per timestep
    mu_t::Array{Float64, 2}; var_t::Array{Float64, 2} 
    x_hat::Array{Float64, 5}
    eps::Float64
end

function tdBN(C)
    tdBN(ones(C), zeros(C), zeros(C), zeros(C), zeros(C), zeros(C), zeros(C), zeros(C),
         zeros(C), ones(C), zeros(C, SNN_T_STEPS), zeros(C, SNN_T_STEPS), zeros(0,0,0,0,0), 1e-5)
end

function apply_tdbn!(bn::tdBN, X::Array{Float64, 5}, training::Bool, V_th::Float64)
    B, C, H, W, T = size(X)
    out = similar(X)
    if training
        bn.x_hat = similar(X)
        for t in 1:T
            mu = mean(X[:, :, :, :, t], dims=(1, 3, 4))
            v_val = var(X[:, :, :, :, t], dims=(1, 3, 4), corrected=false)
            bn.mu_t[:, t] .= vec(mu)
            bn.var_t[:, t] .= vec(v_val)
            
            # tdBN scaling: input current normalized such that mean=0, std=1
            std = sqrt.(v_val .+ bn.eps)
            bn.x_hat[:, :, :, :, t] .= (X[:, :, :, :, t] .- mu) ./ std
            out[:, :, :, :, t] .= V_th .* bn.gamma .* bn.x_hat[:, :, :, :, t] .+ bn.beta
            
            bn.running_mu .= 0.9 .* bn.running_mu .+ 0.1 .* vec(mu)
            bn.running_var .= 0.9 .* bn.running_var .+ 0.1 .* vec(v_val)
        end
    else
        for t in 1:T
            std = sqrt.(bn.running_var .+ bn.eps)
            x_hat = (X[:, :, :, :, t] .- reshape(bn.running_mu, 1, C, 1, 1)) ./ reshape(std, 1, C, 1, 1)
            out[:, :, :, :, t] .= V_th .* reshape(bn.gamma, 1, C, 1, 1) .* x_hat .+ reshape(bn.beta, 1, C, 1, 1)
        end
    end
    return out
end

function bwd_tdbn!(bn::tdBN, d_out::Array{Float64, 5}, V_th::Float64)
    B, C, H, W, T = size(d_out)
    N = B * H * W
    dX = similar(d_out)
    bn.d_gamma .= 0.0; bn.d_beta .= 0.0
    
    for t in 1:T
        std = sqrt.(bn.var_t[:, t] .+ bn.eps)
        gamma = reshape(bn.gamma, 1, C, 1, 1)
        
        # Gradients for gamma and beta
        dg = sum(d_out[:, :, :, :, t] .* V_th .* bn.x_hat[:, :, :, :, t], dims=(1, 3, 4))
        db = sum(d_out[:, :, :, :, t], dims=(1, 3, 4))
        bn.d_gamma .+= vec(dg)
        bn.d_beta .+= vec(db)
        
        # Gradient for X (Input) - Standard BN Jacobian
        dx_hat = d_out[:, :, :, :, t] .* V_th .* gamma
        d_var = sum(dx_hat .* (bn.x_hat[:, :, :, :, t] .* -0.5 ./ (std.^2)), dims=(1, 3, 4))
        d_mu = sum(dx_hat .* (-1.0 ./ std), dims=(1, 3, 4)) .+ d_var .* mean(-2.0 .* (bn.x_hat[:, :, :, :, t] .* std), dims=(1,3,4))
        
        dX[:, :, :, :, t] .= (dx_hat ./ std) .+ (d_var .* 2.0 .* (bn.x_hat[:, :, :, :, t] .* std) ./ N) .+ (d_mu ./ N)
    end
    return dX
end

mutable struct SpikeConv
    W::Array{Float64, 4}; b::Vector{Float64}
    dW::Array{Float64, 4}; db::Vector{Float64}
    mW::Array{Float64, 4}; vW::Array{Float64, 4}
    mb::Vector{Float64}; vb::Vector{Float64}
    pad::Int; X_col::Array{Float64, 2}; X_pad::Array{Float64, 4}
end

function SpikeConv(in_c, out_c, k; pad=1)
    sc = sqrt(2.0 / (in_c * k * k))
    SpikeConv(randn(out_c, in_c, k, k) .* sc, zeros(out_c),
              zeros(out_c, in_c, k, k), zeros(out_c),
              zeros(out_c, in_c, k, k), zeros(out_c, in_c, k, k),
              zeros(out_c), zeros(out_c),
              pad, zeros(0,0), zeros(0,0,0,0))
end

# im2col implementation for convolution
function conv2d!(out, X, W, b, pad, X_col, X_pad)
    OC, IC, k, _ = size(W)
    B, _, H, W_in = size(X)
    H_out = H + 2pad - k + 1
    W_out = W_in + 2pad - k + 1
    
    # Lazy buffer allocation
    if size(X_pad) != (B, IC, H+2pad, W_in+2pad)
        X_pad = zeros(B, IC, H+2pad, W_in+2pad)
    else
        X_pad .= 0.0
    end
    X_pad[:, :, pad+1:pad+H, pad+1:pad+W_in] .= X
    
    actual_col_size = B * H_out * W_out
    if size(X_col) != (IC*k*k, actual_col_size)
        X_col = zeros(IC*k*k, actual_col_size)
    end
    
    col_idx = 1
    for b_idx in 1:B, i in 1:H_out, j in 1:W_out
        for ic in 1:IC, ki in 1:k, kj in 1:k
            X_col[(ic-1)*k*k + (ki-1)*k + kj, col_idx] = X_pad[b_idx, ic, i+ki-1, j+kj-1]
        end
        col_idx += 1
    end
    
    W_mat = reshape(W, OC, IC * k * k)
    out_mat = W_mat * X_col .+ b
    out .= reshape(out_mat, OC, H_out, W_out, B)
    out .= permutedims(out, (4, 1, 2, 3)) # (B, OC, H, W)
    return X_col, X_pad
end

function conv2d_grad(d_out, X_col, W, pad, IC, k)
    B, OC, H_out, W_out = size(d_out)
    # Match X_col order: (B fastest, then H, then W)
    d_out_mat = reshape(permutedims(d_out, (2, 1, 3, 4)), OC, :)
    
    # d_out_mat already contains 1/B from the loss d
    # So we only normalize by spatial size (H*W)
    spatial_size = H_out * W_out
    dW_mat = (d_out_mat * X_col') ./ spatial_size
    dW = reshape(dW_mat, OC, IC, k, k)
    db = vec(sum(d_out_mat, dims=2)) ./ spatial_size
    
    W_mat = reshape(W, OC, IC * k * k)
    dX_col = W_mat' * d_out_mat
    return dW, db, dX_col
end

mutable struct ConvLIFBlock
    conv::SpikeConv; bn::tdBN
    U::Array{Float64, 5}; S::Array{Float64, 5}
    inp_raw::Array{Float64, 5}
end

function ConvLIFBlock(in_c, out_c, k; pad=1)
    ConvLIFBlock(SpikeConv(in_c, out_c, k; pad=pad), tdBN(out_c), zeros(0,0,0,0,0), zeros(0,0,0,0,0), zeros(0,0,0,0,0))
end

function fwd_conv_lif!(blk::ConvLIFBlock, X::Array{Float64, 5}; training=false)
    B, IC, H, W, T = size(X)
    OC = size(blk.conv.W, 1)
    k = blk.conv.pad; pad = blk.conv.pad
    H_out = H + 2pad - size(blk.conv.W,3) + 1
    W_out = W + 2pad - size(blk.conv.W,4) + 1
    
    blk.U = zeros(B, OC, H_out, W_out, T + 1)
    blk.S = zeros(B, OC, H_out, W_out, T)
    blk.inp_raw = X
    
    # 1. Convolution
    I_conv = zeros(B, OC, H_out, W_out, T)
    for t in 1:T
        blk.conv.X_col, blk.conv.X_pad = conv2d!(view(I_conv, :, :, :, :, t), X[:, :, :, :, t], blk.conv.W, blk.conv.b, blk.conv.pad, blk.conv.X_col, blk.conv.X_pad)
    end
    
    # 2. tdBN (Applied to current before LIF)
    I_bn = apply_tdbn!(blk.bn, I_conv, training, V_THRESHOLD)
    
    # 3. LIF Interleaved Loop (Accumulate, Threshold, Soft-Reset)
    for t in 1:T
        @. blk.U[:, :, :, :, t+1] = blk.U[:, :, :, :, t] * V_LEAK + I_bn[:, :, :, :, t]
        @. blk.S[:, :, :, :, t] = Float64(blk.U[:, :, :, :, t+1] > V_THRESHOLD)
        @. blk.U[:, :, :, :, t+1] = clamp(blk.U[:, :, :, :, t+1] - blk.S[:, :, :, :, t] * V_THRESHOLD, -1.0, V_THRESHOLD)
    end
    return blk.S
end

surrogate_grad(x) = 1.0 / (1.0 + SURROGATE_SCALE * abs(x))^2

function bwd_conv_lif!(blk::ConvLIFBlock, dS::Array{Float64, 5}, lambda::Float64, clip::Float64)
    B, OC, H, W, T = size(dS)
    dU = zeros(B, OC, H, W)
    dI_bn = zeros(B, OC, H, W, T)
    for t in T:-1:1
        # Surrogate gating only for spike path
        grad_s = surrogate_grad.(blk.U[:, :, :, :, t+1] .- V_THRESHOLD)
        dU_spike = dS[:, :, :, :, t] .* grad_s
        dU_curr = dU_spike .+ dU .* V_LEAK
        dI_bn[:, :, :, :, t] .= dU_curr
        dU = dU_curr
    end
    
    # Backprop through tdBN
    dI_conv = bwd_tdbn!(blk.bn, dI_bn, V_THRESHOLD)
    
    # Backprop through Conv
    dX = zeros(size(blk.inp_raw))
    blk.conv.dW .= 0.0; blk.conv.db .= 0.0
    for t in T:-1:1
        # Need X_col for each timestep? Trace: conv2d! saves it.
        # But we need it for EACH t. Re-running im2col is safer for BPTT fidelity.
        # Actually, let's just re-generate X_col for t in this loop.
        H_in, W_in = size(blk.inp_raw, 3), size(blk.inp_raw, 4)
        pad = blk.conv.pad; IC = size(blk.conv.W, 2); k = size(blk.conv.W, 3)
        H_out, W_out = H, W
        
        X_p = zeros(B, IC, H_in+2pad, W_in+2pad)
        X_p[:, :, pad+1:pad+H_in, pad+1:pad+W_in] .= blk.inp_raw[:, :, :, :, t]
        X_c = zeros(IC*k*k, B*H_out*W_out)
        c_idx = 1
        for bi in 1:B, i in 1:H_out, j in 1:W_out
            for ic in 1:IC, ki in 1:k, kj in 1:k
                X_c[(ic-1)*k*k + (ki-1)*k + kj, c_idx] = X_p[bi, ic, i+ki-1, j+kj-1]
            end
            c_idx += 1
        end
        
        tdW, tdb, dXc = conv2d_grad(dI_conv[:, :, :, :, t], X_c, blk.conv.W, pad, IC, k)
        blk.conv.dW .+= tdW ./ T
        blk.conv.db .+= tdb ./ T
        
        # dX_col to spatial dX (with pad offset check)
        d_out_mat = reshape(dXc, IC, k, k, B, H_out, W_out)
        for bi in 1:B, i in 1:H_out, j in 1:W_out
            for ic in 1:IC, ki in 1:k, kj in 1:k
                ri, rj = i + ki - 1 - pad, j + kj - 1 - pad
                if 1 <= ri <= H_in && 1 <= rj <= W_in
                    dX[bi, ic, ri, rj, t] += d_out_mat[ic, ki, kj, bi, i, j]
                end
            end
        end
    end
    # L2
    blk.conv.dW .+= lambda .* blk.conv.W
    # Clip
    clamp!(blk.conv.dW, -clip, clip)
    clamp!(blk.conv.db, -clip, clip)
    return dX
end

mutable struct SpikePool
    argmax_idx::Array{Int, 6} # (B, C, H_out, W_out, T, 2)
end

function SpikePool()
    SpikePool(zeros(Int, 0,0,0,0,0,0))
end

function fwd_pool!(p::SpikePool, S::Array{Float64, 5})
    B, C, H, W, T = size(S)
    pH, pW = H ÷ 2, W ÷ 2
    out = zeros(B, C, pH, pW, T)
    p.argmax_idx = zeros(Int, B, C, pH, pW, T, 2)
    for t in 1:T, c in 1:C, i in 1:pH, j in 1:pW
        patch = S[:, c, 2i-1:2i, 2j-1:2j, t] # (B, 2, 2)
        flat = reshape(patch, B, 4)
        for b in 1:B
            val, idx = findmax(flat[b, :])
            out[b, c, i, j, t] = val
            li = idx
            p.argmax_idx[b, c, i, j, t, 1] = 2i - 2 + ((li - 1) ÷ 2 + 1)
            p.argmax_idx[b, c, i, j, t, 2] = 2j - 2 + ((li - 1) % 2 + 1)
        end
    end
    return out
end

function bwd_pool!(p::SpikePool, dOut::Array{Float64, 5}, in_shape)
    B, C, pH, pW, T = size(dOut)
    dS = zeros(in_shape)
    for t in 1:T, c in 1:C, i in 1:pH, j in 1:pW, b in 1:B
        r, col = p.argmax_idx[b, c, i, j, t, 1], p.argmax_idx[b, c, i, j, t, 2]
        dS[b, c, r, col, t] += dOut[b, c, i, j, t]
    end
    return dS
end

mutable struct SpikeResBlock
    blk1::ConvLIFBlock; blk2::ConvLIFBlock
    blk2_out::Array{Float64, 5}; X_input::Array{Float64, 5}
end

function SpikeResBlock(C)
    SpikeResBlock(ConvLIFBlock(C, C, 3; pad=1), ConvLIFBlock(C, C, 3; pad=1), zeros(0,0,0,0,0), zeros(0,0,0,0,0))
end

function fwd_res_block!(res::SpikeResBlock, X::Array{Float64, 5}; training=false)
    res.X_input = X
    out1 = fwd_conv_lif!(res.blk1, X, training=training)
    res.blk2_out = fwd_conv_lif!(res.blk2, out1, training=training)
    # Spike-OR skip connection
    return max.(res.blk2_out, X)
end

function bwd_res_block!(res::SpikeResBlock, dOut::Array{Float64, 5}, lambda::Float64, clip::Float64)
    # Correct Spike-OR gradient routing
    mask_blk2 = Float64.(res.blk2_out .>= res.X_input)
    mask_skip = Float64.(res.X_input .>= res.blk2_out)
    count = mask_blk2 .+ mask_skip
    
    dblk2 = dOut .* mask_blk2 ./ max.(count, 1.0)
    dskip = dOut .* mask_skip ./ max.(count, 1.0)
    
    dX2 = bwd_conv_lif!(res.blk2, dblk2, lambda, clip)
    dX1 = bwd_conv_lif!(res.blk1, dX2, lambda, clip)
    return dX1 .+ dskip
end

mutable struct SpikeDense
    W::Matrix{Float64}; b::Vector{Float64}
    dW::Matrix{Float64}; db::Vector{Float64}
    mW::Matrix{Float64}; vW::Matrix{Float64}
    mb::Vector{Float64}; vb::Vector{Float64}
end

function SpikeDense(ni, no)
    sc = sqrt(2.0 / ni)
    SpikeDense(randn(no, ni) .* sc, zeros(no), zeros(no, ni), zeros(no), zeros(no, ni), zeros(no, ni), zeros(no), zeros(no))
end

mutable struct LIFBlock
    layer::SpikeDense
    U::Array{Float64, 3}; S::Array{Float64, 3}; mask::Array{Float64, 3}
    inp::Array{Float64, 3}
end

function LIFBlock(ni, no)
    LIFBlock(SpikeDense(ni, no), zeros(0,0,0), zeros(0,0,0), zeros(0,0,0), zeros(0,0,0))
end

function fwd_lif!(blk::LIFBlock, X_seq::Array{Float64, 3}; training=false)
    ni, B, T = size(X_seq)
    no = size(blk.layer.W, 1)
    blk.inp = X_seq
    blk.U = zeros(no, B, T + 1)
    blk.S = zeros(no, B, T)
    blk.mask = ones(no, B, T)
    
    # Pre-compute current
    inp_raw = reshape(blk.layer.W * reshape(X_seq, ni, B*T), no, B, T) .+ blk.layer.b
    
    scale = 1.0 / (1.0 - SPIKE_DROPOUT)
    for t in 1:T
        @. blk.U[:, :, t+1] = blk.U[:, :, t] * V_LEAK + inp_raw[:, :, t]
        spikes = Float64.(blk.U[:, :, t+1] .> V_THRESHOLD)
        
        if training
            m = Float64.(rand(no, B) .> SPIKE_DROPOUT) .* scale
            blk.mask[:, :, t] .= m
            spikes .*= m
        end
        blk.S[:, :, t] .= spikes
        @. blk.U[:, :, t+1] = clamp(blk.U[:, :, t+1] - spikes * V_THRESHOLD, -1.0, V_THRESHOLD)
    end
    return blk.S
end

function bwd_lif!(blk::LIFBlock, dS::Array{Float64, 3}, lambda::Float64, clip::Float64)
    no, B, T = size(dS)
    ni = size(blk.layer.W, 2)
    dU = zeros(no, B)
    dX = zeros(ni, B, T)
    blk.layer.dW .= 0.0; blk.layer.db .= 0.0
    
    for t in T:-1:1
        grad_s = surrogate_grad.(blk.U[:, :, t+1] .- V_THRESHOLD)
        dS_masked = dS[:, :, t] .* blk.mask[:, :, t]
        dU_spike = dS_masked .* grad_s
        dU_curr = dU_spike .+ dU .* V_LEAK
        
        # dW: (no, B) * (B, ni) -> sums over B. dU_curr already has 1/B.
        blk.layer.dW .+= (dU_curr * blk.inp[:, :, t]') ./ T
        blk.layer.db .+= vec(mean(dU_curr, dims=2)) ./ T
        
        dX[:, :, t] .= blk.layer.W' * dU_curr
        dU .= dU_curr
    end
    blk.layer.dW .+= lambda .* blk.layer.W
    clamp!(blk.layer.dW, -clip, clip)
    clamp!(blk.layer.db, -clip, clip)
    return dX
end

mutable struct TemporalAttention
    w::Vector{Float64}; dw::Vector{Float64}
    mw::Vector{Float64}; vw::Vector{Float64}
end

function TemporalAttention()
    TemporalAttention(randn(SNN_T_STEPS) .* 0.1, zeros(SNN_T_STEPS), zeros(SNN_T_STEPS), zeros(SNN_T_STEPS))
end

function fwd_attention!(attn::TemporalAttention, S)
    alpha = softmax_s(attn.w) # (T,)
    no, B, T = size(S)
    # Sum over T with attention weights
    # S is (no, B, T)
    readout = zeros(no, B)
    for t in 1:T
        readout .+= S[:, :, t] .* alpha[t]
    end
    return readout, alpha
end

function bwd_attention!(attn::TemporalAttention, d_readout, S, alpha)
    no, B, T = size(S)
    dL_dalpha = zeros(T)
    for t in 1:T
        # dL/d_alpha[t] = sum(d_readout * S[t])
        dL_dalpha[t] = sum(d_readout .* S[:, :, t]) / B
    end
    # Softmax Jacobian
    attn.dw .= alpha .* (dL_dalpha .- dot(dL_dalpha, alpha))
end

# --- Network & Runner ---

mutable struct GestureSNNv7
    stem::ConvLIFBlock
    res1::SpikeResBlock
    pool1::SpikePool
    trans1::ConvLIFBlock
    res2a::SpikeResBlock
    res2b::SpikeResBlock
    pool2::SpikePool
    trans2::ConvLIFBlock
    res3::SpikeResBlock
    lif1::LIFBlock; lif2::LIFBlock
    attn::TemporalAttention
    head::SpikeDense
    training::Bool
end

function GestureSNNv7()
    GestureSNNv7(
        ConvLIFBlock(2, 32, 3; pad=1),
        SpikeResBlock(32),
        SpikePool(),
        ConvLIFBlock(32, 64, 3; pad=1),
        SpikeResBlock(64), SpikeResBlock(64),
        SpikePool(),
        ConvLIFBlock(64, 128, 3; pad=1),
        SpikeResBlock(128),
        LIFBlock(128*8*8, 512), LIFBlock(512, 256),
        TemporalAttention(),
        SpikeDense(256, 11),
        true
    )
end

function forward_v7!(net::GestureSNNv7, X::Matrix{Float64})
    B = size(X, 2)
    # Corrected data layout: (xd, yd, t, pol, B) -> (B, pol, xd, yd, T)
    X_reshaped = reshape(X, DVS_RES_DOWN, DVS_RES_DOWN, SNN_T_STEPS, 2, B)
    X_seq = permutedims(X_reshaped, (5, 4, 1, 2, 3))
    
    s = fwd_conv_lif!(net.stem, X_seq; training=net.training)
    s = fwd_res_block!(net.res1, s; training=net.training)
    s = fwd_pool!(net.pool1, s)
    
    s = fwd_conv_lif!(net.trans1, s; training=net.training)
    s = fwd_res_block!(net.res2a, s; training=net.training)
    s = fwd_res_block!(net.res2b, s; training=net.training)
    s = fwd_pool!(net.pool2, s)
    
    s = fwd_conv_lif!(net.trans2, s; training=net.training)
    s = fwd_res_block!(net.res3, s; training=net.training)
    
    # Flatten (B, C, H, W, T) -> (C*H*W, B, T)
    s_p = permutedims(s, (2, 3, 4, 1, 5))
    s_flat = reshape(s_p, :, B, SNN_T_STEPS)
    
    h1 = fwd_lif!(net.lif1, s_flat; training=net.training)
    h2 = fwd_lif!(net.lif2, h1; training=net.training)
    
    readout, alpha = fwd_attention!(net.attn, h2)
    return net.head.W * readout .+ net.head.b, alpha
end

function backward_v7!(net::GestureSNNv7, d, alpha, l2, clip)
    B = size(d, 2)
    T = SNN_T_STEPS
    
    # Head gradient - d already has 1/B
    readout_val = zeros(256, B)
    alpha_val = alpha
    for t in 1:T; readout_val .+= net.lif2.S[:, :, t] .* alpha_val[t]; end
    net.head.dW .= (d * readout_val') .+ l2 .* net.head.W
    net.head.db .= vec(sum(d, dims=2))
    
    # Attention and Dense layers
    d_readout = net.head.W' * d
    bwd_attention!(net.attn, d_readout, net.lif2.S, alpha_val)
    
    no2, _, _ = size(net.lif2.S)
    dS2 = reshape(d_readout, no2, B, 1) .* reshape(alpha_val, 1, 1, T)
    dS1 = bwd_lif!(net.lif2, dS2, l2, clip)
    dS_flat = bwd_lif!(net.lif1, dS1, l2, clip)
    
    # Conv blocks
    res = DVS_RES_DOWN
    dS_s3 = permutedims(reshape(dS_flat, 128, 8, 8, B, T), (4, 1, 2, 3, 5))
    
    dS_res3 = bwd_res_block!(net.res3, dS_s3, l2, clip)
    dS_trans2 = bwd_conv_lif!(net.trans2, dS_res3, l2, clip)
    
    dS_p2 = bwd_pool!(net.pool2, dS_trans2, (B, 64, 16, 16, T))
    dS_res2b = bwd_res_block!(net.res2b, dS_p2, l2, clip)
    dS_res2a = bwd_res_block!(net.res2a, dS_res2b, l2, clip)
    dS_trans1 = bwd_conv_lif!(net.trans1, dS_res2a, l2, clip)
    
    dS_p1 = bwd_pool!(net.pool1, dS_trans1, (B, 32, 32, 32, T))
    dS_res1 = bwd_res_block!(net.res1, dS_p1, l2, clip)
    bwd_conv_lif!(net.stem, dS_res1, l2, clip)
end

function adam_step!(layer, lr, step)
    b1=0.9; b2=0.999; eps=1e-8
    layer.mW .= b1 .* layer.mW .+ (1-b1) .* layer.dW
    layer.vW .= b2 .* layer.vW .+ (1-b2) .* (layer.dW .^ 2)
    m_hat = layer.mW ./ (1 - b1^step)
    v_hat = layer.vW ./ (1 - b2^step)
    layer.W .-= lr .* m_hat ./ (sqrt.(v_hat) .+ eps)
    
    layer.mb .= b1 .* layer.mb .+ (1-b1) .* layer.db
    layer.vb .= b2 .* layer.vb .+ (1-b2) .* (layer.db .^ 2)
    mb_hat = layer.mb ./ (1 - b1^step)
    vb_hat = layer.vb ./ (1 - b2^step)
    layer.b .-= lr .* mb_hat ./ (sqrt.(vb_hat) .+ eps)
end

function adam_bn!(bn::tdBN, lr, step)
    b1=0.9; b2=0.999; eps=1e-8
    bn.m_gamma .= b1 .* bn.m_gamma .+ (1-b1) .* bn.d_gamma
    bn.v_gamma .= b2 .* bn.v_gamma .+ (1-b2) .* (bn.d_gamma .^ 2)
    mg_hat = bn.m_gamma ./ (1 - b1^step)
    vg_hat = bn.v_gamma ./ (1 - b2^step)
    bn.gamma .-= lr .* mg_hat ./ (sqrt.(vg_hat) .+ eps)
    
    bn.m_beta .= b1 .* bn.m_beta .+ (1-b1) .* bn.d_beta
    bn.v_beta .= b2 .* bn.v_beta .+ (1-b2) .* (bn.d_beta .^ 2)
    mb_hat = bn.m_beta ./ (1 - b1^step)
    vb_hat = bn.v_beta ./ (1 - b2^step)
    bn.beta .-= lr .* mb_hat ./ (sqrt.(vb_hat) .+ eps)
end

function augment_dvs(X_b::Matrix{Float64}, res, T_steps)
    B = size(X_b, 2)
    # Bypass 20%
    if rand() < 0.2; return X_b; end
    
    X_aug = copy(X_b)
    X_tensor = reshape(X_aug, res, res, T_steps, 2, B)
    for b in 1:B
        # Polarity swap: axis 4
        if rand() > 0.5
            tmp = copy(X_tensor[:, :, :, 1, b])
            X_tensor[:, :, :, 1, b] .= X_tensor[:, :, :, 2, b]
            X_tensor[:, :, :, 2, b] .= tmp
        end
        # Spatial flip: axis 1
        if rand() > 0.5
            X_tensor[:, :, :, :, b] .= X_tensor[end:-1:1, :, :, :, b]
        end
        # Temporal jitter ±1: axis 3
        shift = rand(-1:1)
        if shift != 0
            X_tensor[:, :, :, :, b] .= circshift(X_tensor[:, :, :, :, b], (0, 0, shift, 0))
        end
        # Sparse event dropout 5%
        mask = rand(res, res, T_steps, 2) .> 0.05
        X_tensor[:, :, :, :, b] .*= mask
    end
    return reshape(X_tensor, :, B)
end

function run_dvs_omega_v7()
    println("\n" * "█"^80)
    println("   OMEGA v7.1 — RESNET + SPIKE-OR + tdBN (STABILIZED)")
    println("█"^80)
    
    train_list = read_trial_list(joinpath(DVS_DATA_DIR, "trials_to_train.txt"))
    test_list = read_trial_list(joinpath(DVS_DATA_DIR, "trials_to_test.txt"))
    X_train, y_train = load_split(train_list, SNN_T_STEPS)
    X_test, y_test = load_split(test_list, SNN_T_STEPS)
    
    epochs = 300; batch_size = 48; peak_lr = 8e-4; warmup = 15; min_lr = 1e-5; l2 = 1e-5; grad_clip = 1.0
    n = length(y_train); net = GestureSNNv7()
    best_acc = 0.0; step = 0; start_epoch = 1
    checkpoint_path = "dvs_omega_v7_checkpoint.jls"
    
    if isfile(checkpoint_path)
        println(">> Resuming from checkpoint...")
        data = deserialize(checkpoint_path)
        net = data.net; best_acc = data.best_acc; step = data.step; start_epoch = data.epoch + 1
    end
    
    for ep in start_epoch:epochs
        if ep <= warmup
            lr = peak_lr * ep / warmup
        else
            lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * (ep - warmup) / (epochs - warmup)))
        end
        
        perm = randperm(n)
        net.training = true
        total_loss = 0.0
        for i in 1:batch_size:n
            end_idx = min(i + batch_size - 1, n)
            batch_indices = perm[i:end_idx]
            X_b = augment_dvs(X_train[:, batch_indices], DVS_RES_DOWN, SNN_T_STEPS)
            y_b = y_train[batch_indices]
            B_curr = length(y_b)
            
            y_pred, alpha = forward_v7!(net, X_b)
            probs = softmax_s(y_pred)
            
            # Cross entropy with label smoothing
            y_smooth = fill(LABEL_SMOOTHING / 11, 11, B_curr)
            for b_i in 1:B_curr; y_smooth[y_b[b_i], b_i] = 1.0 - LABEL_SMOOTHING; end
            loss = -sum(y_smooth .* log.(probs .+ 1e-8)) / B_curr
            total_loss += loss
            
            d = (probs .- y_smooth) ./ B_curr
            backward_v7!(net, d, alpha, l2, grad_clip)
            
            step += 1
            # Optimizer steps
            for block in [net.stem.conv, net.trans1.conv, net.trans2.conv, net.res1.blk1.conv, net.res1.blk2.conv,
                          net.res2a.blk1.conv, net.res2a.blk2.conv, net.res2b.blk1.conv, net.res2b.blk2.conv,
                          net.res3.blk1.conv, net.res3.blk2.conv, net.lif1.layer, net.lif2.layer, net.head]
                adam_step!(block, lr, step)
            end
            for bn in [net.stem.bn, net.trans1.bn, net.trans2.bn, net.res1.blk1.bn, net.res1.blk2.bn,
                       net.res2a.blk1.bn, net.res2a.blk2.bn, net.res2b.blk1.bn, net.res2b.blk2.bn,
                       net.res3.blk1.bn, net.res3.blk2.bn]
                adam_bn!(bn, lr, step)
            end
            # Attention step
            attn = net.attn
            attn.mw .= 0.9attn.mw .+ 0.1attn.dw
            attn.vw .= 0.999attn.vw .+ 0.001(attn.dw .^ 2)
            attn.w .-= lr .* (attn.mw ./ (1-0.9^step)) ./ (sqrt.(attn.vw ./ (1-0.999^step)) .+ 1e-8)
        end
        
        # Evaluation
        net.training = false
        test_preds = Int[]
        for i in 1:batch_size:length(y_test)
            e_idx = min(i + batch_size - 1, length(y_test))
            X_te_b = X_test[:, i:e_idx]; y_te_b = y_test[i:e_idx]
            y_p_te, _ = forward_v7!(net, X_te_b)
            append!(test_preds, [argmax(y_p_te[:, j]) for j in 1:size(y_p_te, 2)])
        end
        acc = mean(test_preds .== y_test)
        if acc > best_acc; best_acc = acc; end
        
        # Firing rates
        f_stem = mean(net.stem.S) * 100
        f_res1 = mean(net.res1.blk1.S) * 100
        f_res2 = mean(net.res2a.blk1.S) * 100
        f_res3 = mean(net.res3.blk1.S) * 100
        
        @printf("Epoch %3d | Loss: %.4f | Acc: %5.2f%% (Best: %5.2f%%) | LR: %.5f | Firing: %.1f%% / %.1f%% / %.1f%% / %.1f%%\n",
                ep, total_loss / (n/batch_size), acc*100, best_acc*100, lr, f_stem, f_res1, f_res2, f_res3)
        
        serialize(checkpoint_path, (net=net, best_acc=best_acc, step=step, epoch=ep))
    end
end

run_dvs_omega_v7()
