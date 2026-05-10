using Serialization, Statistics, Random, Printf, Dates, LinearAlgebra
include("../../src/dvs_loader.jl")

const SURROGATE_SCALE = 10.0
const V_THRESHOLD = 1.0
const V_LEAK = 0.9
const SNN_T_STEPS = 20
const SPIKE_DROPOUT = 0.05
const LABEL_SMOOTHING = 0.1

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

mutable struct tdBN
    gamma::Array{Float64, 4}; beta::Array{Float64, 4}
    running_mean::Array{Float64, 4}; running_var::Array{Float64, 4}
    momentum::Float64
    m_gamma::Array{Float64, 4}; v_gamma::Array{Float64, 4}
    m_beta::Array{Float64, 4}; v_beta::Array{Float64, 4}
    d_gamma::Array{Float64, 4}; d_beta::Array{Float64, 4}
    # Cache for backward pass
    X_raw::Array{Float64, 5}
    mu_batch::Array{Float64, 5}
    var_batch::Array{Float64, 5}
end

function tdBN(C)
    tdBN(ones(1, C, 1, 1), zeros(1, C, 1, 1),
         zeros(1, C, 1, 1), ones(1, C, 1, 1), 0.1,
         zeros(1, C, 1, 1), zeros(1, C, 1, 1),
         zeros(1, C, 1, 1), zeros(1, C, 1, 1),
         zeros(1, C, 1, 1), zeros(1, C, 1, 1),
         zeros(0,0,0,0,0), zeros(0,0,0,0,0), zeros(0,0,0,0,0))
end

function apply_tdbn!(bn::tdBN, X::Array{Float64, 5}, V_th, training)
    B, C, H, W, T = size(X)
    bn.X_raw = zeros(size(X))
    bn.mu_batch = zeros(1, C, 1, 1, T)
    bn.var_batch = zeros(1, C, 1, 1, T)
    out = zeros(size(X))
    for t in 1:T
        x_t = X[:, :, :, :, t]
        if training
            mu = mean(x_t, dims=(1, 3, 4))
            v_val = var(x_t, dims=(1, 3, 4), corrected=false) # N-denominator for BN
            bn.mu_batch[:, :, :, :, t] .= mu
            bn.var_batch[:, :, :, :, t] .= v_val
            bn.X_raw[:, :, :, :, t] .= x_t
            bn.running_mean .= (1 - bn.momentum) .* bn.running_mean .+ bn.momentum .* mu
            bn.running_var .= (1 - bn.momentum) .* bn.running_var .+ bn.momentum .* v_val
        else
            mu = bn.running_mean
            v_val = bn.running_var
        end
        inv_std = 1.0 ./ sqrt.(v_val .+ 1e-5)
        out[:, :, :, :, t] = V_th .* bn.gamma .* ((x_t .- mu) .* inv_std) .+ bn.beta
    end
    return out
end

function bwd_tdbn!(bn::tdBN, dY::Array{Float64, 5}, V_th)
    B, C, H, W, T = size(dY)
    N = B * H * W
    dX = zeros(size(dY))
    bn.d_gamma .= 0.0; bn.d_beta .= 0.0
    for t in 1:T
        dy = dY[:, :, :, :, t]
        mu = bn.mu_batch[:, :, :, :, t]
        v_val = bn.var_batch[:, :, :, :, t]
        x_raw = bn.X_raw[:, :, :, :, t]
        
        inv_std = 1.0 ./ sqrt.(v_val .+ 1e-5)
        x_hat = (x_raw .- mu) .* inv_std
        
        bn.d_gamma .+= reshape(sum(dy .* x_hat, dims=(1, 3, 4)), 1, C, 1, 1) .* V_th
        bn.d_beta .+= reshape(sum(dy, dims=(1, 3, 4)), 1, C, 1, 1)
        
        # dy_dx_hat * dx_hat_dx
        duh = dy .* bn.gamma .* V_th
        dvar = sum(duh .* (x_raw .- mu) .* (-0.5) .* (inv_std .^ 3), dims=(1, 3, 4))
        dmu = sum(-duh .* inv_std, dims=(1, 3, 4)) .+ dvar .* (-2.0 ./ N) .* sum(x_raw .- mu, dims=(1, 3, 4))
        dX[:, :, :, :, t] = duh .* inv_std .+ (dvar .* 2.0 ./ N) .* (x_raw .- mu) .+ dmu ./ N
    end
    return dX
end
mutable struct SpikeConv
    in_ch::Int; out_ch::Int
    k::Int; pad::Int
    W::Array{Float64, 4}; b::Vector{Float64}
    dW::Array{Float64, 4}; db::Vector{Float64}
    mW::Array{Float64, 4}; vW::Array{Float64, 4}
    mb::Vector{Float64}; vb::Vector{Float64}
    inp::Array{Float64, 5}
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

mutable struct TemporalAttention
    w::Vector{Float64}; dw::Vector{Float64}
    mw::Vector{Float64}; vw::Vector{Float64}
end

function TemporalAttention(T)
    TemporalAttention(ones(T) ./ T, zeros(T), zeros(T), zeros(T))
end

mutable struct ConvLIFBlock
    conv::SpikeConv
    bn::tdBN
    U::Array{Float64, 5}; S::Array{Float64, 5}
    v_th_hist::Vector{Float64}
    mask::Array{Float64, 5}
end

function ConvLIFBlock(in_ch, out_ch, k; pad=1)
    ConvLIFBlock(SpikeConv(in_ch, out_ch, k; pad=pad), tdBN(out_ch),
                 zeros(0,0,0,0,0), zeros(0,0,0,0,0), zeros(0), zeros(0,0,0,0,0))
end

mutable struct LIFBlock
    layer::SpikeDense
    U::Array{Float64, 3}; S::Array{Float64, 3}; inp::Array{Float64, 3}
    v_th_hist::Vector{Float64}
    mask::Array{Float64, 3}
end

function LIFBlock(ni, no)
    LIFBlock(SpikeDense(ni, no), zeros(0,0,0), zeros(0,0,0), zeros(0,0,0), zeros(0), zeros(0,0,0))
end


function augment_dvs(X_b::Matrix{Float64}, res, T)
    B = size(X_b, 2)
    X_aug = copy(X_b)
    X_tensor = reshape(X_aug, 2, res, res, T, B)
    for b in 1:B
        if rand() > 0.5
            tmp = copy(X_tensor[1, :, :, :, b])
            X_tensor[1, :, :, :, b] .= X_tensor[2, :, :, :, b]
            X_tensor[2, :, :, :, b] .= tmp
        end
        if rand() > 0.5
            X_tensor[:, :, :, :, b] .= X_tensor[:, :, end:-1:1, :, b]
        end
        shift = rand(-2:2)
        if shift != 0
            X_tensor[:, :, :, :, b] .= circshift(X_tensor[:, :, :, :, b], (0, 0, 0, shift))
        end
        mask = rand(size(X_tensor[:, :, :, :, b])...) .> 0.1
        X_tensor[:, :, :, :, b] .*= mask
    end
    return reshape(X_tensor, :, B)
end

function conv2d!(out::AbstractArray{Float64, 4}, X::AbstractArray{Float64, 4}, W::Array{Float64, 4}, b::Vector{Float64}, pad::Int)
    B, IC, H, W_in = size(X)
    OC, _, k, _ = size(W)
    H_out, W_out = H + 2pad - k + 1, W_in + 2pad - k + 1
    actual_col_size = B * H_out * W_out
    X_col = zeros(IC * k * k, actual_col_size)
    X_pad = zeros(B, IC, H + 2pad, W_in + 2pad)
    X_pad[:, :, pad+1:H+pad, pad+1:W_in+pad] .= X
    @inbounds for row in 1:H_out, col in 1:W_out, b_idx in 1:B
        col_idx = (row-1)*W_out*B + (col-1)*B + b_idx
        for ic in 1:IC, i in 1:k, j in 1:k
            X_col[(ic-1)*k*k + (i-1)*k + j, col_idx] = X_pad[b_idx, ic, row+i-1, col+j-1]
        end
    end
    W_mat = reshape(W, OC, IC * k * k)
    out_mat = W_mat * X_col
    out_reshaped = reshape(out_mat, OC, B, H_out, W_out)
    for oc in 1:OC, b_idx in 1:B, r in 1:H_out, c in 1:W_out
        out[b_idx, oc, r, c] = out_reshaped[oc, b_idx, r, c] + b[oc]
    end
end

function conv2d_grad(X::AbstractArray{Float64, 4}, W::Array{Float64, 4}, d_out::AbstractArray{Float64, 4}, pad::Int)
    B, IC, H, W_in = size(X)
    OC, _, k, _ = size(W)
    H_out, W_out = size(d_out, 3), size(d_out, 4)
    actual_col_size = B * H_out * W_out
    X_col = zeros(IC * k * k, actual_col_size)
    X_pad = zeros(B, IC, H + 2pad, W_in + 2pad)
    X_pad[:, :, pad+1:H+pad, pad+1:W_in+pad] = X
    db = vec(sum(d_out, dims=(1,3,4))) ./ B # Sum spatial, average batch
    d_out_perm = permutedims(d_out, (2, 1, 3, 4))
    d_out_mat = reshape(d_out_perm, OC, actual_col_size)
    @inbounds for row in 1:H_out, col in 1:W_out, b_idx in 1:B
        col_idx = (row-1)*W_out*B + (col-1)*B + b_idx
        for ic in 1:IC, i in 1:k, j in 1:k
            X_col[(ic-1)*k*k + (i-1)*k + j, col_idx] = X_pad[b_idx, ic, row+i-1, col+j-1]
        end
    end
    # dW is normalized by sample count B
    dW_mat = (d_out_mat * X_col') ./ B
    dW = reshape(dW_mat, OC, IC, k, k)
    W_mat = reshape(W, OC, IC * k * k)
    dX_col = W_mat' * d_out_mat
    dX_pad = zeros(B, IC, H + 2pad, W_in + 2pad)
    for row in 1:H_out, col in 1:W_out, b_idx in 1:B
        col_idx = (row-1)*W_out*B + (col-1)*B + b_idx
        for ic in 1:IC, i in 1:k, j in 1:k
            dX_pad[b_idx, ic, row+i-1, col+j-1] += dX_col[(ic-1)*k*k + (i-1)*k + j, col_idx]
        end
    end
    # No normalization on dX
    dX = dX_pad[:, :, pad+1:H+pad, pad+1:W_in+pad]
    return dW, db, dX
end

function fwd_conv_lif!(blk::ConvLIFBlock, X_seq::Array{Float64, 5}; training=false)
    B, C, H, W, T = size(X_seq)
    OC = blk.conv.out_ch
    blk.conv.inp = X_seq
    conv_out = zeros(B, OC, H, W, T)
    for t in 1:T
        out_t = @view conv_out[:, :, :, :, t]
        conv2d!(out_t, X_seq[:, :, :, :, t], blk.conv.W, blk.conv.b, blk.conv.pad)
    end
    I_bn = apply_tdbn!(blk.bn, conv_out, V_THRESHOLD, training)
    blk.U = zeros(B, OC, H, W, T + 1)
    blk.S = zeros(B, OC, H, W, T)
    blk.mask = ones(B, OC, H, W, T)
    for t in 1:T
        for b in 1:B, c in 1:OC, i in 1:H, j in 1:W
            blk.U[b, c, i, j, t+1] = blk.U[b, c, i, j, t] * V_LEAK + I_bn[b, c, i, j, t]
        end
        spikes = Float64.(blk.U[:, :, :, :, t+1] .> V_THRESHOLD)
        if training
            # Proper Bernoulli dropout with scaling
            scale = 1.0 / (1.0 - SPIKE_DROPOUT)
            mask = Float64.(rand(B, OC, H, W) .> SPIKE_DROPOUT) .* scale
            spikes .*= mask
            blk.mask[:, :, :, :, t] .= mask
        end
        blk.S[:, :, :, :, t] .= spikes
        for b in 1:B, c in 1:OC, i in 1:H, j in 1:W
            # Clamped soft reset to prevent runaway firing
            blk.U[b, c, i, j, t+1] = clamp(blk.U[b, c, i, j, t+1] - spikes[b, c, i, j] * V_THRESHOLD, -1.0, V_THRESHOLD)
        end
    end
    return blk.S
end

function bwd_conv_lif!(blk::ConvLIFBlock, dS::Array{Float64, 5}, lambda::Float64, clip::Float64)
    B, OC, H, W, T = size(dS)
    dU = zeros(B, OC, H, W)
    dI_bn = zeros(B, OC, H, W, T)
    dS_masked = dS .* blk.mask
    for t in T:-1:1
        grad_s = surrogate_grad.(blk.U[:, :, :, :, t+1] .- V_THRESHOLD)
        # Gate entire accumulated gradient through surrogate
        dU_curr = (dS_masked[:, :, :, :, t] .+ dU .* V_LEAK) .* grad_s
        dI_bn[:, :, :, :, t] .= dU_curr
        dU = dU_curr
    end
    dI_raw = bwd_tdbn!(blk.bn, dI_bn, V_THRESHOLD)
    return bwd_conv!(blk.conv, dI_raw, lambda, clip, T)
end

function bwd_conv!(l::SpikeConv, d_out_seq::Array{Float64, 5}, lambda::Float64, clip::Float64, T::Int)
    B, OC, H, W, _ = size(d_out_seq)
    l.dW .= 0.0; l.db .= 0.0
    dX_seq = zeros(size(l.inp))
    for t in 1:T
        dw, db, dx = conv2d_grad(l.inp[:, :, :, :, t], l.W, d_out_seq[:, :, :, :, t], l.pad)
        l.dW .+= dw ./ T
        l.db .+= db ./ T
        dX_seq[:, :, :, :, t] .= dx
    end
    l.dW .+= lambda .* l.W
    gnorm = sqrt(sum(l.dW.^2) + sum(l.db.^2))
    if gnorm > clip
        sc = clip / gnorm; l.dW .*= sc; l.db .*= sc
    end
    return dX_seq
end

mutable struct SpikeResBlock
    blk1::ConvLIFBlock; blk2::ConvLIFBlock
    clamp_mask::Array{Float64, 5}
end

function SpikeResBlock(C)
    SpikeResBlock(ConvLIFBlock(C, C, 3; pad=1), ConvLIFBlock(C, C, 3; pad=1), zeros(0,0,0,0,0))
end

function fwd_res_block!(res::SpikeResBlock, X::Array{Float64, 5}; training=false)
    out1 = fwd_conv_lif!(res.blk1, X; training=training)
    out2 = fwd_conv_lif!(res.blk2, out1; training=training)
    sum_spikes = out2 .+ X
    res.clamp_mask = Float64.(sum_spikes .<= 1.0)
    return min.(sum_spikes, 1.0)
end

function bwd_res_block!(res::SpikeResBlock, dOut::Array{Float64, 5}, lambda::Float64, clip::Float64)
    dOut_masked = dOut .* res.clamp_mask
    dout1 = bwd_conv_lif!(res.blk2, dOut_masked, lambda, clip)
    dX = bwd_conv_lif!(res.blk1, dout1, lambda, clip)
    return (dX .+ dOut_masked)
end

mutable struct SpikePool
    argmax_idx::Array{Int, 6}
end

function SpikePool()
    SpikePool(zeros(Int, 0,0,0,0,0,0))
end

function fwd_pool!(p::SpikePool, S::Array{Float64, 5})
    B, C, H, W, T = size(S)
    pH, pW = H÷2, W÷2
    p.argmax_idx = zeros(Int, B, C, pH, pW, T, 2)
    out = zeros(B, C, pH, pW, T)
    for t in 1:T, c in 1:C, i in 1:pH, j in 1:pW
        patch = S[:, c, 2i-1:2i, 2j-1:2j, t]
        for b in 1:B
            val, idx = findmax(patch[b, :, :])
            out[b, c, i, j, t] = val
            p.argmax_idx[b, c, i, j, t, 1] = 2i - 2 + idx[1]
            p.argmax_idx[b, c, i, j, t, 2] = 2j - 2 + idx[2]
        end
    end
    return out
end

function bwd_pool!(p::SpikePool, dS_pool::Array{Float64, 5}, H, W)
    B, C, pH, pW, T = size(dS_pool)
    dS = zeros(B, C, H, W, T)
    for t in 1:T, b in 1:B, c in 1:C, i in 1:pH, j in 1:pW
        r = p.argmax_idx[b, c, i, j, t, 1]
        col = p.argmax_idx[b, c, i, j, t, 2]
        dS[b, c, r, col, t] = dS_pool[b, c, i, j, t]
    end
    return dS
end

function fwd_lif!(blk::LIFBlock, X_seq::Array{Float64, 3}; training=false)
    ni, B, T = size(X_seq)
    no = blk.layer.no
    blk.inp = X_seq
    blk.U = zeros(no, B, T + 1)
    blk.S = zeros(no, B, T)
    blk.mask = ones(no, B, T)
    blk.v_th_hist = fill(V_THRESHOLD, T)
    @inbounds for t in 1:T
        curr = blk.layer.W * X_seq[:, :, t] .+ blk.layer.b
        for b in 1:B, n in 1:no
            blk.U[n, b, t+1] = blk.U[n, b, t] * V_LEAK + curr[n, b]
        end
        spikes = Float64.(blk.U[:, :, t+1] .> V_THRESHOLD)
        if training
            # Proper Bernoulli dropout with scaling
            scale = 1.0 / (1.0 - SPIKE_DROPOUT)
            mask = Float64.(rand(no, B) .> SPIKE_DROPOUT) .* scale
            spikes .*= mask
            blk.mask[:, :, t] .= mask
        end
        blk.S[:, :, t] .= spikes
        for b in 1:B, n in 1:no
            # Clamped soft reset to prevent runaway firing
            blk.U[n, b, t+1] = clamp(blk.U[n, b, t+1] - spikes[n, b] * V_THRESHOLD, -1.0, V_THRESHOLD)
        end
    end
    return blk.S
end

function bwd_lif!(blk::LIFBlock, dS::Array{Float64, 3}, lambda::Float64, clip::Float64)
    no, B, T = size(dS)
    l = blk.layer; l.dW .= 0.0; l.db .= 0.0
    dU = zeros(no, B); dX = zeros(size(blk.inp))
    dS_masked = dS .* blk.mask
    for t in T:-1:1
        grad_s = surrogate_grad.(blk.U[:, :, t+1] .- V_THRESHOLD)
        dU_curr = (dS_masked[:, :, t] .+ dU .* V_LEAK) .* grad_s
        l.dW .+= (dU_curr * blk.inp[:, :, t]') ./ (B * T)
        l.db .+= vec(mean(dU_curr, dims=2)) ./ T
        dX[:, :, t] = l.W' * dU_curr
        dU = dU_curr
    end
    l.dW .+= lambda .* l.W
    gnorm = sqrt(sum(l.dW.^2) + sum(l.db.^2))
    if gnorm > clip; s = clip/gnorm; l.dW .*= s; l.db .*= s; end
    return dX
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

function bwd_attention!(attn::TemporalAttention, d_readout, S, alpha)
    no, B, T = size(S)
    dL_dalpha = [sum(d_readout .* S[:, :, t]) / B for t in 1:T]
    for t in 1:T
        attn.dw[t] = alpha[t] * (dL_dalpha[t] - dot(dL_dalpha, alpha))
    end
end

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
    lif1::LIFBlock
    lif2::LIFBlock
    attn::TemporalAttention
    head::SpikeDense
    training::Bool
end

function GestureSNNv7()
    res = DVS_RES_DOWN
    GestureSNNv7(
        ConvLIFBlock(2, 32, 3; pad=1),
        SpikeResBlock(32),
        SpikePool(),
        ConvLIFBlock(32, 64, 3; pad=1),
        SpikeResBlock(64),
        SpikeResBlock(64),
        SpikePool(),
        ConvLIFBlock(64, 128, 3; pad=1),
        SpikeResBlock(128),
        LIFBlock(128 * (res÷4)^2, 512),
        LIFBlock(512, 256),
        TemporalAttention(SNN_T_STEPS),
        SpikeDense(256, 11),
        true
    )
end

function forward_v7!(net::GestureSNNv7, X::Matrix{Float64})
    B = size(X, 2)
    X_seq = reshape(X, 2, DVS_RES_DOWN, DVS_RES_DOWN, SNN_T_STEPS, B)
    X_seq = permutedims(X_seq, (5, 1, 2, 3, 4))
    s = fwd_conv_lif!(net.stem, X_seq; training=net.training)
    s = fwd_res_block!(net.res1, s; training=net.training)
    s = fwd_pool!(net.pool1, s)
    s = fwd_conv_lif!(net.trans1, s; training=net.training)
    s = fwd_res_block!(net.res2a, s; training=net.training)
    s = fwd_res_block!(net.res2b, s; training=net.training)
    s = fwd_pool!(net.pool2, s)
    s = fwd_conv_lif!(net.trans2, s; training=net.training)
    s = fwd_res_block!(net.res3, s; training=net.training)
    B_sz, C_sz, H_sz, W_sz, T_sz = size(s)
    h_flat = reshape(permutedims(s, (2,3,4,1,5)), C_sz*H_sz*W_sz, B_sz, T_sz)
    h1 = fwd_lif!(net.lif1, h_flat; training=net.training)
    h2 = fwd_lif!(net.lif2, h1; training=net.training)
    readout, _ = apply_attention(net.attn, h2)
    return softmax_s(net.head.W * readout .+ net.head.b)
end

function backward_v7!(net::GestureSNNv7, y_oh, y_pred, weights; lr=1e-3, l2=1e-5, clip=1.0)
    B = size(y_oh, 2)
    num_classes = size(y_oh, 1)
    y_smooth = y_oh .* (1 - LABEL_SMOOTHING) .+ (LABEL_SMOOTHING / num_classes)
    # Apply normalized class weights to the gradient
    y_b = [argmax(y_oh[:, i]) for i in 1:B]
    w_batch = weights[y_b]
    w_batch = w_batch ./ mean(w_batch) # Stabilize per-batch gradient scale
    d = (y_pred .- y_smooth) .* reshape(w_batch, 1, B)
    
    readout, alpha = apply_attention(net.attn, net.lif2.S)
    net.head.dW = d * readout' ./ B .+ l2 .* net.head.W
    net.head.db = vec(mean(d, dims=2))
    # Gradient clipping on head
    gnorm_h = sqrt(sum(net.head.dW.^2) + sum(net.head.db.^2))
    if gnorm_h > clip; sc = clip/gnorm_h; net.head.dW .*= sc; net.head.db .*= sc; end
    
    d_readout = net.head.W' * d
    bwd_attention!(net.attn, d_readout, net.lif2.S, alpha)
    no, _, T = size(net.lif2.S)
    dS2 = reshape(d_readout, no, B, 1) .* reshape(alpha, 1, 1, T)
    dS1 = bwd_lif!(net.lif2, dS2, l2, clip)
    dS_flat = bwd_lif!(net.lif1, dS1, l2, clip)
    
    # Derivations of spatial shapes
    res = DVS_RES_DOWN
    C3, H3, W3 = 128, res÷4, res÷4
    @assert size(dS_flat, 1) == C3 * H3 * W3 "Shape mismatch in backward reshape"
    dS_s3 = reshape(dS_flat, C3, H3, W3, B, T)
    dS_s3 = permutedims(dS_s3, (4, 1, 2, 3, 5))
    dS_res3 = bwd_res_block!(net.res3, dS_s3, l2, clip)
    dS_trans2 = bwd_conv_lif!(net.trans2, dS_res3, l2, clip)
    dS_pool2 = bwd_pool!(net.pool2, dS_trans2, res÷2, res÷2)
    dS_res2b = bwd_res_block!(net.res2b, dS_pool2, l2, clip)
    dS_res2a = bwd_res_block!(net.res2a, dS_res2b, l2, clip)
    dS_trans1 = bwd_conv_lif!(net.trans1, dS_res2a, l2, clip)
    dS_pool1 = bwd_pool!(net.pool1, dS_trans1, res, res)
    dS_res1 = bwd_res_block!(net.res1, dS_pool1, l2, clip)
    bwd_conv_lif!(net.stem, dS_res1, l2, clip)
end

function adam_step_v7!(net::GestureSNNv7, lr, t)
    bc1, bc2 = 1.0-0.9^t, 1.0-0.999^t
    function upd!(p, dp, mp, vp)
        @. mp = 0.9*mp + 0.1*dp; @. vp = 0.999*vp + 0.001*dp^2
        @. p -= lr*(mp/bc1)/(sqrt(vp/bc2)+1e-8)
    end
    upd!(net.head.W, net.head.dW, net.head.mW, net.head.vW)
    upd!(net.head.b, net.head.db, net.head.mb, net.head.vb)
    for l in [net.lif1.layer, net.lif2.layer]
        upd!(l.W, l.dW, l.mW, l.vW); upd!(l.b, l.db, l.mb, l.vb)
    end
    upd!(net.attn.w, net.attn.dw, net.attn.mw, net.attn.vw)
    blocks = [net.stem, net.res1.blk1, net.res1.blk2, 
              net.trans1, net.res2a.blk1, net.res2a.blk2, net.res2b.blk1, net.res2b.blk2,
              net.trans2, net.res3.blk1, net.res3.blk2]
    for b in blocks
        upd!(b.conv.W, b.conv.dW, b.conv.mW, b.conv.vW)
        upd!(b.conv.b, b.conv.db, b.conv.mb, b.conv.vb)
        upd!(b.bn.gamma, b.bn.d_gamma, b.bn.m_gamma, b.bn.v_gamma)
        upd!(b.bn.beta, b.bn.d_beta, b.bn.m_beta, b.bn.v_beta)
    end
end

function run_dvs_omega_v7()
    println("\n" * "█"^80)
    println("   NEUROMORPHIC GESTURE RECOGNITION — OMEGA v7 (ResNet + tdBN)")
    println("   6-Layer ResNet | tdBN | Data Augmentation | Correct Pools & Attention")
    println("█"^80)
    train_list = read_trial_list(joinpath(DVS_DATA_DIR, "trials_to_train.txt"))
    test_list = read_trial_list(joinpath(DVS_DATA_DIR, "trials_to_test.txt"))
    X_train, y_train = load_split(train_list, SNN_T_STEPS)
    X_test, y_test = load_split(test_list, SNN_T_STEPS)
    
    # Sanity Check
    X_sample = reshape(X_train[:, 1], 2, DVS_RES_DOWN, DVS_RES_DOWN, SNN_T_STEPS)
    pol1 = sum(X_sample[1, :, :, :]); pol2 = sum(X_sample[2, :, :, :])
    corr = cor(vec(X_sample[1,:,:,:]), vec(X_sample[2,:,:,:]))
    println(">> Data Sanity: Polarity 1: $pol1, Polarity 2: $pol2, Correlation: $corr")
    if pol1 == pol2 || corr > 0.8
        println("!! WARNING: Polarity channels may be corrupted.")
    end
    
    counts = zeros(11); for l in y_train; counts[l] += 1; end
    weights = sum(counts) ./ (11 .* max.(counts, 1.0))
    epochs = 300
    batch_size = 32
    n = length(y_train)
    net = GestureSNNv7()
    best_acc = 0.0; step = 0; start_epoch = 1
    checkpoint_path = "dvs_omega_v7_checkpoint.jls"
    if isfile(checkpoint_path)
        println(">> Resuming from checkpoint...")
        cp = deserialize(checkpoint_path)
        net = cp.net; best_acc = cp.best_acc; step = cp.step; start_epoch = cp.epoch + 1
    end
    
    for ep in start_epoch:epochs
        peak_lr = 5e-3
        if ep <= 10
            lr = peak_lr * ep / 10
        else
            lr = 0.0001 + 0.5 * (peak_lr - 0.0001) * (1 + cos(pi * (ep-10) / (epochs-10)))
        end
        perm = randperm(n)
        net.training = true
        start_time = time()
        for i in 1:batch_size:n
            step += 1
            idx = perm[i:min(i+batch_size-1, n)]
            X_b = augment_dvs(X_train[:, idx], DVS_RES_DOWN, SNN_T_STEPS)
            y_b = y_train[idx]
            y_oh = zeros(11, length(y_b))
            for (j, l) in enumerate(y_b); y_oh[l, j] = 1.0; end
            
            y_pred = forward_v7!(net, X_b)
            backward_v7!(net, y_oh, y_pred, weights; lr=lr)
            adam_step_v7!(net, lr, step)
            
            if i % 128 == 1 || i + batch_size > n
                @printf("  [Epoch %d] Progress: %d/%d (Batch: %.2f s)\n", ep, i + length(y_b) - 1, n, (time() - start_time))
            end
        end
        
        if ep == 1 || ep % 2 == 0
            net.training = false
            y_pred_te = forward_v7!(net, X_test)
            preds_te = [argmax(y_pred_te[:, i]) for i in 1:size(y_pred_te, 2)]
            acc = mean(preds_te .== y_test)
            
            if acc > best_acc; best_acc = acc; end
            blocks = [net.stem, net.res1.blk1, net.res2a.blk1, net.res3.blk1]
            f_rates = [mean(b.S) for b in blocks]
            @printf("  Epoch %3d | Test Acc: %5.2f%% (Best: %5.2f%%) | LR: %7.5f | Firing: %s\n", 
                    ep, acc*100, best_acc*100, lr, join([@sprintf("%.1f%%", f*100) for f in f_rates], " / "))
            serialize(checkpoint_path, (net=net, best_acc=best_acc, step=step, epoch=ep))
        end
    end
end

run_dvs_omega_v7()
