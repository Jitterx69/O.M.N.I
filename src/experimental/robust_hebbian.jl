using Statistics, Random, Printf

include("../core.jl")
include("../modules/uncertainty.jl")
include("hebbian.jl")

mutable struct CrossAssetAttention
    n_assets::Int
    n_feats::Int
    d_model::Int
    
    W_q::Matrix{Float64}
    W_k::Matrix{Float64}
    W_v::Matrix{Float64}
    W_o::Matrix{Float64}
    
    # Optimizer states
    m_q::Matrix{Float64}; v_q::Matrix{Float64}
    m_k::Matrix{Float64}; v_k::Matrix{Float64}
    m_v::Matrix{Float64}; v_v::Matrix{Float64}
    m_o::Matrix{Float64}; v_o::Matrix{Float64}
end

function CrossAssetAttention(n_assets, n_feats, d_model)
    k = sqrt(1.0 / n_feats)
    CrossAssetAttention(
        n_assets, n_feats, d_model,
        randn(d_model, n_feats) .* k,
        randn(d_model, n_feats) .* k,
        randn(d_model, n_feats) .* k,
        randn(n_feats, d_model) .* k,
        zeros(d_model, n_feats), zeros(d_model, n_feats),
        zeros(d_model, n_feats), zeros(d_model, n_feats),
        zeros(d_model, n_feats), zeros(d_model, n_feats),
        zeros(n_feats, d_model), zeros(n_feats, d_model)
    )
end

function fwd_attn(l::CrossAssetAttention, X::Array{Float64, 3})
    # X: (features, assets, batch)
    d_feat, n_assets, B = size(X)
    
    # Project to Q, K, V
    Q = zeros(l.d_model, n_assets, B)
    K = zeros(l.d_model, n_assets, B)
    V = zeros(l.d_model, n_assets, B)
    
    for b in 1:B
        Q[:, :, b] = l.W_q * X[:, :, b]
        K[:, :, b] = l.W_k * X[:, :, b]
        V[:, :, b] = l.W_v * X[:, :, b]
    end
    
    # Softmax Dot-Product Attention
    scale = sqrt(l.d_model)
    out = zeros(d_feat, n_assets, B)
    
    for b in 1:B
        # Attn: (n_assets, n_assets)
        scores = (K[:, :, b]' * Q[:, :, b]) ./ scale
        # Simple softmax over columns
        mx = maximum(scores, dims=1)
        exp_s = exp.(scores .- mx)
        attn = exp_s ./ sum(exp_s, dims=1)
        
        # Z: (d_model, n_assets)
        Z = V[:, :, b] * attn
        out[:, :, b] = l.W_o * Z
    end
    
    return out
end

mutable struct RobustPlasticNet
    attn::CrossAssetAttention
    encoder::Dense
    plastic::PlasticDense
    head::Dense
    drop_rate::Float64
    training::Bool
end

function RobustPlasticNet(n_assets, n_feats_per_asset, n_days, hidden)
    # Total input: n_assets * n_feats_per_asset * n_days
    # But attention works on assets
    attn = CrossAssetAttention(n_assets, n_feats_per_asset * n_days, 64)
    enc = Dense(n_assets * n_feats_per_asset * n_days, hidden)
    pla = PlasticDense(hidden, hidden)
    hd = Dense(hidden, 2)
    RobustPlasticNet(attn, enc, pla, hd, 0.4, true)
end

function forward_robust!(net::RobustPlasticNet, X::Matrix{Float64})
    B = size(X, 2)
    # Reshape for attention: (features_per_asset*days, n_assets, batch)
    # Our loader produced: [asset1_ret, asset2_ret..., asset1_vol, ...]
    # For simplicity, we'll treat the 360-dim vector as a grouped asset signal
    X_reshaped = reshape(X, (Int(size(X, 1) / 4), 4, B))
    
    # 1. Attention (Cross-Asset Relational Reasoning)
    X_attn = fwd_attn(net.attn, X_reshaped)
    h_in = reshape(X_attn, (size(X, 1), B))
    
    # 2. Encoder with Dropout
    z1 = fwd!(net.encoder, h_in)
    if net.training
        mask = rand(size(z1)...) .> net.drop_rate
        z1 = z1 .* mask ./ (1.0 - net.drop_rate)
    end
    h1 = lrelu.(z1)
    
    # 3. Plastic Layer
    z2 = fwd!(net.plastic, h1)
    h2 = lrelu.(z2)
    
    # Hebbian adaptation during training
    if net.training
        update_trace!(net.plastic, h1, h2)
    end
    
    net.plastic.out_act = h2
    return softmax_c(fwd!(net.head, h2))
end

function forward_adapted!(net::RobustPlasticNet, X::Matrix{Float64}, uncertainty_scale::Float64=1.0)
    # Inference with Dynamic Uncertainty-Gated Hebbian Adaptation
    # If uncertainty is high, we update traces aggressively (10x faster)
    B = size(X, 2)
    X_reshaped = reshape(X, (Int(size(X, 1) / 4), 4, B))
    X_attn = fwd_attn(net.attn, X_reshaped)
    h_in = reshape(X_attn, (size(X, 1), B))
    
    z1 = fwd!(net.encoder, h_in)
    h1 = lrelu.(z1)
    z2 = fwd!(net.plastic, h1)
    h2 = lrelu.(z2)
    
    # Gated Adaptation: Scale (1-DECAY) based on uncertainty
    # TRACE_DECAY is constant, but we can call update_trace multiple times
    # or just implement a custom one here
    hebb = (h2 * h1') ./ B
    clamp!(hebb, -TRACE_CLAMP, TRACE_CLAMP)
    
    # The more uncertain we are, the more we shift the weights
    learning_factor = (1.0 - TRACE_DECAY) * uncertainty_scale
    net.plastic.H .= (1.0 - learning_factor) .* net.plastic.H .+ learning_factor .* hebb
    
    return softmax_c(fwd!(net.head, h2))
end
