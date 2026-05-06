using Statistics, Random, Printf, Dates

include("../core.jl")
include("kan.jl")
include("hebbian.jl")

mutable struct OmegaNet
    kan_enc::ChebKANLayer
    plastic::PlasticDense
    recon_head::Dense      # For Self-Supervised pre-training
    predict_head::Dense    # For Directional forecasting
    training::Bool
end

function OmegaNet(in_dim::Int, hidden::Int, out_dim::Int, degree::Int=3)
    kan = ChebKANLayer(in_dim, hidden, degree)
    pla = PlasticDense(hidden, hidden)
    rec = Dense(hidden, in_dim)
    pre = Dense(hidden, out_dim)
    OmegaNet(kan, pla, rec, pre, true)
end

function forward_recon!(n::OmegaNet, X::Matrix{Float64})
    h1 = lrelu.(fwd!(n.kan_enc, X))
    h2 = lrelu.(fwd!(n.plastic, h1))
    
    # Self-Supervised Trace Update (Relate current state to latent)
    if n.training
        update_trace!(n.plastic, h1, h2)
    end
    
    fwd!(n.recon_head, h2) # Reconstruct X
end

function forward_predict!(n::OmegaNet, X::Matrix{Float64})
    h1 = lrelu.(fwd!(n.kan_enc, X))
    h2 = lrelu.(fwd!(n.plastic, h1))
    
    if n.training
        update_trace!(n.plastic, h1, h2)
    end
    
    softmax_c(fwd!(n.predict_head, h2))
end

function backward_recon!(n::OmegaNet, X_target, X_pred; lr=1e-3, t=1, l2=1e-4)
    # MSE Loss for reconstruction
    d = X_pred .- X_target
    B = size(X_target, 2)
    
    # Head Grad
    d = bwd!(n.recon_head, d, l2, 1.0)
    
    # Plastic Grad
    pre_act_p = n.plastic.W * n.plastic.inp .+ n.plastic.b .+ 
                n.plastic.alpha .* n.plastic.H * n.plastic.inp
    d = d .* lrelu_d.(pre_act_p)
    d = bwd!(n.plastic, d, l2, 1.0)
    
    # KAN Grad
    bwd!(n.kan_enc, d, l2, 1.0)
    
    # Update
    adam_step_omega!(n, lr, t, true)
end

function backward_predict!(n::OmegaNet, y_oh, y_pred; lr=1e-3, t=1, l2=1e-4)
    d = y_pred .- y_oh
    
    d = bwd!(n.predict_head, d, l2, 1.0)
    
    pre_act_p = n.plastic.W * n.plastic.inp .+ n.plastic.b .+ 
                n.plastic.alpha .* n.plastic.H * n.plastic.inp
    d = d .* lrelu_d.(pre_act_p)
    d = bwd!(n.plastic, d, l2, 1.0)
    
    bwd!(n.kan_enc, d, l2, 1.0)
    
    adam_step_omega!(n, lr, t, false)
end

function adam_step_omega!(n::OmegaNet, lr, t, is_recon::Bool)
    bc1 = 1.0 - 0.9^t
    bc2 = 1.0 - 0.999^t
    
    # Update KAN
    l = n.kan_enc
    @. l.mW_base = 0.9 * l.mW_base + (1-0.9) * l.dW_base
    @. l.vW_base = 0.999 * l.vW_base + (1-0.999) * l.dW_base^2
    @. l.W_base -= lr * (l.mW_base/bc1) / (sqrt(l.vW_base/bc2) + 1e-8)
    @. l.mC_flat = 0.9 * l.mC_flat + (1-0.9) * l.dC_flat
    @. l.vC_flat = 0.999 * l.vC_flat + (1-0.999) * l.dC_flat^2
    @. l.C_flat -= lr * (l.mC_flat/bc1) / (sqrt(l.vC_flat/bc2) + 1e-8)
    
    # Update Plastic
    p = n.plastic
    @. p.mW = 0.9 * p.mW + (1-0.9) * p.dW
    @. p.vW = 0.999 * p.vW + (1-0.999) * p.dW^2
    @. p.W -= lr * (p.mW/bc1) / (sqrt(p.vW/bc2) + 1e-8)
    @. p.malpha = 0.9 * p.malpha + (1-0.9) * p.dalpha
    @. p.valpha = 0.999 * p.valpha + (1-0.999) * p.dalpha^2
    @. p.alpha -= lr * (p.malpha/bc1) / (sqrt(p.valpha/bc2) + 1e-8)
    
    # Update Heads
    h = is_recon ? n.recon_head : n.predict_head
    @. h.mW = 0.9 * h.mW + (1-0.9) * h.dW
    @. h.vW = 0.999 * h.vW + (1-0.999) * h.dW^2
    @. h.W -= lr * (h.mW/bc1) / (sqrt(h.vW/bc2) + 1e-8)
end
