include("../core.jl")

const SURROGATE_ALPHA = 2.0
const THRESHOLD = 1.0
const LEAK = 0.8

function surrogate_gradient(x)
    return 1.0 / (1.0 + SURROGATE_ALPHA * abs(x))^2
end

mutable struct LIFLayer
    ni::Int
    no::Int
    W::Matrix{Float64}
    b::Vector{Float64}
    
    dW::Matrix{Float64}
    db::Vector{Float64}
    
    mW::Matrix{Float64}; vW::Matrix{Float64}
    mb::Vector{Float64}; vb::Vector{Float64}
    
    U::Array{Float64, 3}
    S::Array{Float64, 3}
    
    inp_spikes::Array{Float64, 3}
end

function LIFLayer(ni, no)
    scale = sqrt(2.0 / ni)
    LIFLayer(
        ni, no,
        randn(no, ni) .* scale,
        zeros(no),
        zeros(no, ni), zeros(no),
        zeros(no, ni), zeros(no, ni),
        zeros(no), zeros(no),
        zeros(0, 0, 0), zeros(0, 0, 0),
        zeros(0, 0, 0)
    )
end

function fwd!(l::LIFLayer, X_spikes::Array{Float64, 3})
    no, B, T = size(X_spikes, 1), size(X_spikes, 2), size(X_spikes, 3)
    l.inp_spikes = X_spikes
    
    l.U = zeros(l.no, B, T + 1)
    l.S = zeros(l.no, B, T)
    
    for t in 1:T
        curr_in = l.W * X_spikes[:, :, t] .+ l.b
        l.U[:, :, t+1] = LEAK .* l.U[:, :, t] .+ (1.0 - LEAK) .* curr_in
        
        spikes = Float64.(l.U[:, :, t+1] .> THRESHOLD)
        l.S[:, :, t] = spikes
        
        l.U[:, :, t+1] .-= spikes .* THRESHOLD
    end
    
    return l.S
end

function bwd!(l::LIFLayer, dS_next::Array{Float64, 3}, λ::Float64, clip::Float64=1.0)
    no, B, T = size(dS_next)
    
    l.dW .= 0.0
    l.db .= 0.0
    
    dU = zeros(l.no, B)
    dX = zeros(size(l.inp_spikes))
    
    for t in T:-1:1
        grad_surr = surrogate_gradient.(l.U[:, :, t+1] .- THRESHOLD)
        
        dU_curr = dS_next[:, :, t] .+ dU .* LEAK
        dU_fire = dU_curr .* grad_surr
        
        dI = dU_fire .* (1.0 - LEAK)
        
        l.dW .+= (dI * l.inp_spikes[:, :, t]') ./ B
        l.db .+= vec(sum(dI, dims=2)) ./ B
        
        dX[:, :, t] = l.W' * dI
        
        dU = dU_curr .- dU_fire .* THRESHOLD
    end
    
    l.dW .+= λ .* l.W
    
    gnorm = sqrt(sum(l.dW .^ 2) + sum(l.db .^ 2))
    if gnorm > clip
        s = clip / gnorm
        l.dW .*= s; l.db .*= s
    end
    
    return dX
end

mutable struct SpikingNet
    lif1::LIFLayer
    lif2::LIFLayer
    T::Int
    training::Bool
end

function SpikingNet(in_dim, hidden, out_dim; T=15)
    SpikingNet(
        LIFLayer(in_dim, hidden),
        LIFLayer(hidden, out_dim),
        T, true
    )
end

function encode_rate(X::Matrix{Float64}, T::Int)
    dim, B = size(X)
    X_norm = (X .- minimum(X)) ./ (maximum(X) - minimum(X) + 1e-8)
    spikes = zeros(dim, B, T)
    for t in 1:T
        spikes[:, :, t] = Float64.(rand(dim, B) .< X_norm)
    end
    return spikes
end

function forward!(n::SpikingNet, X::Matrix{Float64})
    in_spikes = encode_rate(X, n.T)
    h_spikes = fwd!(n.lif1, in_spikes)
    out_spikes = fwd!(n.lif2, h_spikes)
    
    spike_counts = sum(out_spikes, dims=3)[:, :, 1]
    return softmax_c(spike_counts)
end

function backward!(n::SpikingNet, y_oh, y_pred; l2=1e-4, clip=1.0)
    B = size(y_oh, 2)
    
    d_counts = (y_pred .- y_oh) ./ n.T
    
    dS_out = zeros(n.lif2.no, B, n.T)
    for t in 1:n.T
        dS_out[:, :, t] = d_counts
    end
    
    dX_h = bwd!(n.lif2, dS_out, l2, clip)
    bwd!(n.lif1, dX_h, l2, clip)
end

function adam_step!(n::SpikingNet, lr, t; β1=0.9, β2=0.999, ε=1e-8)
    bc1 = 1.0 - β1^t
    bc2 = 1.0 - β2^t
    
    for l in [n.lif1, n.lif2]
        @. l.mW = β1 * l.mW + (1-β1) * l.dW
        @. l.vW = β2 * l.vW + (1-β2) * l.dW^2
        @. l.mb = β1 * l.mb + (1-β1) * l.db
        @. l.vb = β2 * l.vb + (1-β2) * l.db^2
        @. l.W -= lr * (l.mW/bc1) / (sqrt(l.vW/bc2) + ε)
        @. l.b -= lr * (l.mb/bc1) / (sqrt(l.vb/bc2) + ε)
    end
end

function evaluate(net::SpikingNet, X, y)
    ŷ = forward!(net, X)
    preds = [argmax(ŷ[:, i]) - 1 for i in 1:size(ŷ, 2)]
    yoh = onehot(y)
    acc = mean(preds .== y)
    tp = sum((preds .== 1) .& (y .== 1))
    fp = sum((preds .== 1) .& (y .== 0))
    fn = sum((preds .== 0) .& (y .== 1))
    tn = sum((preds .== 0) .& (y .== 0))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    loss = xent(ŷ, yoh)
    auc = compute_auc(vec(ŷ[2, :]), y)
    (acc=acc, prec=prec, rec=rec, f1=f1, loss=loss, auc=auc,
     cm=(tp=tp, fp=fp, fn=fn, tn=tn))
end

function run_snn()
    println("\n" * "═"^80)
    println("   SPIKING NEURAL NETWORK (SNN) — Neuromorphic Computing Engine")
    println("   LIF Neurons | Fast Sigmoid Surrogate Gradients | Rate Encoding")
    println("═"^80)

    X_train, y_train, X_test, y_test = load_data(; top_k=300, verbose=false)
    nf = size(X_train, 1)
    
    HIDDEN = 128
    T_STEPS = 15
    net = SpikingNet(nf, HIDDEN, 2; T=T_STEPS)
    
    params = length(net.lif1.W) + length(net.lif1.b) + length(net.lif2.W) + length(net.lif2.b)
    println("\n Architecture: RateEncoder($nf) -> LIF($nf -> $HIDDEN) -> LIF($HIDDEN -> 2)")
    println("   Simulation Time Steps: $T_STEPS")
    println("   Total Parameters: $params")
    println("  " * "-"^70)

    n = size(X_train, 2)
    step = 0
    epochs = 50
    batch_size = 32
    
    best_acc = 0.0
    t0 = now()

    for epoch in 1:epochs
        lr = 2e-3 * 0.5 * (1 + cos(pi * epoch / epochs))
        
        perm = randperm(n)
        for start in 1:batch_size:n
            step += 1
            idx = perm[start:min(start+batch_size-1, n)]
            
            X_batch = X_train[:, idx]
            y_oh = onehot(y_train[idx])
            
            y_pred = forward!(net, X_batch)
            backward!(net, y_oh, y_pred; l2=1e-4, clip=1.0)
            adam_step!(net, lr, step)
        end
        
        if epoch == 1 || epoch % 5 == 0 || epoch == epochs
            m_te = evaluate(net, X_test, y_test)
            best_acc = max(best_acc, m_te.acc)
            @printf("  Ep %2d | Test Acc: %5.1f%% | F1: %.3f | LR: %.1e\n", 
                    epoch, m_te.acc*100, m_te.f1, lr)
        end
    end
    
    elapsed = Dates.value(now() - t0) / 1000.0
    m = evaluate(net, X_test, y_test)
    
    println("\n" * "-"^78)
    println("   SNN TRAINING COMPLETE")
    println("-"^78)
    @printf("  Training Time:   %.1fs\n", elapsed)
    @printf("  Best Accuracy:   %.2f%%\n", best_acc * 100)
    @printf("  Final F1:        %.4f\n", m.f1)

    mkpath(joinpath(CORE_PROJECT_DIR, "results"))
    open(joinpath(CORE_PROJECT_DIR, "results", "snn_results.txt"), "w") do io
        println(io, "Spiking Neural Network Results -- $(now())")
        println(io, "="^60)
        println(io, "Architecture: 2-Layer LIF ($nf -> $HIDDEN -> 2)")
        println(io, "Time Steps: $T_STEPS")
        println(io, "Parameters: $params")
        @printf(io, "\nBest Accuracy: %.4f\n", best_acc)
        @printf(io, "Final F1:      %.4f\n", m.f1)
    end
    println("\n   Results saved to: results/snn_results.txt")
end

run_snn()
