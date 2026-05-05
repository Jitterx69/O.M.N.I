include("../core.jl")

σ(x) = 1.0 / (1.0 + exp(-clamp(x, -50.0, 50.0)))
silu(x) = x * σ(x)
silu_d(x) = begin
    s = σ(x)
    s + x * s * (1.0 - s)
end

mutable struct ChebKANLayer
    in_dim::Int
    out_dim::Int
    degree::Int
    
    W_base::Matrix{Float64}
    C_flat::Matrix{Float64}
    
    mW_base::Matrix{Float64}; vW_base::Matrix{Float64}
    mC_flat::Matrix{Float64}; vC_flat::Matrix{Float64}
    
    dW_base::Matrix{Float64}
    dC_flat::Matrix{Float64}
    
    X_raw::Matrix{Float64}
    X_norm::Matrix{Float64}
    base_act::Matrix{Float64}
    T::Array{Float64, 3}
    dT::Array{Float64, 3}
    T_flat::Matrix{Float64}
end

function ChebKANLayer(in_dim::Int, out_dim::Int, degree::Int)
    W_base = randn(out_dim, in_dim) .* sqrt(2.0 / in_dim)
    
    C_flat = randn(out_dim, in_dim * (degree + 1)) .* 0.1
    
    ChebKANLayer(
        in_dim, out_dim, degree,
        W_base, C_flat,
        zeros(out_dim, in_dim), zeros(out_dim, in_dim),
        zeros(out_dim, in_dim * (degree + 1)), zeros(out_dim, in_dim * (degree + 1)),
        zeros(out_dim, in_dim), zeros(out_dim, in_dim * (degree + 1)),
        zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0,0), zeros(0,0,0), zeros(0,0)
    )
end

function count_layer_params(l::ChebKANLayer)
    length(l.W_base) + length(l.C_flat)
end

function fwd!(l::ChebKANLayer, X::Matrix{Float64})
    in_dim, B = size(X)
    D = l.degree
    
    l.X_raw = X
    l.X_norm = tanh.(X)
    l.base_act = silu.(X)
    
    Y_base = l.W_base * l.base_act
    
    l.T = zeros(D + 1, in_dim, B)
    l.dT = zeros(D + 1, in_dim, B)
    
    l.T[1, :, :] .= 1.0
    l.dT[1, :, :] .= 0.0
    
    if D >= 1
        l.T[2, :, :] = l.X_norm
        l.dT[2, :, :] .= 1.0
    end
    
    for k in 3:(D + 1)
        l.T[k, :, :] = 2.0 .* l.X_norm .* l.T[k-1, :, :] .- l.T[k-2, :, :]
        l.dT[k, :, :] = 2.0 .* l.T[k-1, :, :] .+ 2.0 .* l.X_norm .* l.dT[k-1, :, :] .- l.dT[k-2, :, :]
    end
    
    l.T_flat = reshape(l.T, in_dim * (D + 1), B)
    
    Y_cheb = l.C_flat * l.T_flat
    
    return Y_base .+ Y_cheb
end

function bwd!(l::ChebKANLayer, dY::Matrix{Float64}, λ::Float64, clip::Float64=1.0)
    in_dim, B = size(l.X_raw)
    D = l.degree
    
    l.dW_base = (dY * l.base_act') ./ B .+ λ .* l.W_base
    l.dC_flat = (dY * l.T_flat') ./ B .+ λ .* l.C_flat
    
    gnorm_base = sqrt(sum(l.dW_base .^ 2))
    if gnorm_base > clip
        l.dW_base .*= (clip / gnorm_base)
    end
    gnorm_C = sqrt(sum(l.dC_flat .^ 2))
    if gnorm_C > clip
        l.dC_flat .*= (clip / gnorm_C)
    end
    
    
    d_base_act = l.W_base' * dY   # (in, B)
    dX_base = d_base_act .* silu_d.(l.X_raw)
    
    d_T_flat = l.C_flat' * dY     # (in * (D+1), B)
    d_T = reshape(d_T_flat, D + 1, in_dim, B)
    
    dX_norm = reshape(sum(d_T .* l.dT, dims=1), in_dim, B)
    
    dX_cheb = dX_norm .* (1.0 .- l.X_norm .^ 2)
    
    return dX_base .+ dX_cheb
end

mutable struct KANNet
    layers::Vector{ChebKANLayer}
    training::Bool
end

function KANNet(sizes::Vector{Int}, degree::Int=3)
    layers = [ChebKANLayer(sizes[i], sizes[i+1], degree) for i in 1:length(sizes)-1]
    KANNet(layers, true)
end

function set_mode!(n::KANNet, training::Bool)
    n.training = training
end

function forward!(n::KANNet, X::Matrix{Float64})
    h = X
    for i in 1:length(n.layers)-1
        h = fwd!(n.layers[i], h)
    end
    softmax_c(fwd!(n.layers[end], h))
end

function backward!(n::KANNet, y_onehot, y_pred; l2=1e-4, clip=1.0)
    d = y_pred .- y_onehot
    
    for i in length(n.layers):-1:1
        d = bwd!(n.layers[i], d, l2, clip)
    end
end

function adam_step!(n::KANNet, lr, t; β1=0.9, β2=0.999, ε=1e-8)
    bc1 = 1.0 - β1^t
    bc2 = 1.0 - β2^t
    for l in n.layers
        @. l.mW_base = β1 * l.mW_base + (1-β1) * l.dW_base
        @. l.vW_base = β2 * l.vW_base + (1-β2) * l.dW_base^2
        @. l.W_base -= lr * (l.mW_base/bc1) / (sqrt(l.vW_base/bc2) + ε)
        
        @. l.mC_flat = β1 * l.mC_flat + (1-β1) * l.dC_flat
        @. l.vC_flat = β2 * l.vC_flat + (1-β2) * l.dC_flat^2
        @. l.C_flat -= lr * (l.mC_flat/bc1) / (sqrt(l.vC_flat/bc2) + ε)
    end
end

function count_params(n::KANNet)
    sum(count_layer_params(l) for l in n.layers)
end

function run_kan()
    println("\n" * "╔" * "═"^78 * "╗")
    println("║" * " "^11 * " KOLMOGOROV-ARNOLD NETWORKS (KANs) — 2024 Architecture" * " "^11 * "║")
    println("║" * " "^15 * "Edge-Based Learnable Functions (Chebyshev Polynomials)" * " "^11 * "║")
    println("╚" * "═"^78 * "╝")
    
    X_train, y_train, X_test, y_test = load_data(; top_k=100, verbose=false)
    nf = size(X_train, 1)

    DEGREE = 4
    sizes = [nf, 8, 2]
    net = KANNet(sizes, DEGREE)
    
    params = count_params(net)
    println("\n Architecture: KAN $(join(sizes, " → ")) | Chebyshev Degree: $DEGREE")
    println("   Total Parameters: $params")
    println("   (Notice how few parameters this uses compared to standard MLPs!)")
    println("  " * "─"^70)

    n = size(X_train, 2)
    step = 0
    epochs = 80
    batch_size = 32
    
    best_acc = 0.0
    best_ep = 0
    
    t0 = now()
    
    for epoch in 1:epochs
        lr = 3e-3 * 0.5 * (1 + cos(π * epoch / epochs))
        
        perm = randperm(n)
        for start in 1:batch_size:n
            step += 1
            idx = perm[start:min(start+batch_size-1, n)]
            
            X_batch = X_train[:, idx]
            y_oh = onehot(y_train[idx])
            
            ŷ = forward!(net, X_batch)
            backward!(net, y_oh, ŷ; l2=1e-4, clip=1.0)
            adam_step!(net, lr, step)
        end
        
        if epoch == 1 || epoch % 10 == 0 || epoch == epochs
            m_train = evaluate(net, X_train, y_train)
            m_test = evaluate(net, X_test, y_test)
            
            if m_test.acc > best_acc
                best_acc = m_test.acc
                best_ep = epoch
            end
            
            @printf("  Ep %3d │ TrL %.4f TrA %5.1f%% │ TeL %.4f TeA %5.1f%% │ F1 %.3f │ LR %.1e\n",
                    epoch, m_train.loss, m_train.acc*100, m_test.loss, m_test.acc*100, m_test.f1, lr)
        end
    end
    
    elapsed = Dates.value(now() - t0) / 1000
    
    m = evaluate(net, X_test, y_test)
    c = m.cm
    
    println("\n" * "─"^78)
    println("   KAN TRAINING COMPLETE")
    println("─"^78)
    @printf("  Training Time: %.1fs\n", elapsed)
    @printf("  Best Accuracy: %.2f%% (at epoch %d)\n", best_acc * 100, best_ep)
    @printf("  Final F1:      %.4f\n", m.f1)
    @printf("  Final AUC:     %.4f\n", m.auc)
    
    println("\n  Confusion Matrix:")
    println("  ┌──────────┬──────────┬──────────┐")
    println("  │          │ Pred: 0  │ Pred: 1  │")
    println("  ├──────────┼──────────┼──────────┤")
    @printf("  │ True: 0  │  %4d    │  %4d    │\n", c.tn, c.fp)
    @printf("  │ True: 1  │  %4d    │  %4d    │\n", c.fn, c.tp)
    println("  └──────────┴──────────┴──────────┘")
    
    mkpath(joinpath(CORE_PROJECT_DIR, "results"))
    open(joinpath(CORE_PROJECT_DIR, "results", "kan_results.txt"), "w") do io
        println(io, "Kolmogorov-Arnold Network Results — $(now())")
        println(io, "="^60)
        println(io, "Architecture: KAN $(join(sizes, " → "))")
        println(io, "Chebyshev Degree: $DEGREE")
        println(io, "Parameters: $params")
        @printf(io, "\nBest Accuracy: %.4f\n", best_acc)
        @printf(io, "Final F1:      %.4f\n", m.f1)
        @printf(io, "Final AUC:     %.4f\n", m.auc)
    end
    
    println("\n   KAN results saved to: results/kan_results.txt")
end

run_kan()
