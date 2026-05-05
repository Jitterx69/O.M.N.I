#=
================================================================================
   CONTINUOUS-DEPTH NEURAL ODEs — Differential Equation Networks
  ==============================================================================
  This module implements a Continuous-Depth Neural ODE from scratch.
  
  Instead of stacking discrete layers, we model the hidden state as a continuous
  trajectory over "time" (depth), governed by the Ordinary Differential Equation:
      dh(t)/dt = tanh(W_h * h(t) + W_t * t + b)

  We evaluate the network by solving this ODE using an Explicit Euler Integrator.
  We backpropagate the gradients *through* the solver steps manually.
================================================================================
=#

include("../core.jl")

# ─── ODE Solver Layer ───────────────────────────────────────────────────────────
mutable struct ODESolverLayer
    dim::Int
    T::Float64       # Total Integration Time
    N::Int           # Number of steps
    dt::Float64
    
    W_h::Matrix{Float64}    # (dim, dim)
    W_t::Vector{Float64}    # (dim)
    b::Vector{Float64}      # (dim)
    
    # Adam states
    mW_h::Matrix{Float64}; vW_h::Matrix{Float64}
    mW_t::Vector{Float64}; vW_t::Vector{Float64}
    mb::Vector{Float64}; vb::Vector{Float64}
    
    # Gradients
    dW_h::Matrix{Float64}
    dW_t::Vector{Float64}
    db::Vector{Float64}
    
    # Forward pass cache for backprop
    H::Array{Float64, 3}    # Trajectory cache (dim, batch, N+1)
end

function ODESolverLayer(dim::Int; T=1.0, N=10)
    W_h = randn(dim, dim) .* sqrt(2.0 / dim)
    W_t = randn(dim) .* 0.1
    b = zeros(dim)
    
    ODESolverLayer(
        dim, Float64(T), N, Float64(T)/N,
        W_h, W_t, b,
        zeros(dim, dim), zeros(dim, dim),
        zeros(dim), zeros(dim),
        zeros(dim), zeros(dim),
        zeros(dim, dim), zeros(dim), zeros(dim),
        zeros(0, 0, 0)
    )
end

function count_layer_params(l::ODESolverLayer)
    length(l.W_h) + length(l.W_t) + length(l.b)
end

function fwd!(l::ODESolverLayer, X::Matrix{Float64})
    dim, B = size(X)
    
    # Allocate trajectory cache
    l.H = zeros(dim, B, l.N + 1)
    l.H[:, :, 1] = X
    
    # Explicit Euler Integration
    for k in 1:l.N
        t = (k - 1) * l.dt
        h_old = l.H[:, :, k]
        
        # dh/dt = tanh(W_h * h + W_t * t + b)
        pre_act = l.W_h * h_old .+ l.W_t .* t .+ l.b
        dh_dt = tanh.(pre_act)
        
        # h_{t+dt} = h_t + dt * dh_dt
        l.H[:, :, k+1] = h_old .+ l.dt .* dh_dt
    end
    
    # Return the state at final time T
    return l.H[:, :, l.N + 1]
end

function bwd!(l::ODESolverLayer, dY::Matrix{Float64}, λ::Float64, clip::Float64=1.0)
    dim, B = size(dY)
    
    # Reset gradients
    l.dW_h .= 0.0
    l.dW_t .= 0.0
    l.db .= 0.0
    
    dh_curr = dY
    
    # Backpropagate exactly through the Euler steps
    for k in l.N:-1:1
        t = (k - 1) * l.dt
        h_old = l.H[:, :, k]
        
        pre_act = l.W_h * h_old .+ l.W_t .* t .+ l.b
        act = tanh.(pre_act)
        
        # Derivative of tanh is 1 - tanh^2
        d_act = l.dt .* dh_curr .* (1.0 .- act .^ 2)
        
        # Accumulate parameter gradients
        l.dW_h .+= (d_act * h_old') ./ B
        l.dW_t .+= vec(sum(d_act .* t, dims=2)) ./ B
        l.db .+= vec(sum(d_act, dims=2)) ./ B
        
        # Gradient with respect to the input of this step (h_old)
        dh_old = dh_curr .+ l.W_h' * d_act
        dh_curr = dh_old
    end
    
    # Add L2 regularization
    l.dW_h .+= λ .* l.W_h
    l.dW_t .+= λ .* l.W_t
    
    # Gradient clipping
    gnorm = sqrt(sum(l.dW_h .^ 2) + sum(l.dW_t .^ 2) + sum(l.db .^ 2))
    if gnorm > clip
        scale = clip / gnorm
        l.dW_h .*= scale
        l.dW_t .*= scale
        l.db .*= scale
    end
    
    # Return gradient for the initial state (to pass to previous layer)
    return dh_curr
end

# ─── Neural ODE Network Struct ──────────────────────────────────────────────────
mutable struct NeuralODENet
    fc1::Dense
    ode::ODESolverLayer
    fc2::Dense
    training::Bool
end

function NeuralODENet(in_dim::Int, latent_dim::Int, out_dim::Int; T=1.0, N=10)
    fc1 = Dense(in_dim, latent_dim)
    ode = ODESolverLayer(latent_dim; T=T, N=N)
    fc2 = Dense(latent_dim, out_dim)
    NeuralODENet(fc1, ode, fc2, true)
end

function set_mode!(n::NeuralODENet, training::Bool)
    n.training = training
end

function forward!(n::NeuralODENet, X::Matrix{Float64})
    # 1. Project to continuous latent space
    h1 = lrelu.(fwd!(n.fc1, X))
    
    # 2. Evolve latent space over time via ODE solver
    h2 = fwd!(n.ode, h1)
    
    # 3. Project to output classes
    softmax_c(fwd!(n.fc2, h2))
end

function backward!(n::NeuralODENet, y_onehot, y_pred; l2=1e-4, clip=1.0)
    d = y_pred .- y_onehot
    
    d = bwd!(n.fc2, d, l2, clip)
    d = bwd!(n.ode, d, l2, clip)
    d = d .* lrelu_d.(n.fc1.W * n.fc1.inp .+ n.fc1.b) # lrelu deriv
    d = bwd!(n.fc1, d, l2, clip)
end

function adam_step!(n::NeuralODENet, lr, t; β1=0.9, β2=0.999, ε=1e-8)
    bc1 = 1.0 - β1^t
    bc2 = 1.0 - β2^t
    
    # Function to update generic Dense layers
    function update_dense!(l::Dense)
        @. l.mW = β1 * l.mW + (1-β1) * l.dW
        @. l.vW = β2 * l.vW + (1-β2) * l.dW^2
        @. l.mb = β1 * l.mb + (1-β1) * l.db
        @. l.vb = β2 * l.vb + (1-β2) * l.db^2
        @. l.W -= lr * (l.mW/bc1) / (sqrt(l.vW/bc2) + ε)
        @. l.b -= lr * (l.mb/bc1) / (sqrt(l.vb/bc2) + ε)
    end
    
    update_dense!(n.fc1)
    update_dense!(n.fc2)
    
    # Update ODE layer
    l = n.ode
    @. l.mW_h = β1 * l.mW_h + (1-β1) * l.dW_h
    @. l.vW_h = β2 * l.vW_h + (1-β2) * l.dW_h^2
    @. l.W_h -= lr * (l.mW_h/bc1) / (sqrt(l.vW_h/bc2) + ε)
    
    @. l.mW_t = β1 * l.mW_t + (1-β1) * l.dW_t
    @. l.vW_t = β2 * l.vW_t + (1-β2) * l.dW_t^2
    @. l.W_t -= lr * (l.mW_t/bc1) / (sqrt(l.vW_t/bc2) + ε)
    
    @. l.mb = β1 * l.mb + (1-β1) * l.db
    @. l.vb = β2 * l.vb + (1-β2) * l.db^2
    @. l.b -= lr * (l.mb/bc1) / (sqrt(l.vb/bc2) + ε)
end

function count_params(n::NeuralODENet)
    length(n.fc1.W) + length(n.fc1.b) + 
    count_layer_params(n.ode) + 
    length(n.fc2.W) + length(n.fc2.b)
end

# ─── Training Execution ─────────────────────────────────────────────────────────

function run_neural_ode()
    println("\n" * "╔" * "═"^78 * "╗")
    println("║" * " "^15 * "  CONTINUOUS-DEPTH NEURAL ODE NETWORK" * " "^16 * "║")
    println("║" * " "^16 * "Explicit Euler Integrator (10 Time Steps)" * " "^15 * "║")
    println("╚" * "═"^78 * "╝")
    
    # 1. Load data
    X_train, y_train, X_test, y_test = load_data(; top_k=300, verbose=false)
    nf = size(X_train, 1)

    # 2. Build Neural ODE
    # 300 features -> 16 continuous dimensions -> 2 classes
    LATENT_DIM = 16
    STEPS = 10
    net = NeuralODENet(nf, LATENT_DIM, 2; T=1.0, N=STEPS)
    
    params = count_params(net)
    println("\n Architecture: Dense($nf → $LATENT_DIM) → ODE(steps=$STEPS) → Dense($LATENT_DIM → 2)")
    println("   Total Parameters: $params")
    println("  " * "─"^70)

    # 3. Train
    n = size(X_train, 2)
    step = 0
    epochs = 100
    batch_size = 32
    
    best_acc = 0.0
    best_ep = 0
    
    t0 = now()
    
    for epoch in 1:epochs
        # Cosine learning rate
        lr = 2e-3 * 0.5 * (1 + cos(π * epoch / epochs))
        
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
        
        # Evaluate
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
    
    # 4. Final Evaluation
    m = evaluate(net, X_test, y_test)
    c = m.cm
    
    println("\n" * "─"^78)
    println("   NEURAL ODE TRAINING COMPLETE")
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
    
    # Save results
    mkpath(joinpath(CORE_PROJECT_DIR, "results"))
    open(joinpath(CORE_PROJECT_DIR, "results", "neural_ode_results.txt"), "w") do io
        println(io, "Continuous-Depth Neural ODE Results — $(now())")
        println(io, "="^60)
        println(io, "Architecture: Dense($nf → $LATENT_DIM) → ODE(steps=$STEPS) → Dense($LATENT_DIM → 2)")
        println(io, "Parameters: $params")
        @printf(io, "\nBest Accuracy: %.4f\n", best_acc)
        @printf(io, "Final F1:      %.4f\n", m.f1)
        @printf(io, "Final AUC:     %.4f\n", m.auc)
    end
    
    println("\n   Neural ODE results saved to: results/neural_ode_results.txt")
end

run_neural_ode()
