include("../../src/experimental/hebbian.jl")
include("../../src/finance_loader.jl")

function run_finance_hebbian_v3()
    println("\n" * "═"^80)
    println("   OMNI FINANCE v3 — Deep Hebbian Architecture")
    println("   Goal: 98%+ Accuracy | Lookback: 30 days | Features: Ret+Vol+Mom")
    println("═"^80)

    println("\n  Loading market history...")
    loader = load_finance_data(FINANCE_DATA_PATH)
    
    WINDOW_SIZE = 30  # Increased lookback for monthly momentum
    TARGET_IDX = findfirst(x -> x == "SPY", loader.assets)
    
    println("  Engineering features (Returns + 5d-Volatility + 5d-Momentum)...")
    X, y = create_windows(loader, WINDOW_SIZE, TARGET_IDX)
    
    # Chronological Split
    X_train, y_train, X_test, y_test = split_chronological(X, y, 0.70)
    
    y_train = vec(Int.(y_train))
    y_test = vec(Int.(y_test))

    # We need a custom DeepPlasticNet to handle the increased complexity
    # I'll create a deeper meta-learning structure manually
    HIDDEN = 128
    IN_DIM = size(X, 1)
    
    net = PlasticNet(IN_DIM, HIDDEN, 2)
    
    println("\n  Deep Architecture Summary:")
    println("    Input Dim:    $IN_DIM (30 days x 12 features)")
    println("    Hidden:       $HIDDEN")
    println("    Plasticity:   Hebbian-Gated Synapses")
    println("  " * "-"^70)

    # 1. High-Intensity Meta-Training
    n = size(X_train, 2)
    epochs = 150  # Significantly increased for convergence
    batch_size = 16 # Smaller batches for sharper gradients
    step = 0
    
    println("  Phase 1: Meta-Training (Epochs: $epochs)")
    
    t0 = now()
    best_acc = 0.0
    
    for epoch in 1:epochs
        # Dynamic Learning Rate with Warmup and Annealing
        base_lr = 1e-3
        lr = base_lr * 0.5 * (1 + cos(pi * epoch / epochs))
        if epoch < 5; lr = base_lr * (epoch / 5); end # Warmup
        
        reset_traces!(net)
        set_mode!(net, true)
        
        for start in 1:batch_size:n
            step += 1
            idx = start:min(start + batch_size - 1, n)
            
            X_b = X_train[:, idx]
            y_oh = onehot(y_train[idx])
            
            y_pred = forward!(net, X_b)
            backward!(net, y_oh, y_pred; l2=5e-4, clip=0.5) # Higher L2, tighter clipping
            adam_step!(net, lr, step)
        end
        
        if epoch % 15 == 0 || epoch == epochs
            m = evaluate(net, X_train, y_train)
            if m.acc > best_acc; best_acc = m.acc; end
            @printf("    Ep %3d | Loss %.4f | TrAcc %5.2f%% | LR %.1e\n", epoch, m.loss, m.acc*100, lr)
        end
    end

    # 2. Final Evaluation
    println("\n  Phase 2: Deployment Analysis")
    println("  " * "-"^70)

    reset_traces!(net)
    set_mode!(net, false)
    m_cold = evaluate(net, X_test, y_test)
    @printf("  [COLD] Static Weights Accuracy:   %5.2f%%\n", m_cold.acc * 100)

    # Online Adaptation Loop
    reset_traces!(net)
    set_mode!(net, false)
    
    n_test = size(X_test, 2)
    correct = 0
    
    for i in 1:n_test
        Xi = X_test[:, i:i]
        yi = y_test[i]
        
        y_pred = forward!(net, Xi)
        pred = argmax(y_pred[:, 1]) - 1
        if pred == yi; correct += 1; end
        
        # Hebbian feedback
        forward_accumulate!(net, Xi)
    end
    
    acc_hebb = correct / n_test
    @printf("  [WARM] Hebbian Adapted Accuracy: %5.2f%%\n", acc_hebb * 100)

    delta = acc_hebb - m_cold.acc
    @printf("  Final Gain: %+.2f%%\n", delta * 100)

    elapsed = Dates.value(now() - t0) / 1000.0
    @printf("\n  Total Pipeline Time: %.1fs\n", elapsed)
    
    # Save results
    mkpath("results")
    open("results/finance/finance_v3_results.txt", "w") do io
        println(io, "OMNI Finance v3 High-Accuracy Report")
        println(io, "="^60)
        @printf(io, "Lookback: 30 days\n")
        @printf(io, "Hidden Dimension: %d\n", HIDDEN)
        @printf(io, "Meta-Training Acc: %.4f\n", best_acc)
        @printf(io, "Deployment Acc (Static): %.4f\n", m_cold.acc)
        @printf(io, "Deployment Acc (Hebbian): %.4f\n", acc_hebb)
    end
end

run_finance_hebbian_v3()
