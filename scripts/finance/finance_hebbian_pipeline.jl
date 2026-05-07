include("../../src/experimental/hebbian.jl")
include("../../src/finance_loader.jl")

function run_finance_hebbian_experiment()
    println("\n" * "═"^80)
    println("   FINANCIAL REGIME ADAPTATION — Differentiable Hebbian Plasticity")
    println("   Chronological Time-Series | Log-Returns | Online Adaptation")
    println("═"^80)

    println("\n  Loading market data history...")
    loader = load_finance_data(FINANCE_DATA_PATH)
    
    WINDOW_SIZE = 10  # 10 days of history
    TARGET_IDX = findfirst(x -> x == "SPY", loader.assets)
    
    println("  Generating sliding windows (Target: SPY Return)...")
    X, y = create_windows(loader, WINDOW_SIZE, TARGET_IDX)
    
    # Chronological Split: 70% Meta-Training, 30% Deployment Deployment
    X_train, y_train, X_test, y_test = split_chronological(X, y, 0.70)
    
    y_train = vec(Int.(y_train))
    y_test = vec(Int.(y_test))

    println("\n  Dataset Summary:")
    println("    Assets:            $(loader.assets)")
    println("    Total Samples:     $(size(X, 2)) days")
    println("    Training Period:   $(size(X_train, 2)) days")
    println("    Deployment Period: $(size(X_test, 2)) days")
    println("    Feature Dim:       $(size(X, 1)) ($WINDOW_SIZE days x $(length(loader.assets)) assets)")

    HIDDEN = 64
    net = PlasticNet(size(X, 1), HIDDEN, 2)
    
    println("\n  Architecture: Encoder -> PlasticDense($HIDDEN -> $HIDDEN) -> Head")
    println("  " * "-"^70)

    # 1. Meta-Training Phase (Backprop + Trace Accumulation)
    n = size(X_train, 2)
    epochs = 50
    batch_size = 32
    step = 0
    
    println("  Phase 1: Meta-Training (Epochs: $epochs)")
    
    t0 = now()
    for epoch in 1:epochs
        lr = 1e-3 * 0.5 * (1 + cos(pi * epoch / epochs))
        
        # Chronological batching to preserve time-series logic
        reset_traces!(net)
        set_mode!(net, true)
        
        for start in 1:batch_size:n
            step += 1
            idx = start:min(start + batch_size - 1, n)
            
            X_b = X_train[:, idx]
            y_oh = onehot(y_train[idx])
            
            y_pred = forward!(net, X_b)
            backward!(net, y_oh, y_pred; l2=1e-4, clip=1.0)
            adam_step!(net, lr, step)
        end
        
        if epoch % 10 == 0 || epoch == epochs
            m = evaluate(net, X_train, y_train)
            @printf("    Ep %2d | Loss %.4f | Acc %5.1f%% | LR %.1e\n", epoch, m.loss, m.acc*100, lr)
        end
    end

    # 2. Deployment Drift Experiment
    println("\n  Phase 2: Deployment Drift Experiment (No Backpropagation)")
    println("  " * "-"^70)

    # TEST A: Cold Deployment (Static Weights)
    reset_traces!(net)
    set_mode!(net, false)
    m_cold = evaluate(net, X_test, y_test)
    @printf("  [COLD] Static Weights Accuracy:   %5.2f%%\n", m_cold.acc * 100)

    # TEST B: Hebbian Deployment (Self-Adapting Weights)
    # We simulate online learning: as each day passes, the network updates its traces
    reset_traces!(net)
    set_mode!(net, false)
    
    n_test = size(X_test, 2)
    correct = 0
    
    for i in 1:n_test
        Xi = X_test[:, i:i]
        yi = y_test[i]
        
        # 1. Predict (Inference)
        y_pred = forward!(net, Xi)
        pred = argmax(y_pred[:, 1]) - 1
        if pred == yi
            correct += 1
        end
        
        # 2. Adapt (Hebbian Trace Update - No Gradient!)
        # We use forward_accumulate! to simulate synaptic update from the signal
        forward_accumulate!(net, Xi)
    end
    
    acc_hebb = correct / n_test
    @printf("  [WARM] Hebbian Adapted Accuracy: %5.2f%%\n", acc_hebb * 100)

    delta = acc_hebb - m_cold.acc
    println()
    if delta > 0
        @printf("  ADAPTATION GAIN: %+.2f%% accuracy\n", delta * 100)
        println("  The Hebbian traces successfully captured the shifting market regime.")
    else
        @printf("  ADAPTATION DELTA: %+.2f%% (neutral or noise)\n", delta * 100)
    end

    elapsed = Dates.value(now() - t0) / 1000.0
    @printf("\n  Total Time: %.1fs\n", elapsed)
    
    # Save results
    mkpath(joinpath(FINANCE_PROJECT_DIR, "results"))
    open(joinpath(FINANCE_PROJECT_DIR, "results", "finance_hebbian_results.txt"), "w") do io
        println(io, "Financial Hebbian Adaptation Experiment -- $(now())")
        println(io, "="^60)
        @printf(io, "Static Accuracy:  %.4f\n", m_cold.acc)
        @printf(io, "Hebbian Accuracy: %.4f\n", acc_hebb)
        @printf(io, "Gain:             %+.4f\n", delta)
    end
    println("  Results saved to results/finance_hebbian_results.txt")
end

run_finance_hebbian_experiment()
