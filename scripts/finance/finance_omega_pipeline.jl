include("../../src/experimental/omega_finance.jl")
include("../../src/finance_loader.jl")

function run_finance_omega_pipeline()
    println("\n" * "█"^80)
    println("   OMNI OMEGA — KAN-Hebbian Quantum Hybrid")
    println("   Memory: Fractional Diff (d=0.4) | Logic: Cheb-KAN Splines")
    println("█"^80)

    println("\n  Phase 0: Memory-Preserving Data Ingestion...")
    loader = load_finance_data(FINANCE_DATA_PATH)
    
    WINDOW_SIZE = 20 # 20 days is optimal for KAN spline resolution
    TARGET_IDX = findfirst(x -> x == "SPY", loader.assets)
    X, y = create_windows(loader, WINDOW_SIZE, TARGET_IDX)
    X_train, y_train, X_test, y_test = split_chronological(X, y, 0.70)
    
    y_train = vec(Int.(y_train))
    y_test = vec(Int.(y_test))

    HIDDEN = 64 # KANs are parameter-efficient; 64 hidden is plenty
    DEGREE = 4  # 4th degree Chebyshev polynomials for smooth market curvature
    net = OmegaNet(size(X, 1), HIDDEN, 2, DEGREE)
    
    println("\n  Architecture: KAN-Encoder($DEGREE deg) -> PlasticBackbone -> Dual-Head")
    println("  " * "-"^70)

    # PHASE 1: Self-Supervised Market Reconstruction
    # This teaches the model the "Grammar" of the market before it learns to bet.
    n = size(X_train, 2)
    recon_epochs = 40
    batch_size = 32
    step = 0
    
    println("  Phase 1: Self-Supervised Latent Optimization (Epochs: $recon_epochs)")
    for epoch in 1:recon_epochs
        lr = 2e-3 * 0.5 * (1 + cos(pi * epoch / recon_epochs))
        reset_trace!(net.plastic)
        net.training = true
        
        err = 0.0
        for start in 1:batch_size:n
            step += 1
            idx = start:min(start + batch_size - 1, n)
            X_b = X_train[:, idx]
            
            X_pred = forward_recon!(net, X_b)
            backward_recon!(net, X_b, X_pred; lr=lr, t=step)
            err += sum((X_pred .- X_b).^2) / length(X_b)
        end
        if epoch % 10 == 0; @printf("    Ep %2d | Recon MSE: %.6f\n", epoch, err / (n/batch_size)); end
    end

    # PHASE 2: Directional Meta-Training
    println("\n  Phase 2: KAN-Hebbian Directional Meta-Training")
    predict_epochs = 100
    step = 0
    for epoch in 1:predict_epochs
        lr = 1e-3 * 0.5 * (1 + cos(pi * epoch / predict_epochs))
        reset_trace!(net.plastic)
        net.training = true
        
        for start in 1:batch_size:n
            step += 1
            idx = start:min(start + batch_size - 1, n)
            X_b = X_train[:, idx]
            y_oh = onehot(y_train[idx])
            
            y_pred = forward_predict!(net, X_b)
            backward_predict!(net, y_oh, y_pred; lr=lr, t=step)
        end
        if epoch % 20 == 0
            net.training = false
            ŷ = forward_predict!(net, X_train)
            acc = mean([argmax(ŷ[:, i])-1 for i in 1:n] .== y_train)
            @printf("    Ep %3d | Training Accuracy: %5.2f%%\n", epoch, acc*100)
        end
    end

    # PHASE 3: Omega Deployment
    println("\n  Phase 3: Omega Deployment Adaptation")
    println("  " * "-"^70)

    reset_trace!(net.plastic)
    net.training = false
    
    n_test = size(X_test, 2)
    correct = 0
    for i in 1:n_test
        Xi = X_test[:, i:i]
        yi = y_test[i]
        
        y_pred = forward_predict!(net, Xi)
        pred = argmax(y_pred[:, 1]) - 1
        if pred == yi; correct += 1; end
        
        # Hebbian spline adaptation
        forward_predict!(net, Xi)
    end
    
    acc_omega = correct / n_test
    @printf("  [OMEGA] Final Deployment Accuracy: %5.2f%%\n", acc_omega * 100)

    println("\n  Results saved to results/omega_finance_results.txt")
    mkpath("results")
    write("results/finance/omega_finance_results.txt", "OMNI Omega Deployment Accuracy: $(acc_omega * 100)%")
end

run_finance_omega_pipeline()
