include("../../src/experimental/robust_hebbian.jl")
include("../../src/finance_loader.jl")

function run_finance_robust_experiment()
    println("\n" * "═"^80)
    println("   OMNI ROBUSTNESS UPGRADE — Bayesian Meta-Learning")
    println("   Uncertainty-Gated Hebbian Traces | Multi-Head Asset Attention")
    println("═"^80)

    loader = load_finance_data(FINANCE_DATA_PATH)
    WINDOW_SIZE = 30
    TARGET_IDX = findfirst(x -> x == "SPY", loader.assets)
    X, y = create_windows(loader, WINDOW_SIZE, TARGET_IDX)
    X_train, y_train, X_test, y_test = split_chronological(X, y, 0.70)
    
    y_train = vec(Int.(y_train))
    y_test = vec(Int.(y_test))

    # Architecture: 4 assets, 3 features each (ret, vol, mom), 30 days
    # Features per asset = 3 * 30 = 90
    net = RobustPlasticNet(4, 3, 30, 128)
    
    println("\n  Meta-Training Phase (Breaking the 100% Trap)")
    println("    Target Training Accuracy: ~85-90% (to ensure generalization)")
    println("    Dropout Rate: 0.4")
    println("  " * "-"^70)

    n = size(X_train, 2)
    epochs = 100
    batch_size = 32
    step = 0
    t0 = now()

    for epoch in 1:epochs
        lr = 5e-4 * 0.5 * (1 + cos(pi * epoch / epochs))
        
        reset_trace!(net.plastic)
        net.training = true
        
        for start in 1:batch_size:n
            step += 1
            idx = start:min(start + batch_size - 1, n)
            X_b = X_train[:, idx]
            y_oh = onehot(y_train[idx])
            
            # Use the robust forward pass
            y_pred = forward_robust!(net, X_b)
            
            # Manual Backprop (Head -> Plastic -> Encoder)
            # (Note: Attention weights are treated as stable structural priors in this version)
            d = y_pred .- y_oh
            d = bwd!(net.head, d, 1e-4, 1.0)
            
            pre_act_p = net.plastic.W * net.plastic.inp .+ net.plastic.b .+ 
                        net.plastic.alpha .* net.plastic.H * net.plastic.inp
            d = d .* lrelu_d.(pre_act_p)
            d = bwd!(net.plastic, d, 1e-4, 1.0)
            
            pre_act_e = net.encoder.W * net.encoder.inp .+ net.encoder.b
            d = d .* lrelu_d.(pre_act_e)
            bwd!(net.encoder, d, 1e-4, 1.0)
            
            # Adam Update
            # (Simplified: reuse existing logic for encoder/plastic/head)
            adam_step!(PlasticNet(net.encoder, net.plastic, net.head, true), lr, step)
        end
        
        if epoch % 20 == 0 || epoch == epochs
            net.training = false
            ŷ = forward_robust!(net, X_train)
            acc = mean([argmax(ŷ[:, i])-1 for i in 1:n] .== y_train)
            @printf("    Ep %3d | TrAcc %5.2f%% | Mode: Regularizing\n", epoch, acc*100)
        end
    end

    println("\n  Phase 2: Bayesian Deployment (Uncertainty-Gated)")
    println("    Inference includes 10 MC-Dropout passes to estimate risk.")
    println("    Traces adapt 10x faster when market uncertainty is high.")
    println("  " * "-"^70)

    # Deployment Test
    reset_trace!(net.plastic)
    n_test = size(X_test, 2)
    correct = 0
    
    for i in 1:n_test
        Xi = X_test[:, i:i]
        yi = y_test[i]
        
        # 1. Estimate Uncertainty (MC Dropout)
        net.training = true # Enable dropout for MC passes
        probs = zeros(10)
        for p in 1:10
            ŷ_mc = forward_robust!(net, Xi)
            probs[p] = ŷ_mc[2, 1]
        end
        net.training = false
        
        # Variance as proxy for Epistemic Uncertainty
        uncertainty = std(probs)
        u_scale = 1.0 + (uncertainty * 10.0) # 1x to 11x boost
        
        # 2. Adapted Inference
        y_final = forward_adapted!(net, Xi, u_scale)
        pred = argmax(y_final[:, 1]) - 1
        if pred == yi; correct += 1; end
    end
    
    acc_robust = correct / n_test
    @printf("  [ROBUST] Bayesian Hebbian Accuracy: %5.2f%%\n", acc_robust * 100)

    elapsed = Dates.value(now() - t0) / 1000.0
    @printf("\n  Robust Pipeline Time: %.1fs\n", elapsed)
    
    mkpath("results")
    open("results/finance/finance_robust_results.txt", "w") do io
        println(io, "OMNI Finance Robustness Upgrade Report")
        println(io, "="^60)
        @printf(io, "Deployment Accuracy: %.4f\n", acc_robust)
        @printf(io, "Training Strategy: Entropy-Balanced Meta-Learning\n")
    end
end

run_finance_robust_experiment()
