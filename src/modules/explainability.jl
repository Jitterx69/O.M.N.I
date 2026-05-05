#=
================================================================================
   EXPLAINABILITY ENGINE (XAI) — SHAP-Inspired Feature Importance
  ===================================================================
  Unboxes the black box by explaining WHICH features drive predictions.

  Features:
    • Permutation Feature Importance (measuring predictive power drop)
    • Gradient-Based Saliency (|∂y/∂x| sensitivity analysis)
    • Combined Score (Permutation + Gradient)
    • Output formatting in terminal-based visualizations

  Usage:
    julia explainability.jl
================================================================================
=#

include("../core.jl")

# ─── Explainability Engine ─────────────────────────────────────────────────────

function compute_permutation_importance(net::Net, X_test::Matrix{Float64}, y_test::Vector{Int})
    # Baseline accuracy
    base_m = evaluate(net, X_test, y_test)
    base_acc = base_m.acc
    
    n_features = size(X_test, 1)
    n_samples = size(X_test, 2)
    importances = zeros(n_features)
    
    println("   Running permutation importance ($n_features features)...")
    
    for i in 1:n_features
        if i % 50 == 0
            @printf("   Processing feature %d/%d...\r", i, n_features)
        end
        
        # Create a copy and shuffle ONLY feature i
        X_shuffled = copy(X_test)
        X_shuffled[i, :] = X_test[i, randperm(n_samples)]
        
        # Evaluate drop in performance
        m = evaluate(net, X_shuffled, y_test)
        
        # Importance = Base Accuracy - Shuffled Accuracy
        # (Higher means the feature is more important)
        importances[i] = base_acc - m.acc
    end
    println("   Completed permutation analysis.        ")
    
    # Normalize to [0, 1] relative to the maximum drop
    max_drop = maximum(importances)
    if max_drop > 0
        importances = importances ./ max_drop
    else
        importances .= 0.0
    end
    
    return importances
end

function compute_gradient_saliency(net::Net, X_test::Matrix{Float64})
    # How sensitive is the output class to small changes in each input feature?
    n_features, n_samples = size(X_test)
    importances = zeros(n_features)
    
    println("   Running gradient-based saliency analysis...")
    
    set_mode!(net, false)
    
    # We'll do this in batches
    batch_size = 32
    step = 0
    
    for start in 1:batch_size:n_samples
        step += 1
        idx = start:min(start+batch_size-1, n_samples)
        X_batch = X_test[:, idx]
        m = length(idx)
        
        # Forward pass
        ŷ = forward!(net, X_batch)
        
        # We want the gradient of the predicted class with respect to the input
        # So we backpropagate a vector of 1s for the predicted class
        preds = [argmax(ŷ[:, i]) for i in 1:m]
        d_out = zeros(size(ŷ))
        for (i, p) in enumerate(preds)
            d_out[p, i] = 1.0
        end
        
        # Manual backward pass down to the input layer
        d = d_out
        d = bwd!(net.layers[end], d, 0.0)
        
        for i in (length(net.layers)-1):-1:1
            d = d .* net.masks[i] .* lrelu_d.(net.pre_act[i])
            d = bwd!(net.bns[i], d)
            # When we hit the first layer, the output 'd' is ∂y/∂x !
            d = bwd!(net.layers[i], d, 0.0)
        end
        
        # d now contains ∂y/∂x for the batch (shape: n_features x m)
        # Average the absolute gradient across the batch
        batch_importance = vec(mean(abs.(d), dims=2))
        
        # Accumulate
        importances .+= batch_importance * (m / n_samples)
    end
    
    # Normalize to [0, 1]
    max_grad = maximum(importances)
    if max_grad > 0
        importances = importances ./ max_grad
    else
        importances .= 0.0
    end
    
    return importances
end

function run_explainability()
    println("\n" * "╔" * "═"^78 * "╗")
    println("║" * " "^18 * " EXPLAINABILITY ENGINE (XAI)" * " "^30 * "║")
    println("╚" * "═"^78 * "╝")
    
    # 1. Load data
    # (Notice we kept the top_k=300 from before)
    X_train, y_train, X_test, y_test = load_data(; top_k=300, verbose=false)
    nf = size(X_train, 1)

    # 2. Train model (using a fast converged model)
    println("\n Training base model (300 → 463 → 2) for XAI...")
    net = Net([nf, 463, 2]; drop=0.3)
    
    n = size(X_train, 2)
    step = 0
    for epoch in 1:40
        lr = 4e-3 * 0.5 * (1 + cos(π * epoch / 40))
        set_mode!(net, true)
        perm = randperm(n)
        for start in 1:32:n
            step += 1
            idx = perm[start:min(start+31, n)]
            ŷ = forward!(net, X_train[:, idx])
            backward!(net, onehot(y_train[idx]), ŷ; l2=2e-4, clip=1.0)
            adam_step!(net, lr, step)
        end
    end
    
    # 3. Compute Importances
    println("\n Analyzing Feature Importance...")
    perm_imp = compute_permutation_importance(net, X_test, y_test)
    grad_imp = compute_gradient_saliency(net, X_test)
    
    # Combined score (weighted average: 60% permutation, 40% gradient)
    combined_imp = 0.6 .* perm_imp .+ 0.4 .* grad_imp
    
    # Normalize combined
    combined_imp = combined_imp ./ maximum(combined_imp)
    
    # 4. Display Results
    # We don't have the original indices easily accessible here, 
    # but we can rank the 300 features that made it through selection.
    
    sorted_idx = sortperm(combined_imp, rev=true)
    
    println("\n   TOP 20 MOST IMPORTANT FEATURES (Combined Score)")
    println("  " * "─"^70)
    println("  Rank │ Feature │ Score │ Bar Chart")
    println("  ─────┼─────────┼───────┼────────────────────────────────────────")
    
    for rank in 1:20
        idx = sorted_idx[rank]
        score = combined_imp[idx]
        
        # Calculate bar length (max 40 chars)
        bar_len = round(Int, score * 40)
        bar = "█"^bar_len * "░"^(40 - bar_len)
        
        @printf("   %2d  │ F_%03d   │ %.3f │ %s\n", rank, idx, score, bar)
    end
    
    println("\n   LEAST IMPORTANT FEATURES (Bottom 5)")
    println("  " * "─"^70)
    for rank in (nf-4):nf
        idx = sorted_idx[rank]
        score = combined_imp[idx]
        @printf("  %3d  │ F_%03d   │ %.3f\n", rank, idx, score)
    end
    
    # Save results
    mkpath(joinpath(CORE_PROJECT_DIR, "results"))
    open(joinpath(CORE_PROJECT_DIR, "results", "explainability_report.csv"), "w") do io
        println(io, "feature_idx,combined_score,permutation_score,gradient_score")
        for idx in sorted_idx
            @printf(io, "%d,%.6f,%.6f,%.6f\n", 
                    idx, combined_imp[idx], perm_imp[idx], grad_imp[idx])
        end
    end
    println("\n   Full feature importance matrix saved to: results/explainability_report.csv")
end

run_explainability()
