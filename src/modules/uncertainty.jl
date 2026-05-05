#=
================================================================================
   UNCERTAINTY QUANTIFICATION — Bayesian MC Dropout (Over-Engineered)
  ======================================================================
  Computes deep uncertainty estimates using Monte Carlo Dropout.
  This doesn't just output a prediction; it outputs a full probability 
  distribution for EVERY sample.

  Features:
    • Predictive Mean (expected probability)
    • Epistemic Uncertainty (model ignorance - variance across passes)
    • Aleatoric Uncertainty (data noise - entropy of mean prediction)
    • Total Uncertainty (Epistemic + Aleatoric)
    • Mutual Information (information gain about the model parameters)
    • Confidence Intervals (95% CI bounds)
    • Risk Stratification (HIGH/MEDIUM/LOW confidence flagging)
    • Kullback-Leibler (KL) Divergence from uniform prior

  Usage:
    julia uncertainty.jl
================================================================================
=#

include("../core.jl")

# ─── Configuration ─────────────────────────────────────────────────────────────
const MC_PASSES = 100            # Number of stochastic forward passes
const CONFIDENCE_THRESH = 0.90   # Threshold for HIGH confidence
const REVIEW_THRESH = 0.65       # Threshold below which human review is flagged
const DROPOUT_RATE = 0.3         # Dropout rate for inference

# ─── Data Structures ───────────────────────────────────────────────────────────
struct UncertaintyMetrics
    pred_class::Int
    mean_prob::Float64
    std_dev::Float64           # Epistemic
    entropy::Float64           # Total Uncertainty
    mutual_info::Float64       # Epistemic (information theoretic)
    aleatoric::Float64
    ci_lower::Float64          # 95% CI Lower
    ci_upper::Float64          # 95% CI Upper
    kl_div::Float64            # KL Divergence from uniform
    risk_tier::String          # HIGH, MEDIUM, LOW
    flag_review::Bool
end

# ─── Math Utilities ────────────────────────────────────────────────────────────
function shannon_entropy(p::Float64)
    p = clamp(p, 1e-10, 1.0 - 1e-10)
    - (p * log2(p) + (1-p) * log2(1-p))
end

function kl_divergence_uniform(p::Float64)
    # KL( P || U ) where U is uniform [0.5, 0.5]
    p = clamp(p, 1e-10, 1.0 - 1e-10)
    p * log2(p / 0.5) + (1-p) * log2((1-p) / 0.5)
end

# ─── Core Uncertainty Engine ───────────────────────────────────────────────────
function compute_mc_uncertainty(net::Net, X::Matrix{Float64})
    n_samples = size(X, 2)
    # Store predictions for class 1 across all MC passes
    # Shape: (MC_PASSES, n_samples)
    mc_preds = zeros(MC_PASSES, n_samples)

    println("\n Running $(MC_PASSES) Monte Carlo Dropout passes...")
    
    # 1. Enable training mode to keep dropout active during inference!
    # This is the core trick of MC Dropout as a Bayesian approximation
    set_mode!(net, true) 
    
    # Temporarily override network dropout if needed, but let's use the one it has
    original_drop = net.drop
    net.drop = DROPOUT_RATE

    for p in 1:MC_PASSES
        if p % 10 == 0
            @printf("   Pass %3d/%d...\r", p, MC_PASSES)
        end
        # Forward pass with stochastic dropout
        ŷ = forward!(net, X)
        mc_preds[p, :] = ŷ[2, :] # Probabilities of class 1
    end
    println("   Completed $(MC_PASSES) passes.          ")

    # Restore network state
    net.drop = original_drop
    set_mode!(net, false)

    # 2. Compute rich statistics
    results = Vector{UncertaintyMetrics}(undef, n_samples)
    
    println(" Calculating epistemic and aleatoric uncertainties...")

    for i in 1:n_samples
        probs = mc_preds[:, i]
        
        # Predictive Mean
        mean_p = mean(probs)
        pred_class = mean_p >= 0.5 ? 1 : 0
        
        # Epistemic Uncertainty (Variance of the predictions)
        # How much the model disagrees with itself
        std_dev = std(probs)
        
        # Aleatoric Uncertainty (Expected entropy of individual predictions)
        # Inherent noise in the data
        expected_entropy = mean(shannon_entropy.(probs))
        
        # Total Uncertainty (Entropy of the mean prediction)
        total_entropy = shannon_entropy(mean_p)
        
        # Mutual Information (Total - Expected)
        # Represents epistemic uncertainty in an information-theoretic way
        mutual_info = total_entropy - expected_entropy
        
        # 95% Confidence Intervals (using empirical quantiles)
        sorted_probs = sort(probs)
        ci_lower = sorted_probs[max(1, round(Int, 0.025 * MC_PASSES))]
        ci_upper = sorted_probs[min(MC_PASSES, round(Int, 0.975 * MC_PASSES))]
        
        # KL Divergence from prior (how much information we gained)
        kl_div = kl_divergence_uniform(mean_p)

        # Risk Stratification
        # Confidence is distance from decision boundary (0.5)
        confidence = pred_class == 1 ? mean_p : (1.0 - mean_p)
        
        risk_tier = "UNKNOWN"
        flag_review = false
        
        # We flag for review if confidence is low OR if epistemic uncertainty (std_dev) is unusually high
        if confidence >= CONFIDENCE_THRESH && std_dev < 0.1
            risk_tier = "HIGH CONFIDENCE"
        elseif confidence < REVIEW_THRESH || std_dev > 0.2
            risk_tier = "LOW CONFIDENCE"
            flag_review = true
        else
            risk_tier = "MEDIUM CONF"
        end

        results[i] = UncertaintyMetrics(
            pred_class, mean_p, std_dev, total_entropy, mutual_info, 
            expected_entropy, ci_lower, ci_upper, kl_div, risk_tier, flag_review
        )
    end

    return results, mc_preds
end

# ─── Pretty Printing & Analysis ────────────────────────────────────────────────
function print_uncertainty_report(results::Vector{UncertaintyMetrics}, y_true::Vector{Int})
    n_samples = length(results)
    
    # Aggregate Stats
    n_flagged = sum(r.flag_review for r in results)
    avg_std = mean(r.std_dev for r in results)
    avg_mi = mean(r.mutual_info for r in results)
    
    # Accuracy stratified by confidence
    high_conf = filter(r -> r.risk_tier == "HIGH CONFIDENCE", results)
    low_conf = filter(r -> r.risk_tier == "LOW CONFIDENCE", results)
    
    # Calculate accuracy of high vs low confidence (need indices to check against y_true)
    high_conf_idx = findall(r -> r.risk_tier == "HIGH CONFIDENCE", results)
    low_conf_idx = findall(r -> r.risk_tier == "LOW CONFIDENCE", results)
    
    acc_high = isempty(high_conf_idx) ? 0.0 : mean(results[i].pred_class == y_true[i] for i in high_conf_idx)
    acc_low = isempty(low_conf_idx) ? 0.0 : mean(results[i].pred_class == y_true[i] for i in low_conf_idx)
    acc_all = mean(results[i].pred_class == y_true[i] for i in 1:n_samples)

    println("\n" * "╔" * "═"^78 * "╗")
    println("║" * " "^18 * "  BAYESIAN UNCERTAINTY ANALYSIS" * " "^29 * "║")
    println("╚" * "═"^78 * "╝")
    
    println("\n   GLOBAL STATISTICS")
    println("  " * "─"^70)
    @printf("  Total Samples Evaluated: %d\n", n_samples)
    @printf("  Mean Epistemic (StdDev): %.4f\n", avg_std)
    @printf("  Mean Mutual Information: %.4f bits\n", avg_mi)
    @printf("  Flagged for Human Review:%4d (%.1f%% of test set)\n", n_flagged, (n_flagged/n_samples)*100)
    
    println("\n   STRATIFIED PERFORMANCE (Does confidence correlate with accuracy?)")
    println("  " * "─"^70)
    @printf("  Overall Accuracy:        %.2f%%\n", acc_all * 100)
    @printf("  High Confidence Acc:     %.2f%%  (n=%d)\n", acc_high * 100, length(high_conf_idx))
    @printf("  Low Confidence Acc:      %.2f%%  (n=%d)\n", acc_low * 100, length(low_conf_idx))
    
    println("\n   TOP 10 MOST UNCERTAIN PREDICTIONS (Require Human Review)")
    println("  " * "─"^70)
    println("  Idx │ True │ Pred │ Mean Prob │ 95% Confidence Interval │ Mutual Info")
    println("  ────┼──────┼──────┼───────────┼─────────────────────────┼────────────")
    
    # Sort by Mutual Information (highest epistemic uncertainty first)
    sorted_idx = sortperm(results, by=r->r.mutual_info, rev=true)
    
    for (rank, idx) in enumerate(sorted_idx[1:min(10, n_samples)])
        r = results[idx]
        true_label = y_true[idx]
        
        # Color coding marker
        marker = r.pred_class == true_label ? " " : ""
        
        @printf("  %3d │  %d   │  %d %s│   %.3f   │ [ %.3f  ...  %.3f ] │   %.4f\n",
                idx, true_label, r.pred_class, marker, r.mean_prob, r.ci_lower, r.ci_upper, r.mutual_info)
    end
    
    println("\n   TOP 5 MOST CONFIDENT PREDICTIONS")
    println("  " * "─"^70)
    println("  Idx │ True │ Pred │ Mean Prob │ 95% Confidence Interval │ Mutual Info")
    println("  ────┼──────┼──────┼───────────┼─────────────────────────┼────────────")
    
    # Sort by lowest mutual info (highest certainty)
    confident_idx = sortperm(results, by=r->r.mutual_info, rev=false)
    for (rank, idx) in enumerate(confident_idx[1:5])
        r = results[idx]
        true_label = y_true[idx]
        marker = r.pred_class == true_label ? " " : ""
        @printf("  %3d │  %d   │  %d %s│   %.3f   │ [ %.3f  ...  %.3f ] │   %.4f\n",
                idx, true_label, r.pred_class, marker, r.mean_prob, r.ci_lower, r.ci_upper, r.mutual_info)
    end

    # Save detailed report
    mkpath(joinpath(CORE_PROJECT_DIR, "results"))
    open(joinpath(CORE_PROJECT_DIR, "results", "uncertainty_report.csv"), "w") do io
        println(io, "sample_idx,true_label,pred_class,mean_prob,std_dev,ci_lower,ci_upper,total_entropy,aleatoric,epistemic_mi,kl_divergence,risk_tier")
        for i in 1:n_samples
            r = results[i]
            @printf(io, "%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s\n",
                    i, y_true[i], r.pred_class, r.mean_prob, r.std_dev, 
                    r.ci_lower, r.ci_upper, r.entropy, r.aleatoric, 
                    r.mutual_info, r.kl_div, r.risk_tier)
        end
    end
    println("\n   Detailed Bayesian metrics saved to: results/uncertainty_report.csv")
end

# ─── Main Execution ────────────────────────────────────────────────────────────
function run_uncertainty_analysis()
    # 1. Load data
    X_train, y_train, X_test, y_test = load_data(; top_k=300, verbose=false)
    nf = size(X_train, 1)

    # 2. Build and train a solid model to analyze
    # (Using the best architecture found by neuroevolution for speed)
    println("\n Training base model (300 → 463 → 2) for Uncertainty Analysis...")
    net = Net([nf, 463, 2]; drop=DROPOUT_RATE)
    
    # Quick train
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
    
    # 3. Run MC Dropout Uncertainty
    results, mc_preds = compute_mc_uncertainty(net, X_test)
    
    # 4. Print Report
    print_uncertainty_report(results, y_test)
end

run_uncertainty_analysis()
