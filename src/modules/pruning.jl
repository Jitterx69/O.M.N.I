#=
================================================================================
   NEURAL PRUNING — Automated Network Compression
  ====================================================================
  Reduces model size and inference time by removing "dead" or 
  unimportant neurons while maintaining accuracy.

  Features:
    • L1-Norm Activation Magnitude Pruning (data-driven importance)
    • Iterative pruning schedule (prune 10-15% → fine-tune → repeat)
    • Automatic stopping criterion when accuracy drops too much
    • Layer restructuring (physically shrinks the weight matrices)

  Usage:
    julia pruning.jl
================================================================================
=#

include("../core.jl")

# ─── Pruning Configuration ─────────────────────────────────────────────────────
const PRUNE_PERCENT_PER_ROUND = 0.15   # Percentage of neurons to prune per round
const MAX_ACC_DROP = 0.02              # Stop if accuracy drops by more than 2%
const FINE_TUNE_EPOCHS = 10            # Epochs to train after each prune step
const MAX_ROUNDS = 10                  # Maximum pruning rounds

# ─── Pruning Core Mechanics ────────────────────────────────────────────────────
function compute_neuron_importance(net::Net, X_train::Matrix{Float64})
    # Run a forward pass over a large batch to compute average activation magnitudes
    n_samples = min(size(X_train, 2), 1000)
    X_batch = X_train[:, 1:n_samples]
    
    set_mode!(net, false)
    forward!(net, X_batch)
    
    importances = Vector{Vector{Float64}}()
    
    # Exclude output layer (we can't prune the final classes)
    # We prune the hidden layers: layer[1] to layer[end-1]
    for i in 1:length(net.layers)-1
        # pre_act[i] is the output of layer i (before activation)
        # We look at the magnitude of LeakyReLU activations
        acts = lrelu.(net.pre_act[i])
        
        # L1 norm across the batch dimension (mean absolute activation)
        imp = vec(mean(abs.(acts), dims=2))
        push!(importances, imp)
    end
    
    return importances
end

function get_pruning_threshold(importances::Vector{Vector{Float64}}, target_sparsity::Float64)
    # Flatten all importances to find a global threshold
    # (Global pruning allows the network to completely remove useless layers if needed)
    all_imp = Float64[]
    for imp in importances
        append!(all_imp, imp)
    end
    
    sort!(all_imp)
    idx = max(1, round(Int, length(all_imp) * target_sparsity))
    return all_imp[idx]
end

function physically_prune_network!(net::Net, importances::Vector{Vector{Float64}}, threshold::Float64)
    # We create new smaller matrices and copy the unpruned weights over
    n_layers = length(net.layers)
    
    # We prune hidden layers (1 to n_layers-1)
    for i in 1:n_layers-1
        # 1. Identify which neurons survive in layer i
        survivors_i = findall(x -> x > threshold, importances[i])
        
        # Safety check: don't let a layer become empty!
        if isempty(survivors_i)
            # Keep the single most important neuron if all fall below threshold
            survivors_i = [argmax(importances[i])]
        end
        
        n_survivors = length(survivors_i)
        n_original = length(importances[i])
        
        # If no pruning happens in this layer, skip structural changes
        if n_survivors == n_original
            continue
        end
        
        # 2. Shrink Layer i (Output dimension shrinks)
        # W_i shape: (out, in). We keep rows in survivors_i
        old_W_i = net.layers[i].W
        old_b_i = net.layers[i].b
        
        net.layers[i].W = old_W_i[survivors_i, :]
        net.layers[i].b = old_b_i[survivors_i]
        
        # Re-initialize Adam momentum buffers for layer i
        in_dim = size(net.layers[i].W, 2)
        net.layers[i].dW = zeros(n_survivors, in_dim)
        net.layers[i].db = zeros(n_survivors)
        net.layers[i].mW = zeros(n_survivors, in_dim)
        net.layers[i].vW = zeros(n_survivors, in_dim)
        net.layers[i].mb = zeros(n_survivors)
        net.layers[i].vb = zeros(n_survivors)
        
        # 3. Shrink BatchNorm i
        old_bn = net.bns[i]
        net.bns[i].γ = old_bn.γ[survivors_i]
        net.bns[i].β = old_bn.β[survivors_i]
        net.bns[i].rμ = old_bn.rμ[survivors_i]
        net.bns[i].rσ² = old_bn.rσ²[survivors_i]
        
        # Reset BN buffers
        net.bns[i].dγ = zeros(n_survivors)
        net.bns[i].dβ = zeros(n_survivors)
        net.bns[i].mγ = zeros(n_survivors)
        net.bns[i].vγ = zeros(n_survivors)
        net.bns[i].mβ = zeros(n_survivors)
        net.bns[i].vβ = zeros(n_survivors)
        net.bns[i].si = zeros(n_survivors)
        
        # 4. Shrink Layer i+1 (Input dimension shrinks)
        # W_{i+1} shape: (out, in). We keep columns in survivors_i
        old_W_next = net.layers[i+1].W
        net.layers[i+1].W = old_W_next[:, survivors_i]
        
        # Re-initialize Adam momentum buffers for layer i+1
        next_out = size(net.layers[i+1].W, 1)
        net.layers[i+1].dW = zeros(next_out, n_survivors)
        net.layers[i+1].mW = zeros(next_out, n_survivors)
        net.layers[i+1].vW = zeros(next_out, n_survivors)
        # b_{i+1} is unaffected
        
        # Update the stored sizes
        net.sizes[i+1] = n_survivors
    end
end

# ─── Iterative Pruning Pipeline ────────────────────────────────────────────────
function run_pruning_pipeline()
    println("\n" * "╔" * "═"^78 * "╗")
    println("║" * " "^18 * "  NEURAL PRUNING (Model Compression)" * " "^24 * "║")
    println("╚" * "═"^78 * "╝")
    
    # 1. Load data
    X_train, y_train, X_test, y_test = load_data(; top_k=300, verbose=false)
    nf = size(X_train, 1)

    # 2. Train initial unpruned model
    println("\n Training initial dense model (300 → 463 → 2)...")
    net = Net([nf, 463, 2]; drop=0.3)
    
    # Train it well so we have good weights to prune
    n = size(X_train, 2)
    step = 0
    for epoch in 1:60
        lr = 4e-3 * 0.5 * (1 + cos(π * epoch / 60))
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
    
    orig_eval = evaluate(net, X_test, y_test)
    orig_acc = orig_eval.acc
    orig_params = count_params(net)
    orig_sizes = copy(net.sizes)
    
    @printf("   Initial Acc: %.2f%% │ Params: %d │ Arch: %s\n", 
            orig_acc*100, orig_params, join(orig_sizes, " → "))

    # 3. Iterative Pruning Loop
    println("\n Starting Iterative Pruning Schedule...")
    println("   Target: Remove $(round(Int, PRUNE_PERCENT_PER_ROUND*100))% neurons per round")
    println("   Stop if accuracy drops below $(round(orig_acc*100 - MAX_ACC_DROP*100, digits=2))%")
    println("  " * "─"^70)
    
    current_acc = orig_acc
    best_net = deepcopy(net)
    
    for round_idx in 1:MAX_ROUNDS
        # Compute importance
        importances = compute_neuron_importance(net, X_train)
        
        # Find threshold for global pruning
        thresh = get_pruning_threshold(importances, PRUNE_PERCENT_PER_ROUND)
        
        # Physically shrink the matrices
        physically_prune_network!(net, importances, thresh)
        
        pre_ft_params = count_params(net)
        
        # Fine-tune the damaged network
        step = 0
        for epoch in 1:FINE_TUNE_EPOCHS
            lr = 1e-4  # low learning rate for fine-tuning
            set_mode!(net, true)
            perm = randperm(n)
            for start in 1:32:n
                step += 1
                idx = perm[start:min(start+31, n)]
                ŷ = forward!(net, X_train[:, idx])
                backward!(net, onehot(y_train[idx]), ŷ; l2=1e-4, clip=1.0)
                adam_step!(net, lr, step)
            end
        end
        
        # Evaluate
        eval_m = evaluate(net, X_test, y_test)
        
        comp_ratio = (1.0 - pre_ft_params / orig_params) * 100
        
        @printf("  Round %2d │ Pruned %4.1f%% Params │ Acc: %.2f%% │ Arch: %s\n",
                round_idx, comp_ratio, eval_m.acc*100, join(net.sizes, "→"))
                
        # Stopping condition
        if eval_m.acc < orig_acc - MAX_ACC_DROP
            println("\n   Accuracy dropped too much (%.2f%%). Stopping pruning." % (eval_m.acc*100))
            break
        end
        
        # Save this state as the best valid pruned model
        best_net = deepcopy(net)
        current_acc = eval_m.acc
    end
    
    # 4. Final Report
    final_params = count_params(best_net)
    total_reduction = (1.0 - final_params / orig_params) * 100
    
    println("\n" * "─"^78)
    println("   PRUNING COMPLETE")
    println("─"^78)
    
    println("  Before Pruning:")
    @printf("    Architecture: %s\n", join(orig_sizes, " → "))
    @printf("    Parameters:   %d\n", orig_params)
    @printf("    Accuracy:     %.2f%%\n", orig_acc * 100)
    
    println("\n  After Pruning:")
    @printf("    Architecture: %s\n", join(best_net.sizes, " → "))
    @printf("    Parameters:   %d  (▼ %.1f%%)\n", final_params, total_reduction)
    @printf("    Accuracy:     %.2f%% (Δ %.2f%%)\n", current_acc * 100, (current_acc - orig_acc)*100)
    
    # Save results
    mkpath(joinpath(CORE_PROJECT_DIR, "results"))
    open(joinpath(CORE_PROJECT_DIR, "results", "pruning_results.txt"), "w") do io
        println(io, "Neural Pruning Results — $(now())")
        println(io, "="^60)
        println(io, "Original Architecture: $(join(orig_sizes, " → "))")
        println(io, "Pruned Architecture:   $(join(best_net.sizes, " → "))")
        @printf(io, "Original Params: %d\n", orig_params)
        @printf(io, "Pruned Params:   %d\n", final_params)
        @printf(io, "Compression:     %.2f%%\n", total_reduction)
        @printf(io, "Original Acc:    %.4f\n", orig_acc)
        @printf(io, "Pruned Acc:      %.4f\n", current_acc)
    end
    
    return best_net
end

run_pruning_pipeline()
