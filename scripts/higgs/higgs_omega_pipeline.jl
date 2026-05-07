include("../../src/higgs_loader.jl")
include("../../src/experimental/info_bottleneck.jl")

function run_higgs_omega()
    println("\n" * "█"^80)
    println("   HIGGS OMEGA PIPELINE — Res-VIB Architecture")
    println("   Physics-Informed Features | Lagrangian Beta | Residual Bottleneck")
    println("█"^80)

    path = joinpath(HIGGS_PROJECT_DIR, "data", "higgs", "HIGGS.csv")
    if !isfile(path)
        println("  ERROR: HIGGS.csv not found at $path")
        return
    end

    ds = index_higgs(path)
    
    n_total = ds.n_samples
    n_test = 500000
    n_train = n_total - n_test
    
    train_idx = collect(1:n_train)
    test_idx = collect(n_train+1:n_total)

    nf = HIGGS_FEATURES
    H_DIMS = [512, 512, 256, 128]
    Z_DIM = 64
    
    net = ResIBNet(nf, H_DIMS, Z_DIM, 2)
    println("\n Architecture: ResEncoder($nf -> $(join(H_DIMS, " -> "))) -> IB($Z_DIM) -> Head(2)")
    println("   Total Parameters: $(count_params(net))")
    println("  " * "-"^70)

    epochs = 10
    batch_size = 1024
    step = 0
    
    β_target_kl = 5.0 # nats
    β = 1e-4
    
    best_acc = 0.0
    
    t0 = now()

    for ep in 1:epochs
        lr = 1e-3 * 0.5 * (1 + cos(pi * ep / epochs))
        
        shuffle!(train_idx)
        
        for start in 1:batch_size:n_train
            step += 1
            stop = min(start + batch_size - 1, n_train)
            idx_b = train_idx[start:stop]
            
            X_b, y_b = load_higgs_batch(ds, idx_b)
            y_oh = higgs_onehot(y_b)
            
            y_pred = forward!(net, X_b)
            backward!(net, y_oh, y_pred, β; l2=1e-5, clip=1.0)
            adam_step!(net, lr, step)
            
            # Lagrangian Beta Adjustment
            if step % 100 == 0
                kl = kl_divergence(net.bottleneck)
                if kl > β_target_kl
                    β *= 1.05
                else
                    β *= 0.98
                end
                β = clamp(β, 1e-6, 1.0)
            end

            if step % 2000 == 0
                @printf("  Batch %7d | Loss: %.4f | KL: %.3f | Beta: %.1e | LR: %.1e\n", 
                        step, xent(y_pred, y_oh), kl_divergence(net.bottleneck), β, lr)
            end
        end

        # Test
        n_test_eval = 50000
        test_sub = test_idx[randperm(n_test)[1:n_test_eval]]
        X_te, y_te = load_higgs_batch(ds, test_sub)
        m = evaluate(net, X_te, y_te)
        
        improved = ""
        if m.acc > best_acc
            best_acc = m.acc
            improved = " *"
        end
        
        println("\n" * "-"^70)
        @printf("  Epoch %2d Complete | Acc: %5.2f%% | F1: %.4f | AUC: %.4f%s\n", 
                ep, m.acc*100, m.f1, m.auc, improved)
        println("-"^70 * "\n")
    end

    elapsed = Dates.value(now() - t0) / 1000.0
    
    println("  HIGGS OMEGA COMPLETE | Time: $(elapsed)s | Best Acc: $(best_acc*100)%")
    
    mkpath("results")
    open("results/physics/higgs_omega_results.txt", "w") do io
        println(io, "HIGGS Omega Results -- $(now())")
        @printf(io, "Best Accuracy: %.4f\n", best_acc)
        @printf(io, "Final Beta: %.1e\n", β)
        @printf(io, "Time: %.1fs\n", elapsed)
    end
end

run_higgs_omega()
