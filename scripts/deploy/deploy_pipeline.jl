using Printf, Dates, Statistics

include("../../src/core.jl")
include("../../src/experimental/kan.jl")
include("../../src/experimental/hebbian.jl")
include("../../src/deploy/serialize.jl")

function run_deploy_pipeline()
    println("\n" * "█"^80)
    println("   O.M.N.I. PRODUCTION DEPLOYMENT ENGINE")
    println("   Binary Serialization | Latency Benchmarking | Inference Validation")
    println("█"^80)

    X_tr, y_tr, X_te, y_te = load_data(; top_k=300, verbose=false)
    in_dim = size(X_tr, 1)

    println("\n  Step 1: Training Champion Model (MLP Baseline)")
    println("  " * "-"^70)

    sizes = [in_dim, 256, 128, 2]
    net = Net(sizes; drop=0.2)
    n = size(X_tr, 2)
    step = 0
    best_acc = 0.0

    for ep in 1:100
        lr = 2e-3 * 0.5 * (1 + cos(pi * ep / 100))
        set_mode!(net, true)
        perm = randperm(n)
        for s in 1:32:n
            step += 1
            idx = perm[s:min(s+31, n)]
            yp = forward!(net, X_tr[:, idx])
            backward!(net, onehot(y_tr[idx]), yp; l2=1e-4, clip=1.0)
            adam_step!(net, lr, step)
        end
        m = evaluate(net, X_te, y_te)
        if m.acc > best_acc; best_acc = m.acc; end
        if ep % 25 == 0
            @printf("    Ep %3d | Acc %5.2f%% | F1 %.4f\n", ep, m.acc*100, m.f1)
        end
    end

    final_m = evaluate(net, X_te, y_te)
    @printf("\n  In-Memory Accuracy: %.2f%%\n", final_m.acc*100)

    println("\n  Step 2: Binary Serialization (.omni format)")
    println("  " * "-"^70)

    mkpath("models")
    model_path = "models/champion.omni"
    serialize_net(net, model_path; model_type=LAYER_DENSE)

    println("\n  Step 3: Deserialization Integrity Check")
    println("  " * "-"^70)

    loaded_net, species = deserialize_net(model_path)
    @printf("  Loaded species: %s\n", uppercase(string(species)))

    loaded_m = evaluate(loaded_net, X_te, y_te)
    @printf("  Deserialized Accuracy: %.2f%%\n", loaded_m.acc*100)

    drift = abs(final_m.acc - loaded_m.acc)
    if drift < 1e-10
        println("  INTEGRITY CHECK: PASSED (zero drift)")
    else
        @printf("  INTEGRITY CHECK: WARNING (drift = %.6f)\n", drift)
    end

    println("\n  Step 4: Inference Latency Benchmark")
    println("  " * "-"^70)

    set_mode!(loaded_net, false)
    N_BENCH = 10000
    X_rand = randn(in_dim, N_BENCH)
    latencies = zeros(N_BENCH)

    for i in 1:N_BENCH
        Xi = X_rand[:, i:i]
        t0 = time_ns()
        forward!(loaded_net, Xi)
        latencies[i] = (time_ns() - t0) / 1000.0
    end

    sort!(latencies)
    p50 = latencies[round(Int, N_BENCH * 0.50)]
    p95 = latencies[round(Int, N_BENCH * 0.95)]
    p99 = latencies[round(Int, N_BENCH * 0.99)]
    mn = mean(latencies)
    throughput = 1e6 / mn

    println()
    println("  ┌─────────────────────────────────────────────────────┐")
    println("  │           PRODUCTION READINESS REPORT               │")
    println("  ├─────────────────────────────────────────────────────┤")
    @printf("  │  Model:        %-35s │\n", join(sizes, " -> "))
    @printf("  │  Parameters:   %-35d │\n", count_params(net))
    @printf("  │  Binary Size:  %-35s │\n", @sprintf("%.1f KB", filesize(model_path)/1024))
    @printf("  │  Accuracy:     %-35s │\n", @sprintf("%.2f%%", loaded_m.acc*100))
    @printf("  │  F1 Score:     %-35s │\n", @sprintf("%.4f", loaded_m.f1))
    println("  ├─────────────────────────────────────────────────────┤")
    println("  │  LATENCY (10,000 inferences)                       │")
    println("  ├─────────────────────────────────────────────────────┤")
    @printf("  │  Mean:         %-31.1f us  │\n", mn)
    @printf("  │  P50:          %-31.1f us  │\n", p50)
    @printf("  │  P95:          %-31.1f us  │\n", p95)
    @printf("  │  P99:          %-31.1f us  │\n", p99)
    @printf("  │  Throughput:   %-27.0f inf/sec  │\n", throughput)
    println("  ├─────────────────────────────────────────────────────┤")

    if p99 < 1000 && drift < 1e-10
        println("  │  STATUS:       PRODUCTION READY                    │")
    elseif p99 < 5000
        println("  │  STATUS:       STAGING READY                       │")
    else
        println("  │  STATUS:       NEEDS OPTIMIZATION                  │")
    end
    println("  └─────────────────────────────────────────────────────┘")

    println("\n  Step 5: Hebbian Model Serialization Test")
    println("  " * "-"^70)

    pnet = PlasticNet(in_dim, 128, 2)
    step = 0
    for ep in 1:50
        lr_t = 2e-3 * 0.5 * (1 + cos(pi * ep / 50))
        reset_traces!(pnet)
        set_mode!(pnet, true)
        perm = randperm(n)
        for s in 1:32:n
            step += 1
            idx = perm[s:min(s+31, n)]
            yp = forward!(pnet, X_tr[:, idx])
            backward!(pnet, onehot(y_tr[idx]), yp; l2=1e-4, clip=1.0)
            adam_step!(pnet, lr_t, step)
        end
    end

    hebb_path = "models/hebbian_champion.omni"
    serialize_net(pnet, hebb_path; model_type=LAYER_PLASTIC)

    loaded_pnet, sp2 = deserialize_net(hebb_path)
    m_orig = evaluate(pnet, X_te, y_te)
    m_load = evaluate(loaded_pnet, X_te, y_te)
    @printf("  Original Hebbian Acc: %.2f%%\n", m_orig.acc*100)
    @printf("  Loaded Hebbian Acc:   %.2f%%\n", m_load.acc*100)

    h_drift = abs(m_orig.acc - m_load.acc)
    if h_drift < 1e-10
        println("  HEBBIAN INTEGRITY: PASSED")
    else
        @printf("  HEBBIAN INTEGRITY: DRIFT = %.6f\n", h_drift)
    end

    mkpath("results")
    open("results/deploy/deployment_report.txt", "w") do io
        println(io, "O.M.N.I. Production Deployment Report -- $(now())")
        println(io, "="^60)
        @printf(io, "MLP Accuracy:     %.4f\n", loaded_m.acc)
        @printf(io, "Hebbian Accuracy: %.4f\n", m_load.acc)
        @printf(io, "Latency P50:      %.1f us\n", p50)
        @printf(io, "Latency P95:      %.1f us\n", p95)
        @printf(io, "Latency P99:      %.1f us\n", p99)
        @printf(io, "Throughput:       %.0f inf/sec\n", throughput)
        @printf(io, "Integrity:        %s\n", drift < 1e-10 ? "PASS" : "FAIL")
    end
    println("\n  Deployment report saved to results/deployment_report.txt")
end

run_deploy_pipeline()
