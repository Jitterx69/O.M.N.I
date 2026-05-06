include("../src/higgs_loader.jl")
include("../src/core.jl")
include("../src/experimental/caji_attention.jl")

lrelu(x) = x > 0 ? x : 0.01x
lrelu_d(x) = x > 0 ? 1.0 : 0.01

function softmax_c(X)
    mx = maximum(X, dims=1)
    e = exp.(X .- mx)
    e ./ sum(e, dims=1)
end

function xent(yp, yoh)
    -mean(sum(yoh .* log.(clamp.(yp, 1e-8, 1.0)), dims=1))
end

mutable struct OmegaHiggsNet
    caji::CAJILayer
    layers::Vector{Dense}
    bns::Vector{BN}
    head::Dense
    drop::Float64
    training::Bool
end

function OmegaHiggsNet(raw_dim::Int, caji_out::Int, hidden::Vector{Int}, out_dim::Int; drop=0.3)
    caji = CAJILayer(4, 4, 16)
    feat_dim = raw_dim + caji_out + 18

    layers = Dense[]
    bns = BN[]
    prev = feat_dim
    for h in hidden
        push!(layers, Dense(prev, h))
        push!(bns, BN(h))
        prev = h
    end
    head = Dense(prev, out_dim)
    OmegaHiggsNet(caji, layers, bns, head, drop, true)
end

function set_mode!(net::OmegaHiggsNet, training::Bool)
    net.training = training
    for bn in net.bns; bn.training = training; end
end

function forward!(net::OmegaHiggsNet, X::Matrix{Float64})
    caji_out = fwd_caji!(net.caji, X)
    pair_feats = compute_pairwise_physics(X)
    h = vcat(X, caji_out, pair_feats)

    for (i, l) in enumerate(net.layers)
        z = fwd!(l, h)
        z = fwd!(net.bns[i], z)
        h = lrelu.(z)
        if net.training && net.drop > 0
            mask = (rand(size(h)...) .> net.drop) ./ (1.0 - net.drop)
            h .*= mask
        end
    end
    return softmax_c(fwd!(net.head, h))
end

function backward!(net::OmegaHiggsNet, X::Matrix{Float64}, y_oh, y_pred; l2=1e-5, clip=1.0)
    B = size(y_oh, 2)
    d = y_pred .- y_oh

    d = bwd!(net.head, d, l2, clip)

    for i in length(net.layers):-1:1
        l = net.layers[i]
        pre = l.W * l.inp .+ l.b
        bn_out = fwd!(net.bns[i], pre)
        d = d .* lrelu_d.(bn_out)
        d = bwd!(net.bns[i], d)
        d = bwd!(l, d, l2, clip)
    end

    caji_d = d[size(X,1)+1:size(X,1)+net.caji.n_jets*net.caji.d_k, :]
    bwd_caji!(net.caji, caji_d, l2, clip)
end

function adam_step!(net::OmegaHiggsNet, lr, t)
    bc1 = 1.0 - 0.9^t; bc2 = 1.0 - 0.999^t
    eps = 1e-8

    function upd!(l::Dense)
        @. l.mW = 0.9*l.mW + 0.1*l.dW; @. l.vW = 0.999*l.vW + 0.001*l.dW^2
        @. l.W -= lr*(l.mW/bc1)/(sqrt(l.vW/bc2)+eps)
        @. l.mb = 0.9*l.mb + 0.1*l.db; @. l.vb = 0.999*l.vb + 0.001*l.db^2
        @. l.b -= lr*(l.mb/bc1)/(sqrt(l.vb/bc2)+eps)
    end

    function upd_bn!(bn::BN)
        @. bn.mγ = 0.9*bn.mγ + 0.1*bn.dγ; @. bn.vγ = 0.999*bn.vγ + 0.001*bn.dγ^2
        @. bn.γ -= lr*(bn.mγ/bc1)/(sqrt(bn.vγ/bc2)+eps)
        @. bn.mβ = 0.9*bn.mβ + 0.1*bn.dβ; @. bn.vβ = 0.999*bn.vβ + 0.001*bn.dβ^2
        @. bn.β -= lr*(bn.mβ/bc1)/(sqrt(bn.vβ/bc2)+eps)
    end

    for l in net.layers; upd!(l); end
    for bn in net.bns; upd_bn!(bn); end
    upd!(net.head)
    adam_caji!(net.caji, lr, t)
end

function count_params(net::OmegaHiggsNet)
    c = count_caji_params(net.caji)
    for l in net.layers; c += length(l.W) + length(l.b); end
    for bn in net.bns; c += length(bn.γ) + length(bn.β); end
    c += length(net.head.W) + length(net.head.b)
    return c
end

function snapshot_weights(net::OmegaHiggsNet)
    ws = Vector{Vector{Float64}}()
    for l in net.layers
        push!(ws, copy(vec(l.W)))
        push!(ws, copy(l.b))
    end
    push!(ws, copy(vec(net.head.W)))
    push!(ws, copy(net.head.b))
    return ws
end

function accumulate_swa!(swa_ws, current_ws, n)
    for i in 1:length(swa_ws)
        @. swa_ws[i] = (swa_ws[i] * (n - 1) + current_ws[i]) / n
    end
end

function apply_swa!(net::OmegaHiggsNet, swa_ws)
    idx = 1
    for l in net.layers
        l.W .= reshape(swa_ws[idx], size(l.W)); idx += 1
        l.b .= swa_ws[idx]; idx += 1
    end
    net.head.W .= reshape(swa_ws[idx], size(net.head.W)); idx += 1
    net.head.b .= swa_ws[idx]
end

function run_higgs_omega_v2()
    println("\n" * "█"^80)
    println("   HIGGS OMEGA v2 — RELATIONAL JET ATTENTION + SWA + DEEP RESIDUAL")
    println("   Cross-Attention Jet Interaction | Pairwise Physics | Weight Averaging")
    println("█"^80)

    higgs_path = joinpath(HIGGS_PROJECT_DIR, "data", "higgs", "HIGGS.csv")
    if !isfile(higgs_path)
        println("  ERROR: HIGGS.csv not found at $higgs_path")
        return
    end

    ds = index_higgs(higgs_path)

    n_total = ds.n_samples
    n_test = 500000
    n_train = n_total - n_test

    train_idx = collect(1:n_train)
    test_idx = collect(n_train+1:n_total)

    println("\n  Dataset: $(n_train) train, $(n_test) test")
    @printf("  Signal: %.1f%% | Weights: Sig=%.3f, Bg=%.3f\n",
            ds.signal_count/n_total*100, ds.class_weights[2], ds.class_weights[1])

    raw_dim = HIGGS_FEATURES
    caji_out_dim = 4 * 16
    pair_dim = 18
    total_in = raw_dim + caji_out_dim + pair_dim

    hidden = [1024, 512, 512, 256, 128]
    net = OmegaHiggsNet(raw_dim, caji_out_dim, hidden, 2; drop=0.3)

    println("\n  Total Input Features: $total_in ($raw_dim raw + $caji_out_dim CAJI + $pair_dim pairwise)")
    println("  Hidden Layers: $(join(hidden, " -> "))")
    println("  Parameters: $(count_params(net))")
    println("  Dropout: $(net.drop)")
    println("  " * "-"^70)

    epochs = 15
    batch_size = 2048
    step = 0
    best_acc = 0.0
    best_ep = 0

    swa_start_epoch = 10
    swa_ws = nothing
    swa_count = 0

    t0 = now()

    for ep in 1:epochs
        if ep <= 3
            lr = 2e-3 * (ep / 3.0)
        else
            lr = 2e-3 * 0.5 * (1 + cos(pi * (ep - 3) / (epochs - 3)))
        end

        shuffle!(train_idx)
        set_mode!(net, true)
        epoch_loss = 0.0
        n_batches = 0

        for start in 1:batch_size:n_train
            step += 1
            stop = min(start + batch_size - 1, n_train)
            idx_b = train_idx[start:stop]

            X_b, y_b = load_higgs_batch(ds, idx_b)
            y_oh = higgs_onehot(y_b)

            y_pred = forward!(net, X_b)
            backward!(net, X_b, y_oh, y_pred; l2=1e-5, clip=1.0)
            adam_step!(net, lr, step)

            epoch_loss += xent(y_pred, y_oh)
            n_batches += 1

            if step % 2000 == 0
                @printf("  Batch %7d | Loss: %.4f | LR: %.1e\n",
                        step, epoch_loss / n_batches, lr)
            end
        end

        if ep >= swa_start_epoch
            cws = snapshot_weights(net)
            if swa_ws === nothing
                swa_ws = deepcopy(cws)
                swa_count = 1
            else
                swa_count += 1
                accumulate_swa!(swa_ws, cws, swa_count)
            end
            @printf("  [SWA] Accumulated snapshot %d\n", swa_count)
        end

        n_eval = min(100000, n_test)
        eval_sub = test_idx[randperm(n_test)[1:n_eval]]
        X_te, y_te = load_higgs_batch(ds, eval_sub)

        set_mode!(net, false)
        y_pred_te = forward!(net, X_te)
        m = higgs_metrics(y_pred_te, y_te)

        improved = ""
        if m.acc > best_acc
            best_acc = m.acc
            best_ep = ep
            improved = " *"
        end

        println("\n" * "─"^70)
        @printf("  Epoch %2d | Acc: %5.2f%% | F1: %.4f | AUC: %.4f | Loss: %.4f%s\n",
                ep, m.acc*100, m.f1, m.auc, epoch_loss/n_batches, improved)
        println("─"^70 * "\n")
    end

    if swa_ws !== nothing
        println("  Applying SWA weights ($swa_count snapshots)...")
        apply_swa!(net, swa_ws)

        n_eval_final = min(200000, n_test)
        eval_final = test_idx[randperm(n_test)[1:n_eval_final]]
        X_final, y_final = load_higgs_batch(ds, eval_final)

        set_mode!(net, false)
        y_pred_swa = forward!(net, X_final)
        m_swa = higgs_metrics(y_pred_swa, y_final)

        println("\n" * "═"^70)
        println("  SWA EVALUATION ($(n_eval_final) samples)")
        println("═"^70)
        @printf("  Acc: %5.2f%% | F1: %.4f | AUC: %.4f\n", m_swa.acc*100, m_swa.f1, m_swa.auc)
        @printf("  Prec: %.4f | Rec: %.4f\n", m_swa.prec, m_swa.rec)
        @printf("  TP: %d | FP: %d | FN: %d | TN: %d\n", m_swa.tp, m_swa.fp, m_swa.fn, m_swa.tn)

        if m_swa.acc > best_acc
            best_acc = m_swa.acc
            println("  SWA IMPROVED over best single model!")
        end
    end

    elapsed = Dates.value(now() - t0) / 1000.0

    println("\n" * "█"^80)
    println("  HIGGS OMEGA v2 COMPLETE")
    println("█"^80)
    @printf("  Best Accuracy: %.2f%%\n", best_acc * 100)
    @printf("  Training Time: %.1fs\n", elapsed)

    mc_preds = zeros(2, length(y_final))
    n_mc = 10
    set_mode!(net, true)
    for _ in 1:n_mc
        mc_preds .+= forward!(net, X_final)
    end
    mc_preds ./= n_mc
    mc_m = higgs_metrics(mc_preds, y_final)
    @printf("  MC-Ensemble (10 passes): Acc: %.2f%% | F1: %.4f\n", mc_m.acc*100, mc_m.f1)

    mkpath("results")
    open("results/higgs_omega_v2_results.txt", "w") do io
        println(io, "HIGGS Omega v2 Results -- $(now())")
        println(io, "="^60)
        @printf(io, "Best Single Model Acc: %.4f\n", best_acc)
        @printf(io, "SWA Acc: %.4f\n", m_swa.acc)
        @printf(io, "MC-Ensemble Acc: %.4f\n", mc_m.acc)
        @printf(io, "Parameters: %d\n", count_params(net))
        @printf(io, "Time: %.1fs\n", elapsed)
    end

    println("\n  Results saved to: results/higgs_omega_v2_results.txt")
end

run_higgs_omega_v2()
