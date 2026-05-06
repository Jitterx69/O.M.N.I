include("../core.jl")

const TRACE_DECAY = 0.95
const TRACE_CLAMP = 3.0

mutable struct PlasticDense
    ni::Int
    no::Int

    W::Matrix{Float64}
    b::Vector{Float64}
    alpha::Matrix{Float64}

    H::Matrix{Float64}

    dW::Matrix{Float64}
    db::Vector{Float64}
    dalpha::Matrix{Float64}

    mW::Matrix{Float64}; vW::Matrix{Float64}
    mb::Vector{Float64}; vb::Vector{Float64}
    malpha::Matrix{Float64}; valpha::Matrix{Float64}

    inp::Matrix{Float64}
    out_act::Matrix{Float64}
end

function PlasticDense(ni::Int, no::Int)
    PlasticDense(
        ni, no,
        randn(no, ni) .* sqrt(2.0 / ni),
        zeros(no),
        randn(no, ni) .* 0.01,
        zeros(no, ni),
        zeros(no, ni), zeros(no), zeros(no, ni),
        zeros(no, ni), zeros(no, ni),
        zeros(no), zeros(no),
        zeros(no, ni), zeros(no, ni),
        zeros(0, 0), zeros(0, 0)
    )
end

function effective_weights(l::PlasticDense)
    l.W .+ l.alpha .* l.H
end

function fwd!(l::PlasticDense, X::Matrix{Float64})
    l.inp = X
    W_eff = effective_weights(l)
    W_eff * X .+ l.b
end

function update_trace!(l::PlasticDense, pre::Matrix{Float64}, post::Matrix{Float64})
    B = size(pre, 2)
    hebb = (post * pre') ./ B
    clamp!(hebb, -TRACE_CLAMP, TRACE_CLAMP)
    l.H .= TRACE_DECAY .* l.H .+ (1.0 - TRACE_DECAY) .* hebb
    clamp!(l.H, -TRACE_CLAMP, TRACE_CLAMP)
end

function reset_trace!(l::PlasticDense)
    l.H .= 0.0
end

function bwd!(l::PlasticDense, d::Matrix{Float64}, λ::Float64, clip::Float64=1.0)
    B = size(d, 2)

    dW_eff = d * l.inp' ./ B

    l.dW = dW_eff .+ λ .* l.W
    l.dalpha = dW_eff .* l.H
    l.db = vec(sum(d, dims=2)) ./ B

    gnorm = sqrt(sum(l.dW .^ 2) + sum(l.db .^ 2) + sum(l.dalpha .^ 2))
    if gnorm > clip
        s = clip / gnorm
        l.dW .*= s
        l.db .*= s
        l.dalpha .*= s
    end

    W_eff = effective_weights(l)
    W_eff' * d
end

function count_plastic_params(l::PlasticDense)
    length(l.W) + length(l.b) + length(l.alpha)
end

mutable struct PlasticNet
    encoder::Dense
    plastic::PlasticDense
    head::Dense
    training::Bool
end

function PlasticNet(in_dim::Int, hidden::Int, out_dim::Int)
    enc = Dense(in_dim, hidden)
    pla = PlasticDense(hidden, hidden)
    hd = Dense(hidden, out_dim)
    PlasticNet(enc, pla, hd, true)
end

function set_mode!(n::PlasticNet, training::Bool)
    n.training = training
end

function reset_traces!(n::PlasticNet)
    reset_trace!(n.plastic)
end

function forward!(net::PlasticNet, X::Matrix{Float64})
    h1 = lrelu.(fwd!(net.encoder, X))

    z2 = fwd!(net.plastic, h1)
    h2 = lrelu.(z2)

    if net.training
        update_trace!(net.plastic, h1, h2)
    end

    net.plastic.out_act = h2

    softmax_c(fwd!(net.head, h2))
end

function forward_accumulate!(net::PlasticNet, X::Matrix{Float64})
    h1 = lrelu.(fwd!(net.encoder, X))
    z2 = fwd!(net.plastic, h1)
    h2 = lrelu.(z2)
    update_trace!(net.plastic, h1, h2)
    return nothing
end

function backward!(net::PlasticNet, y_oh, y_pred; l2=1e-4, clip=1.0)
    d = y_pred .- y_oh

    d = bwd!(net.head, d, l2, clip)

    pre_act = net.plastic.W * net.plastic.inp .+ net.plastic.b .+
              net.plastic.alpha .* net.plastic.H * net.plastic.inp
    d = d .* lrelu_d.(pre_act)

    d = bwd!(net.plastic, d, l2, clip)

    d = d .* lrelu_d.(net.encoder.W * net.encoder.inp .+ net.encoder.b)
    d = bwd!(net.encoder, d, l2, clip)
end

function adam_step!(net::PlasticNet, lr, t; β1=0.9, β2=0.999, ε=1e-8)
    bc1 = 1.0 - β1^t
    bc2 = 1.0 - β2^t

    function update_dense!(l::Dense)
        @. l.mW = β1 * l.mW + (1-β1) * l.dW
        @. l.vW = β2 * l.vW + (1-β2) * l.dW^2
        @. l.mb = β1 * l.mb + (1-β1) * l.db
        @. l.vb = β2 * l.vb + (1-β2) * l.db^2
        @. l.W -= lr * (l.mW/bc1) / (sqrt(l.vW/bc2) + ε)
        @. l.b -= lr * (l.mb/bc1) / (sqrt(l.vb/bc2) + ε)
    end

    update_dense!(net.encoder)
    update_dense!(net.head)

    p = net.plastic
    @. p.mW = β1 * p.mW + (1-β1) * p.dW
    @. p.vW = β2 * p.vW + (1-β2) * p.dW^2
    @. p.W -= lr * (p.mW/bc1) / (sqrt(p.vW/bc2) + ε)

    @. p.mb = β1 * p.mb + (1-β1) * p.db
    @. p.vb = β2 * p.vb + (1-β2) * p.db^2
    @. p.b -= lr * (p.mb/bc1) / (sqrt(p.vb/bc2) + ε)

    @. p.malpha = β1 * p.malpha + (1-β1) * p.dalpha
    @. p.valpha = β2 * p.valpha + (1-β2) * p.dalpha^2
    @. p.alpha -= lr * (p.malpha/bc1) / (sqrt(p.valpha/bc2) + ε)
end

function count_params(net::PlasticNet)
    length(net.encoder.W) + length(net.encoder.b) +
    count_plastic_params(net.plastic) +
    length(net.head.W) + length(net.head.b)
end

function evaluate(net::PlasticNet, X, y)
    set_mode!(net, false)
    ŷ = forward!(net, X)
    preds = [argmax(ŷ[:, i]) - 1 for i in 1:size(ŷ, 2)]
    yoh = onehot(y)
    acc = mean(preds .== y)
    tp = sum((preds .== 1) .& (y .== 1))
    fp = sum((preds .== 1) .& (y .== 0))
    fn = sum((preds .== 0) .& (y .== 1))
    tn = sum((preds .== 0) .& (y .== 0))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    loss = xent(ŷ, yoh)
    auc = compute_auc(vec(ŷ[2, :]), y)
    set_mode!(net, true)
    (acc=acc, prec=prec, rec=rec, f1=f1, loss=loss, auc=auc,
     cm=(tp=tp, fp=fp, fn=fn, tn=tn))
end

function analyze_plasticity(net::PlasticNet)
    p = net.plastic

    W_norm = sqrt(sum(p.W .^ 2))
    alpha_norm = sqrt(sum(p.alpha .^ 2))
    H_norm = sqrt(sum(p.H .^ 2))
    contrib_norm = sqrt(sum((p.alpha .* p.H) .^ 2))

    ratio = contrib_norm / max(W_norm, 1e-8)

    alpha_flat = vec(p.alpha)
    alpha_pos = sum(alpha_flat .> 0.01)
    alpha_neg = sum(alpha_flat .< -0.01)
    alpha_dormant = length(alpha_flat) - alpha_pos - alpha_neg

    println("\n   PLASTICITY ANALYSIS")
    println("  " * "-"^60)
    @printf("  Fixed Weight Norm (||W||):         %.4f\n", W_norm)
    @printf("  Plasticity Coeff Norm (||alpha||): %.4f\n", alpha_norm)
    @printf("  Hebbian Trace Norm (||H||):        %.4f\n", H_norm)
    @printf("  Adaptive Contrib (||alpha*H||):    %.4f\n", contrib_norm)
    @printf("  Adaptation Ratio (contrib/fixed):  %.4f (%.1f%%)\n", ratio, ratio * 100)
    println()
    @printf("  Excitatory Synapses (alpha > 0):   %d\n", alpha_pos)
    @printf("  Inhibitory Synapses (alpha < 0):   %d\n", alpha_neg)
    @printf("  Dormant Synapses:                  %d\n", alpha_dormant)

    H_flat = vec(p.H)
    active_synapses = sum(abs.(H_flat) .> 0.01)
    total_synapses = length(H_flat)

    println()
    @printf("  Active Trace Entries:              %d / %d (%.1f%%)\n",
            active_synapses, total_synapses, active_synapses / total_synapses * 100)
    @printf("  Mean |H|:                          %.6f\n", mean(abs.(H_flat)))
    @printf("  Max  |H|:                          %.4f\n", maximum(abs.(H_flat)))

    return ratio
end

function run_adaptation_experiment(net::PlasticNet, X_train, y_train, X_test, y_test)
    println("\n   DEPLOYMENT ADAPTATION EXPERIMENT")
    println("  " * "-"^60)
    println("  Phase 1: Evaluate with ZERO Hebbian traces (standard inference)")

    reset_traces!(net)
    m_cold = evaluate(net, X_test, y_test)
    @printf("  Cold Accuracy:  %.2f%%  |  F1: %.4f  |  AUC: %.4f\n",
            m_cold.acc * 100, m_cold.f1, m_cold.auc)

    println("\n  Phase 2: Accumulating traces from training distribution...")

    reset_traces!(net)
    set_mode!(net, false)

    n = size(X_train, 2)
    batch_size = 64
    perm = randperm(n)
    for start in 1:batch_size:n
        idx = perm[start:min(start + batch_size - 1, n)]
        forward_accumulate!(net, X_train[:, idx])
    end

    println("  Phase 3: Evaluate with ADAPTED weights (traces accumulated)")

    m_warm = evaluate(net, X_test, y_test)
    @printf("  Warm Accuracy:  %.2f%%  |  F1: %.4f  |  AUC: %.4f\n",
            m_warm.acc * 100, m_warm.f1, m_warm.auc)

    delta = m_warm.acc - m_cold.acc
    println()
    if delta > 0
        @printf("  Adaptation Gain: +%.2f%% accuracy\n", delta * 100)
        println("  The network successfully adapted its effective weights during deployment.")
    elseif delta == 0
        println("  Adaptation Gain: 0.00% (neutral)")
        println("  The network maintained performance. Traces did not degrade predictions.")
    else
        @printf("  Adaptation Delta: %.2f%% accuracy\n", delta * 100)
        println("  The traces slightly perturbed predictions on this distribution.")
    end

    return m_cold, m_warm
end

function run_hebbian_plasticity()
    println("\n" * "=" * "="^78 * "=")
    println("   DIFFERENTIABLE HEBBIAN PLASTICITY")
    println("   Meta-Learned Synaptic Adaptation at Deployment")
    println("=" * "="^78 * "=")

    X_train, y_train, X_test, y_test = load_data(; top_k=300, verbose=false)
    nf = size(X_train, 1)

    HIDDEN = 128
    net = PlasticNet(nf, HIDDEN, 2)

    params = count_params(net)
    plastic_params = count_plastic_params(net.plastic)

    println("\n Architecture: Encoder($nf -> $HIDDEN) -> PlasticDense($HIDDEN -> $HIDDEN) -> Head($HIDDEN -> 2)")
    println("   Total Parameters: $params")
    @printf("   Plastic Layer: %d params (%d W + %d b + %d alpha)\n",
            plastic_params, length(net.plastic.W), length(net.plastic.b), length(net.plastic.alpha))
    @printf("   Trace Decay: %.2f\n", TRACE_DECAY)
    println("  " * "-"^70)

    n = size(X_train, 2)
    step = 0
    epochs = 100
    batch_size = 32

    best_acc = 0.0
    best_ep = 0

    t0 = now()

    for epoch in 1:epochs
        lr = 2e-3 * 0.5 * (1 + cos(pi * epoch / epochs))

        reset_traces!(net)
        set_mode!(net, true)

        perm = randperm(n)
        for start in 1:batch_size:n
            step += 1
            idx = perm[start:min(start + batch_size - 1, n)]

            X_batch = X_train[:, idx]
            y_oh = onehot(y_train[idx])

            y_pred = forward!(net, X_batch)
            backward!(net, y_oh, y_pred; l2=1e-4, clip=1.0)
            adam_step!(net, lr, step)
        end

        if epoch == 1 || epoch % 10 == 0 || epoch == epochs
            m_tr = evaluate(net, X_train, y_train)
            m_te = evaluate(net, X_test, y_test)

            improved = ""
            if m_te.acc > best_acc
                best_acc = m_te.acc
                best_ep = epoch
                improved = " *"
            end

            @printf("  Ep %3d | TrL %.4f TrA %5.1f%% | TeL %.4f TeA %5.1f%% | F1 %.3f | LR %.1e%s\n",
                    epoch, m_tr.loss, m_tr.acc*100, m_te.loss, m_te.acc*100, m_te.f1, lr, improved)
        end
    end

    elapsed = Dates.value(now() - t0) / 1000.0

    m = evaluate(net, X_test, y_test)
    c = m.cm

    println("\n" * "-"^78)
    println("   TRAINING COMPLETE")
    println("-"^78)
    @printf("  Training Time:   %.1fs\n", elapsed)
    @printf("  Best Accuracy:   %.2f%% (epoch %d)\n", best_acc * 100, best_ep)
    @printf("  Final Accuracy:  %.2f%%\n", m.acc * 100)
    @printf("  Final F1:        %.4f\n", m.f1)
    @printf("  Final AUC:       %.4f\n", m.auc)

    println("\n  Confusion Matrix:")
    println("  +-----------+----------+----------+")
    println("  |           | Pred: 0  | Pred: 1  |")
    println("  +-----------+----------+----------+")
    @printf("  | True: 0   |  %4d    |  %4d    |\n", c.tn, c.fp)
    @printf("  | True: 1   |  %4d    |  %4d    |\n", c.fn, c.tp)
    println("  +-----------+----------+----------+")

    adaptation_ratio = analyze_plasticity(net)

    m_cold, m_warm = run_adaptation_experiment(net, X_train, y_train, X_test, y_test)

    mkpath(joinpath(CORE_PROJECT_DIR, "results"))
    open(joinpath(CORE_PROJECT_DIR, "results", "hebbian_results.txt"), "w") do io
        println(io, "Differentiable Hebbian Plasticity Results -- $(now())")
        println(io, "="^60)
        println(io, "Architecture: Encoder($nf -> $HIDDEN) -> PlasticDense($HIDDEN -> $HIDDEN) -> Head($HIDDEN -> 2)")
        println(io, "Parameters: $params (Plastic: $plastic_params)")
        @printf(io, "Trace Decay: %.2f\n", TRACE_DECAY)
        @printf(io, "\nBest Accuracy: %.4f (epoch %d)\n", best_acc, best_ep)
        @printf(io, "Final Accuracy: %.4f\n", m.acc)
        @printf(io, "Final F1: %.4f\n", m.f1)
        @printf(io, "Final AUC: %.4f\n", m.auc)
        @printf(io, "\nAdaptation Ratio (||alpha*H|| / ||W||): %.4f\n", adaptation_ratio)
        @printf(io, "\nDeployment Adaptation Experiment:\n")
        @printf(io, "  Cold (no traces) Accuracy:   %.4f\n", m_cold.acc)
        @printf(io, "  Warm (adapted) Accuracy:     %.4f\n", m_warm.acc)
        @printf(io, "  Adaptation Delta:            %+.4f\n", m_warm.acc - m_cold.acc)
    end

    println("\n   Results saved to: results/hebbian_results.txt")
end

run_hebbian_plasticity()
