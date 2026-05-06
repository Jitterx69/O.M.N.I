include("../core.jl")

mutable struct StochasticBottleneck
    h_dim::Int
    z_dim::Int

    W_mu::Matrix{Float64}
    b_mu::Vector{Float64}
    W_lv::Matrix{Float64}
    b_lv::Vector{Float64}

    dW_mu::Matrix{Float64}; db_mu::Vector{Float64}
    dW_lv::Matrix{Float64}; db_lv::Vector{Float64}

    mW_mu::Matrix{Float64}; vW_mu::Matrix{Float64}
    mb_mu::Vector{Float64}; vb_mu::Vector{Float64}
    mW_lv::Matrix{Float64}; vW_lv::Matrix{Float64}
    mb_lv::Vector{Float64}; vb_lv::Vector{Float64}

    inp::Matrix{Float64}
    mu::Matrix{Float64}
    lv::Matrix{Float64}
    eps::Matrix{Float64}
    z::Matrix{Float64}
end

function StochasticBottleneck(h_dim::Int, z_dim::Int)
    StochasticBottleneck(
        h_dim, z_dim,
        randn(z_dim, h_dim) .* sqrt(2.0 / h_dim),
        zeros(z_dim),
        randn(z_dim, h_dim) .* 0.01,
        zeros(z_dim),
        zeros(z_dim, h_dim), zeros(z_dim),
        zeros(z_dim, h_dim), zeros(z_dim),
        zeros(z_dim, h_dim), zeros(z_dim, h_dim),
        zeros(z_dim), zeros(z_dim),
        zeros(z_dim, h_dim), zeros(z_dim, h_dim),
        zeros(z_dim), zeros(z_dim),
        zeros(0, 0), zeros(0, 0), zeros(0, 0),
        zeros(0, 0), zeros(0, 0)
    )
end

function count_bottleneck_params(sb::StochasticBottleneck)
    length(sb.W_mu) + length(sb.b_mu) + length(sb.W_lv) + length(sb.b_lv)
end

function fwd!(sb::StochasticBottleneck, h::Matrix{Float64}; training::Bool=true)
    sb.inp = h
    sb.mu = sb.W_mu * h .+ sb.b_mu
    sb.lv = sb.W_lv * h .+ sb.b_lv
    clamp!(sb.lv, -10.0, 10.0)

    if training
        sb.eps = randn(size(sb.mu))
        sb.z = sb.mu .+ exp.(0.5 .* sb.lv) .* sb.eps
    else
        sb.z = copy(sb.mu)
    end
    return sb.z
end

function kl_divergence(sb::StochasticBottleneck)
    B = size(sb.mu, 2)
    kl_per_sample = 0.5 .* sum(sb.mu .^ 2 .+ exp.(sb.lv) .- 1.0 .- sb.lv, dims=1)
    return mean(kl_per_sample)
end

function kl_per_dimension(sb::StochasticBottleneck)
    0.5 .* (vec(mean(sb.mu .^ 2, dims=2)) .+
            vec(mean(exp.(sb.lv), dims=2)) .- 1.0 .-
            vec(mean(sb.lv, dims=2)))
end

function bwd!(sb::StochasticBottleneck, dz::Matrix{Float64}, β::Float64, λ::Float64, clip::Float64=1.0)
    B = size(dz, 2)

    dmu = dz .+ β .* sb.mu
    sigma = exp.(0.5 .* sb.lv)
    dlv = dz .* 0.5 .* sigma .* sb.eps .+ β .* 0.5 .* (exp.(sb.lv) .- 1.0)

    sb.dW_mu = dmu * sb.inp' ./ B .+ λ .* sb.W_mu
    sb.db_mu = vec(sum(dmu, dims=2)) ./ B
    sb.dW_lv = dlv * sb.inp' ./ B .+ λ .* sb.W_lv
    sb.db_lv = vec(sum(dlv, dims=2)) ./ B

    gnorm = sqrt(sum(sb.dW_mu .^ 2) + sum(sb.db_mu .^ 2) +
                 sum(sb.dW_lv .^ 2) + sum(sb.db_lv .^ 2))
    if gnorm > clip
        s = clip / gnorm
        sb.dW_mu .*= s; sb.db_mu .*= s
        sb.dW_lv .*= s; sb.db_lv .*= s
    end

    sb.W_mu' * dmu .+ sb.W_lv' * dlv
end

mutable struct IBNet
    encoder::Dense
    bottleneck::StochasticBottleneck
    classifier::Dense
    training::Bool
end

function IBNet(in_dim::Int, h_dim::Int, z_dim::Int, out_dim::Int)
    IBNet(Dense(in_dim, h_dim), StochasticBottleneck(h_dim, z_dim), Dense(z_dim, out_dim), true)
end

function set_mode!(n::IBNet, training::Bool)
    n.training = training
end

function forward!(net::IBNet, X::Matrix{Float64})
    h = lrelu.(fwd!(net.encoder, X))
    z = fwd!(net.bottleneck, h; training=net.training)
    softmax_c(fwd!(net.classifier, z))
end

function backward!(net::IBNet, y_oh, y_pred, β; l2=1e-4, clip=1.0)
    d = y_pred .- y_oh
    dz = bwd!(net.classifier, d, l2, clip)
    dh = bwd!(net.bottleneck, dz, β, l2, clip)
    dh = dh .* lrelu_d.(net.encoder.W * net.encoder.inp .+ net.encoder.b)
    bwd!(net.encoder, dh, l2, clip)
end

function adam_step!(net::IBNet, lr, t; β1=0.9, β2=0.999, ε=1e-8)
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
    update_dense!(net.classifier)

    sb = net.bottleneck
    @. sb.mW_mu = β1 * sb.mW_mu + (1-β1) * sb.dW_mu
    @. sb.vW_mu = β2 * sb.vW_mu + (1-β2) * sb.dW_mu^2
    @. sb.W_mu -= lr * (sb.mW_mu/bc1) / (sqrt(sb.vW_mu/bc2) + ε)

    @. sb.mb_mu = β1 * sb.mb_mu + (1-β1) * sb.db_mu
    @. sb.vb_mu = β2 * sb.vb_mu + (1-β2) * sb.db_mu^2
    @. sb.b_mu -= lr * (sb.mb_mu/bc1) / (sqrt(sb.vb_mu/bc2) + ε)

    @. sb.mW_lv = β1 * sb.mW_lv + (1-β1) * sb.dW_lv
    @. sb.vW_lv = β2 * sb.vW_lv + (1-β2) * sb.dW_lv^2
    @. sb.W_lv -= lr * (sb.mW_lv/bc1) / (sqrt(sb.vW_lv/bc2) + ε)

    @. sb.mb_lv = β1 * sb.mb_lv + (1-β1) * sb.db_lv
    @. sb.vb_lv = β2 * sb.vb_lv + (1-β2) * sb.db_lv^2
    @. sb.b_lv -= lr * (sb.mb_lv/bc1) / (sqrt(sb.vb_lv/bc2) + ε)
end

function count_params(net::IBNet)
    length(net.encoder.W) + length(net.encoder.b) +
    count_bottleneck_params(net.bottleneck) +
    length(net.classifier.W) + length(net.classifier.b)
end

function evaluate(net::IBNet, X, y)
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

function analyze_bottleneck(net::IBNet, X::Matrix{Float64})
    set_mode!(net, true)
    h = lrelu.(fwd!(net.encoder, X))
    fwd!(net.bottleneck, h; training=true)

    kl_dims = kl_per_dimension(net.bottleneck)
    total_kl = sum(kl_dims)
    z_dim = net.bottleneck.z_dim

    active_threshold = 0.01
    active = sum(kl_dims .> active_threshold)
    collapsed = z_dim - active

    sorted_idx = sortperm(kl_dims, rev=true)

    println("\n   INFORMATION BOTTLENECK ANALYSIS")
    println("  " * "-"^60)
    println("  Dim  |  KL(d)   |  % of Total  | Status")
    println("  " * "-"^55)
    for i in sorted_idx
        pct = kl_dims[i] / max(total_kl, 1e-8) * 100
        status = kl_dims[i] > active_threshold ? "ACTIVE" : "COLLAPSED"
        @printf("  %3d  |  %7.4f  |    %5.1f%%     | %s\n", i, kl_dims[i], pct, status)
    end

    println("  " * "-"^55)
    @printf("  Total KL Divergence:    %.4f nats\n", total_kl)
    @printf("  Active Dimensions:      %d / %d\n", active, z_dim)
    @printf("  Collapsed Dimensions:   %d / %d\n", collapsed, z_dim)
    @printf("  Effective Compression:  %.1f%% of latent space pruned\n",
            collapsed / z_dim * 100)

    mu_spread = vec(std(net.bottleneck.mu, dims=2))
    @printf("\n  Mean Spread of mu:      %.4f\n", mean(mu_spread))
    @printf("  Max  Spread of mu:      %.4f\n", maximum(mu_spread))

    return kl_dims, active
end

function run_information_bottleneck()
    println("\n" * "=" * "="^78 * "=")
    println("   VARIATIONAL INFORMATION BOTTLENECK (VIB)")
    println("   Optimal Compression via Mutual Information Minimization")
    println("=" * "="^78 * "=")

    X_train, y_train, X_test, y_test = load_data(; top_k=300, verbose=false)
    nf = size(X_train, 1)

    H_DIM = 64
    Z_DIM = 16
    BETA_START = 1e-4
    BETA_END = 0.05

    net = IBNet(nf, H_DIM, Z_DIM, 2)
    params = count_params(net)

    println("\n Architecture: Encoder($nf -> $H_DIM) -> Bottleneck(mu+lv -> $Z_DIM) -> Classifier($Z_DIM -> 2)")
    println("   Total Parameters: $params")
    @printf("   Beta Schedule: %.1e -> %.1e (linear warmup)\n", BETA_START, BETA_END)
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
        β_ib = BETA_START + (BETA_END - BETA_START) * min(epoch / 60.0, 1.0)

        set_mode!(net, true)
        perm = randperm(n)
        for start in 1:batch_size:n
            step += 1
            idx = perm[start:min(start + batch_size - 1, n)]

            X_batch = X_train[:, idx]
            y_oh = onehot(y_train[idx])

            y_pred = forward!(net, X_batch)
            backward!(net, y_oh, y_pred, β_ib; l2=1e-4, clip=1.0)
            adam_step!(net, lr, step)
        end

        if epoch == 1 || epoch % 10 == 0 || epoch == epochs
            m_te = evaluate(net, X_test, y_test)
            kl_val = kl_divergence(net.bottleneck)

            improved = ""
            if m_te.acc > best_acc
                best_acc = m_te.acc
                best_ep = epoch
                improved = " *"
            end

            @printf("  Ep %3d | TeA %5.1f%% | F1 %.3f | KL %.3f | beta %.1e | LR %.1e%s\n",
                    epoch, m_te.acc*100, m_te.f1, kl_val, β_ib, lr, improved)
        end
    end

    elapsed = Dates.value(now() - t0) / 1000.0

    m = evaluate(net, X_test, y_test)
    c = m.cm

    println("\n" * "-"^78)
    println("   VIB TRAINING COMPLETE")
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

    kl_dims, n_active = analyze_bottleneck(net, X_test)

    mkpath(joinpath(CORE_PROJECT_DIR, "results"))
    open(joinpath(CORE_PROJECT_DIR, "results", "ib_results.txt"), "w") do io
        println(io, "Variational Information Bottleneck Results -- $(now())")
        println(io, "="^60)
        println(io, "Architecture: Encoder($nf -> $H_DIM) -> Bottleneck($Z_DIM) -> Classifier($Z_DIM -> 2)")
        println(io, "Parameters: $params")
        @printf(io, "Beta: %.1e -> %.1e\n", BETA_START, BETA_END)
        @printf(io, "\nBest Accuracy: %.4f (epoch %d)\n", best_acc, best_ep)
        @printf(io, "Final Accuracy: %.4f\n", m.acc)
        @printf(io, "Final F1: %.4f\n", m.f1)
        @printf(io, "Final AUC: %.4f\n", m.auc)
        @printf(io, "\nActive Latent Dimensions: %d / %d\n", n_active, Z_DIM)
        @printf(io, "Effective Compression: %.1f%%\n", (Z_DIM - n_active) / Z_DIM * 100)
        println(io, "\nPer-Dimension KL:")
        for (i, kl) in enumerate(kl_dims)
            @printf(io, "  Dim %d: %.6f %s\n", i, kl, kl > 0.01 ? "ACTIVE" : "COLLAPSED")
        end
    end

    println("\n   Results saved to: results/ib_results.txt")
end

run_information_bottleneck()
