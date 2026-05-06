include("../core.jl")

mutable struct LiquidCell
    dim::Int
    input_dim::Int

    W_x::Matrix{Float64}
    W_h::Matrix{Float64}
    b_h::Vector{Float64}

    W_tau::Matrix{Float64}
    b_tau::Vector{Float64}

    dW_x::Matrix{Float64}
    dW_h::Matrix{Float64}
    db_h::Vector{Float64}
    dW_tau::Matrix{Float64}
    db_tau::Vector{Float64}

    mW_x::Matrix{Float64}; vW_x::Matrix{Float64}
    mW_h::Matrix{Float64}; vW_h::Matrix{Float64}
    mb_h::Vector{Float64}; vb_h::Vector{Float64}
    mW_tau::Matrix{Float64}; vW_tau::Matrix{Float64}
    mb_tau::Vector{Float64}; vb_tau::Vector{Float64}
end

function LiquidCell(input_dim::Int, dim::Int)
    scale_xh = sqrt(2.0 / (input_dim + dim))
    scale_hh = sqrt(2.0 / (dim + dim))

    LiquidCell(
        dim, input_dim,
        randn(dim, input_dim) .* scale_xh,
        randn(dim, dim) .* scale_hh,
        zeros(dim),
        randn(dim, input_dim) .* scale_xh .* 0.5,
        ones(dim),
        zeros(dim, input_dim), zeros(dim, dim), zeros(dim),
        zeros(dim, input_dim), zeros(dim),
        zeros(dim, input_dim), zeros(dim, input_dim),
        zeros(dim, dim), zeros(dim, dim),
        zeros(dim), zeros(dim),
        zeros(dim, input_dim), zeros(dim, input_dim),
        zeros(dim), zeros(dim),
    )
end

function count_cell_params(c::LiquidCell)
    length(c.W_x) + length(c.W_h) + length(c.b_h) +
    length(c.W_tau) + length(c.b_tau)
end

mutable struct LiquidODEBlock
    cell::LiquidCell
    T::Float64
    N::Int
    dt::Float64

    H_trace::Array{Float64, 3}
    tau_trace::Array{Float64, 3}
    f_trace::Array{Float64, 3}
    X_held::Matrix{Float64}
end

function LiquidODEBlock(input_dim::Int, dim::Int; T=1.0, N=20)
    cell = LiquidCell(input_dim, dim)
    LiquidODEBlock(
        cell, Float64(T), N, Float64(T) / N,
        zeros(0, 0, 0), zeros(0, 0, 0), zeros(0, 0, 0),
        zeros(0, 0)
    )
end

function compute_tau(cell::LiquidCell, X::Matrix{Float64})
    raw = cell.W_tau * X .+ cell.b_tau
    softplus_tau = log.(1.0 .+ exp.(clamp.(raw, -20.0, 20.0)))
    return softplus_tau .+ 0.1
end

function compute_dynamics(cell::LiquidCell, h::Matrix{Float64}, X::Matrix{Float64})
    tanh.(cell.W_h * h .+ cell.W_x * X .+ cell.b_h)
end

function fwd!(block::LiquidODEBlock, h0::Matrix{Float64}, X::Matrix{Float64})
    c = block.cell
    dim, B = size(h0)

    block.X_held = X
    block.H_trace = zeros(dim, B, block.N + 1)
    block.tau_trace = zeros(dim, B, block.N)
    block.f_trace = zeros(dim, B, block.N)

    block.H_trace[:, :, 1] = h0

    tau = compute_tau(c, X)

    for k in 1:block.N
        h_k = block.H_trace[:, :, k]
        f_k = compute_dynamics(c, h_k, X)

        block.tau_trace[:, :, k] = tau
        block.f_trace[:, :, k] = f_k

        leak = h_k ./ tau
        dh = -leak .+ f_k
        block.H_trace[:, :, k+1] = h_k .+ block.dt .* dh
    end

    return block.H_trace[:, :, block.N + 1]
end

function bwd!(block::LiquidODEBlock, dY::Matrix{Float64}, λ::Float64, clip::Float64=1.0)
    c = block.cell
    dim, B = size(dY)
    X = block.X_held

    c.dW_x .= 0.0; c.dW_h .= 0.0; c.db_h .= 0.0
    c.dW_tau .= 0.0; c.db_tau .= 0.0

    dh = copy(dY)

    for k in block.N:-1:1
        h_k = block.H_trace[:, :, k]
        tau_k = block.tau_trace[:, :, k]
        f_k = block.f_trace[:, :, k]

        d_leak_h = block.dt .* dh .* (-1.0 ./ tau_k)
        d_f = block.dt .* dh .* (1.0 .- f_k .^ 2)

        c.dW_h .+= (d_f * h_k') ./ B
        c.dW_x .+= (d_f * X') ./ B
        c.db_h .+= vec(sum(d_f, dims=2)) ./ B

        d_tau_contrib = block.dt .* dh .* h_k ./ (tau_k .^ 2)

        raw_tau = c.W_tau * X .+ c.b_tau
        sig_raw = 1.0 ./ (1.0 .+ exp.(-clamp.(raw_tau, -20.0, 20.0)))
        d_tau_raw = d_tau_contrib .* sig_raw

        c.dW_tau .+= (d_tau_raw * X') ./ B
        c.db_tau .+= vec(sum(d_tau_raw, dims=2)) ./ B

        dh_prev = dh .+ d_leak_h .+ c.W_h' * d_f
        dh = dh_prev
    end

    c.dW_x .+= λ .* c.W_x
    c.dW_h .+= λ .* c.W_h
    c.dW_tau .+= λ .* c.W_tau

    gnorm = sqrt(
        sum(c.dW_x .^ 2) + sum(c.dW_h .^ 2) + sum(c.db_h .^ 2) +
        sum(c.dW_tau .^ 2) + sum(c.db_tau .^ 2)
    )
    if gnorm > clip
        s = clip / gnorm
        c.dW_x .*= s; c.dW_h .*= s; c.db_h .*= s
        c.dW_tau .*= s; c.db_tau .*= s
    end

    return dh
end

mutable struct LiquidNet
    encoder::Dense
    liquid::LiquidODEBlock
    head::Dense
    training::Bool
end

function LiquidNet(in_dim::Int, latent_dim::Int, out_dim::Int; T=1.0, N=20)
    enc = Dense(in_dim, latent_dim)
    liq = LiquidODEBlock(in_dim, latent_dim; T=T, N=N)
    hd = Dense(latent_dim, out_dim)
    LiquidNet(enc, liq, hd, true)
end

function set_mode!(n::LiquidNet, training::Bool)
    n.training = training
end

function forward!(net::LiquidNet, X::Matrix{Float64})
    h0 = lrelu.(fwd!(net.encoder, X))
    h_final = fwd!(net.liquid, h0, X)
    softmax_c(fwd!(net.head, h_final))
end

function backward!(net::LiquidNet, y_oh, y_pred; l2=1e-4, clip=1.0)
    d = y_pred .- y_oh
    d = bwd!(net.head, d, l2, clip)
    d = bwd!(net.liquid, d, l2, clip)
    d = d .* lrelu_d.(net.encoder.W * net.encoder.inp .+ net.encoder.b)
    d = bwd!(net.encoder, d, l2, clip)
end

function adam_step!(net::LiquidNet, lr, t; β1=0.9, β2=0.999, ε=1e-8)
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

    c = net.liquid.cell
    @. c.mW_x = β1 * c.mW_x + (1-β1) * c.dW_x
    @. c.vW_x = β2 * c.vW_x + (1-β2) * c.dW_x^2
    @. c.W_x -= lr * (c.mW_x/bc1) / (sqrt(c.vW_x/bc2) + ε)

    @. c.mW_h = β1 * c.mW_h + (1-β1) * c.dW_h
    @. c.vW_h = β2 * c.vW_h + (1-β2) * c.dW_h^2
    @. c.W_h -= lr * (c.mW_h/bc1) / (sqrt(c.vW_h/bc2) + ε)

    @. c.mb_h = β1 * c.mb_h + (1-β1) * c.db_h
    @. c.vb_h = β2 * c.vb_h + (1-β2) * c.db_h^2
    @. c.b_h -= lr * (c.mb_h/bc1) / (sqrt(c.vb_h/bc2) + ε)

    @. c.mW_tau = β1 * c.mW_tau + (1-β1) * c.dW_tau
    @. c.vW_tau = β2 * c.vW_tau + (1-β2) * c.dW_tau^2
    @. c.W_tau -= lr * (c.mW_tau/bc1) / (sqrt(c.vW_tau/bc2) + ε)

    @. c.mb_tau = β1 * c.mb_tau + (1-β1) * c.db_tau
    @. c.vb_tau = β2 * c.vb_tau + (1-β2) * c.db_tau^2
    @. c.b_tau -= lr * (c.mb_tau/bc1) / (sqrt(c.vb_tau/bc2) + ε)
end

function count_params(net::LiquidNet)
    length(net.encoder.W) + length(net.encoder.b) +
    count_cell_params(net.liquid.cell) +
    length(net.head.W) + length(net.head.b)
end

function evaluate(net::LiquidNet, X, y)
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

function analyze_tau_distribution(net::LiquidNet, X::Matrix{Float64})
    h0 = lrelu.(fwd!(net.encoder, X))
    fwd!(net.liquid, h0, X)

    all_tau = net.liquid.tau_trace
    dim, B, steps = size(all_tau)

    tau_flat = reshape(all_tau, dim, B * steps)
    per_neuron_mean = vec(mean(tau_flat, dims=2))
    per_neuron_std = vec(std(tau_flat, dims=2))

    sorted_idx = sortperm(per_neuron_mean)

    println("\n   ADAPTIVE TIME CONSTANT DISTRIBUTION")
    println("  " * "-"^70)
    println("  Neuron  |  Mean Tau  |  Std Tau   | Regime")
    println("  " * "-"^60)

    for i in sorted_idx
        regime = "STANDARD"
        if per_neuron_mean[i] < 0.3
            regime = "FAST (reactive)"
        elseif per_neuron_mean[i] > 2.0
            regime = "SLOW (memory)"
        end
        @printf("  %4d    |   %6.3f   |   %6.4f   | %s\n",
                i, per_neuron_mean[i], per_neuron_std[i], regime)
    end

    global_min = minimum(all_tau)
    global_max = maximum(all_tau)
    global_mean = mean(all_tau)
    spread = global_max / max(global_min, 1e-8)

    println("  " * "-"^60)
    @printf("  Global Min Tau:  %.4f\n", global_min)
    @printf("  Global Max Tau:  %.4f\n", global_max)
    @printf("  Global Mean Tau: %.4f\n", global_mean)
    @printf("  Dynamic Range:   %.1fx\n", spread)

    return per_neuron_mean, per_neuron_std
end

function run_liquid_network()
    println("\n" * "=" * "="^78 * "=")
    println("   LIQUID TIME-CONSTANT NETWORK (LNN)")
    println("   Adaptive ODE with Input-Dependent Temporal Dynamics")
    println("=" * "="^78 * "=")

    X_train, y_train, X_test, y_test = load_data(; top_k=300, verbose=false)
    nf = size(X_train, 1)

    LATENT = 24
    STEPS = 20
    net = LiquidNet(nf, LATENT, 2; T=1.0, N=STEPS)

    params = count_params(net)
    println("\n Architecture: Encoder($nf -> $LATENT) -> LiquidODE(steps=$STEPS) -> Head($LATENT -> 2)")
    println("   Total Parameters: $params")
    println("   Integration: Euler, dt=$(net.liquid.dt), T=$(net.liquid.T)")
    println("  " * "-"^70)

    n = size(X_train, 2)
    step = 0
    epochs = 120
    batch_size = 32

    best_acc = 0.0
    best_ep = 0
    stall_count = 0
    prev_loss = Inf

    t0 = now()

    for epoch in 1:epochs
        lr = 3e-3 * 0.5 * (1 + cos(pi * epoch / epochs))

        perm = randperm(n)
        for start in 1:batch_size:n
            step += 1
            idx = perm[start:min(start + batch_size - 1, n)]

            X_batch = X_train[:, idx]
            y_oh = onehot(y_train[idx])

            y_pred = forward!(net, X_batch)
            backward!(net, y_oh, y_pred; l2=2e-4, clip=1.0)
            adam_step!(net, lr, step)
        end

        if epoch == 1 || epoch % 10 == 0 || epoch == epochs
            m_tr = evaluate(net, X_train, y_train)
            m_te = evaluate(net, X_test, y_test)

            improved = ""
            if m_te.acc > best_acc
                best_acc = m_te.acc
                best_ep = epoch
                stall_count = 0
                improved = " *"
            else
                stall_count += 1
            end

            @printf("  Ep %3d | TrL %.4f TrA %5.1f%% | TeL %.4f TeA %5.1f%% | F1 %.3f | LR %.1e%s\n",
                    epoch, m_tr.loss, m_tr.acc*100, m_te.loss, m_te.acc*100, m_te.f1, lr, improved)
            prev_loss = m_te.loss
        end
    end

    elapsed = Dates.value(now() - t0) / 1000.0

    m = evaluate(net, X_test, y_test)
    c = m.cm

    println("\n" * "-"^78)
    println("   LIQUID NETWORK TRAINING COMPLETE")
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

    println("\n  --- Analyzing Learned Time Constants ---")
    tau_means, tau_stds = analyze_tau_distribution(net, X_test)

    n_fast = sum(tau_means .< 0.3)
    n_slow = sum(tau_means .> 2.0)
    n_mid = LATENT - n_fast - n_slow

    println("\n   TEMPORAL REGIME SUMMARY")
    println("  " * "-"^50)
    @printf("  Fast Neurons (tau < 0.3):   %d / %d\n", n_fast, LATENT)
    @printf("  Standard Neurons:           %d / %d\n", n_mid, LATENT)
    @printf("  Slow/Memory Neurons (>2.0): %d / %d\n", n_slow, LATENT)

    mkpath(joinpath(CORE_PROJECT_DIR, "results"))
    open(joinpath(CORE_PROJECT_DIR, "results", "liquid_nn_results.txt"), "w") do io
        println(io, "Liquid Time-Constant Network Results -- $(now())")
        println(io, "="^60)
        println(io, "Architecture: Encoder($nf -> $LATENT) -> LiquidODE(steps=$STEPS) -> Head($LATENT -> 2)")
        println(io, "Parameters: $params")
        println(io, "Integration Steps: $STEPS, dt=$(net.liquid.dt)")
        @printf(io, "\nBest Accuracy: %.4f (epoch %d)\n", best_acc, best_ep)
        @printf(io, "Final Accuracy: %.4f\n", m.acc)
        @printf(io, "Final F1: %.4f\n", m.f1)
        @printf(io, "Final AUC: %.4f\n", m.auc)
        println(io, "\nTemporal Regime Distribution:")
        @printf(io, "  Fast (tau < 0.3): %d\n", n_fast)
        @printf(io, "  Standard: %d\n", n_mid)
        @printf(io, "  Slow (tau > 2.0): %d\n", n_slow)
        println(io, "\nPer-Neuron Mean Tau:")
        for i in 1:length(tau_means)
            @printf(io, "  Neuron %d: mean=%.4f std=%.4f\n", i, tau_means[i], tau_stds[i])
        end
    end

    println("\n   Results saved to: results/liquid_nn_results.txt")
end

run_liquid_network()
