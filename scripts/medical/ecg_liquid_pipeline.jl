include("../../src/ecg_loader.jl")

const LNN_LATENT = 32
const LNN_STEPS = 25
const LNN_T = 1.0
const LNN_DT = LNN_T / LNN_STEPS
const LNN_TRACE_CLAMP = 5.0

lrelu(x) = x > 0 ? x : 0.01x
lrelu_d(x) = x > 0 ? 1.0 : 0.01

mutable struct TemporalDense
    W::Matrix{Float64}; b::Vector{Float64}
    dW::Matrix{Float64}; db::Vector{Float64}
    mW::Matrix{Float64}; vW::Matrix{Float64}
    mb::Vector{Float64}; vb::Vector{Float64}
    inp::Matrix{Float64}
end

function TemporalDense(ni, no)
    TemporalDense(
        randn(no, ni) .* sqrt(2.0 / ni), zeros(no),
        zeros(no, ni), zeros(no),
        zeros(no, ni), zeros(no, ni),
        zeros(no), zeros(no),
        zeros(0, 0)
    )
end

function fwd!(l::TemporalDense, X)
    l.inp = X
    l.W * X .+ l.b
end

function bwd!(l::TemporalDense, d, λ, clip=1.0)
    m = size(d, 2)
    l.dW = d * l.inp' ./ m .+ λ .* l.W
    l.db = vec(sum(d, dims=2)) ./ m
    gnorm = sqrt(sum(l.dW .^ 2) + sum(l.db .^ 2))
    if gnorm > clip
        s = clip / gnorm
        l.dW .*= s; l.db .*= s
    end
    l.W' * d
end

mutable struct TemporalLiquidCell
    dim::Int
    input_dim::Int

    W_x::Matrix{Float64}
    W_h::Matrix{Float64}
    b_h::Vector{Float64}

    W_tau::Matrix{Float64}
    b_tau::Vector{Float64}

    dW_x::Matrix{Float64}; dW_h::Matrix{Float64}; db_h::Vector{Float64}
    dW_tau::Matrix{Float64}; db_tau::Vector{Float64}

    mW_x::Matrix{Float64}; vW_x::Matrix{Float64}
    mW_h::Matrix{Float64}; vW_h::Matrix{Float64}
    mb_h::Vector{Float64}; vb_h::Vector{Float64}
    mW_tau::Matrix{Float64}; vW_tau::Matrix{Float64}
    mb_tau::Vector{Float64}; vb_tau::Vector{Float64}
end

function TemporalLiquidCell(input_dim, dim)
    sx = sqrt(2.0 / (input_dim + dim))
    sh = sqrt(2.0 / (dim + dim))
    TemporalLiquidCell(
        dim, input_dim,
        randn(dim, input_dim) .* sx,
        randn(dim, dim) .* sh,
        zeros(dim),
        randn(dim, input_dim) .* sx .* 0.5,
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

mutable struct TemporalLiquidODE
    cell::TemporalLiquidCell
    N::Int
    dt::Float64

    H_trace::Array{Float64, 3}
    tau_trace::Array{Float64, 3}
    f_trace::Array{Float64, 3}
    X_seq::Array{Float64, 3}
end

function TemporalLiquidODE(input_dim, dim; N=LNN_STEPS)
    cell = TemporalLiquidCell(input_dim, dim)
    TemporalLiquidODE(
        cell, N, LNN_T / N,
        zeros(0,0,0), zeros(0,0,0), zeros(0,0,0), zeros(0,0,0)
    )
end

function segment_signal(X::Matrix{Float64}, N::Int)
    T, B = size(X)
    seg_len = T ÷ N
    remainder = T - seg_len * N

    segments = zeros(seg_len, B, N)
    for k in 1:N
        s = (k-1) * seg_len + 1
        e = k * seg_len
        segments[:, :, k] = X[s:e, :]
    end

    pooled = zeros(seg_len, B, N)
    for k in 1:N
        pooled[:, :, k] = segments[:, :, k]
    end
    return pooled
end

function compute_tau(c::TemporalLiquidCell, x_seg::Matrix{Float64})
    raw = c.W_tau * x_seg .+ c.b_tau
    softplus = log.(1.0 .+ exp.(clamp.(raw, -20.0, 20.0)))
    return softplus .+ 0.1
end

function fwd!(block::TemporalLiquidODE, X::Matrix{Float64}, h0::Matrix{Float64})
    c = block.cell
    dim, B = size(h0)
    T_signal = size(X, 1)

    seg_len = T_signal ÷ block.N
    block.X_seq = zeros(seg_len, B, block.N)
    for k in 1:block.N
        s = (k-1) * seg_len + 1
        e = k * seg_len
        block.X_seq[:, :, k] = X[s:e, :]
    end

    block.H_trace = zeros(dim, B, block.N + 1)
    block.tau_trace = zeros(dim, B, block.N)
    block.f_trace = zeros(dim, B, block.N)

    block.H_trace[:, :, 1] = h0

    for k in 1:block.N
        h_k = block.H_trace[:, :, k]
        x_k = vec(mean(block.X_seq[:, :, k], dims=1))'
        x_seg = repeat(x_k, c.input_dim, 1)
        x_seg = block.X_seq[:, :, k]

        x_pooled = zeros(seg_len, B)
        x_pooled .= block.X_seq[:, :, k]
        x_mean = mean(x_pooled, dims=1)
        x_input = repeat(x_mean, c.input_dim, 1)[1:c.input_dim, :]

        tau = compute_tau(c, x_input)
        f_k = tanh.(c.W_h * h_k .+ c.W_x * x_input .+ c.b_h)

        block.tau_trace[:, :, k] = tau
        block.f_trace[:, :, k] = f_k

        leak = h_k ./ tau
        dh = -leak .+ f_k
        block.H_trace[:, :, k+1] = h_k .+ block.dt .* dh
    end

    return block.H_trace[:, :, block.N + 1]
end

function bwd!(block::TemporalLiquidODE, dY::Matrix{Float64}, λ::Float64, clip::Float64=1.0)
    c = block.cell
    dim, B = size(dY)
    seg_len = size(block.X_seq, 1)

    c.dW_x .= 0.0; c.dW_h .= 0.0; c.db_h .= 0.0
    c.dW_tau .= 0.0; c.db_tau .= 0.0

    dh = copy(dY)

    for k in block.N:-1:1
        h_k = block.H_trace[:, :, k]
        tau_k = block.tau_trace[:, :, k]
        f_k = block.f_trace[:, :, k]

        x_mean = mean(block.X_seq[:, :, k], dims=1)
        x_input = repeat(x_mean, c.input_dim, 1)[1:c.input_dim, :]

        d_leak = block.dt .* dh .* (-1.0 ./ tau_k)
        d_f = block.dt .* dh .* (1.0 .- f_k .^ 2)

        c.dW_h .+= (d_f * h_k') ./ B
        c.dW_x .+= (d_f * x_input') ./ B
        c.db_h .+= vec(sum(d_f, dims=2)) ./ B

        d_tau = block.dt .* dh .* h_k ./ (tau_k .^ 2)
        raw_tau = c.W_tau * x_input .+ c.b_tau
        sig = 1.0 ./ (1.0 .+ exp.(-clamp.(raw_tau, -20.0, 20.0)))
        d_tau_raw = d_tau .* sig

        c.dW_tau .+= (d_tau_raw * x_input') ./ B
        c.db_tau .+= vec(sum(d_tau_raw, dims=2)) ./ B

        dh = dh .+ d_leak .+ c.W_h' * d_f
    end

    c.dW_x .+= λ .* c.W_x
    c.dW_h .+= λ .* c.W_h
    c.dW_tau .+= λ .* c.W_tau

    gnorm = sqrt(sum(c.dW_x .^2) + sum(c.dW_h .^2) + sum(c.db_h .^2) +
                 sum(c.dW_tau .^2) + sum(c.db_tau .^2))
    if gnorm > clip
        s = clip / gnorm
        c.dW_x .*= s; c.dW_h .*= s; c.db_h .*= s
        c.dW_tau .*= s; c.db_tau .*= s
    end

    return dh
end

mutable struct ECGLiquidNet
    proj::TemporalDense
    liquid::TemporalLiquidODE
    head::TemporalDense
    training::Bool
end

function ECGLiquidNet(signal_len, proj_dim, latent_dim, n_classes; N=LNN_STEPS)
    seg_len = signal_len ÷ N
    proj = TemporalDense(signal_len, latent_dim)
    liq = TemporalLiquidODE(seg_len, latent_dim; N=N)
    head = TemporalDense(latent_dim, n_classes)
    ECGLiquidNet(proj, liq, head, true)
end

function forward!(net::ECGLiquidNet, X::Matrix{Float64})
    h0 = lrelu.(fwd!(net.proj, X))
    h_final = fwd!(net.liquid, X, h0)
    ecg_softmax(fwd!(net.head, h_final))
end

function backward!(net::ECGLiquidNet, y_oh, y_pred; l2=1e-4, clip=1.0)
    d = y_pred .- y_oh
    d = bwd!(net.head, d, l2, clip)
    d = bwd!(net.liquid, d, l2, clip)
    d = d .* lrelu_d.(net.proj.W * net.proj.inp .+ net.proj.b)
    bwd!(net.proj, d, l2, clip)
end

function adam_step!(net::ECGLiquidNet, lr, t; β1=0.9, β2=0.999, ε=1e-8)
    bc1 = 1.0 - β1^t; bc2 = 1.0 - β2^t

    function upd!(l::TemporalDense)
        @. l.mW = β1 * l.mW + (1-β1) * l.dW
        @. l.vW = β2 * l.vW + (1-β2) * l.dW^2
        @. l.mb = β1 * l.mb + (1-β1) * l.db
        @. l.vb = β2 * l.vb + (1-β2) * l.db^2
        @. l.W -= lr * (l.mW/bc1) / (sqrt(l.vW/bc2) + ε)
        @. l.b -= lr * (l.mb/bc1) / (sqrt(l.vb/bc2) + ε)
    end

    upd!(net.proj)
    upd!(net.head)

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

function count_params(net::ECGLiquidNet)
    c = net.liquid.cell
    length(net.proj.W) + length(net.proj.b) +
    length(c.W_x) + length(c.W_h) + length(c.b_h) +
    length(c.W_tau) + length(c.b_tau) +
    length(net.head.W) + length(net.head.b)
end

function run_ecg_liquid()
    println("\n" * "="^80)
    println("   LIQUID TIME-CONSTANT NETWORK v2 — ECG Arrhythmia Classification")
    println("   MIT-BIH Dataset | 5-Class Cardiac Waveform Analysis")
    println("="^80)

    train_path = joinpath(ECG_PROJECT_DIR, "data", "archive", "mitbih_train.csv")
    test_path = joinpath(ECG_PROJECT_DIR, "data", "archive", "mitbih_test.csv")

    println("\n  Indexing training data...")
    ds_train = index_csv(train_path)
    println("  Indexing test data...")
    ds_test = index_csv(test_path)

    print_dataset_info(ds_train, "Train")
    print_dataset_info(ds_test, "Test")

    println("\n  Loading test set into memory for evaluation...")
    X_test, y_test = load_full(ds_test)

    LATENT = LNN_LATENT
    net = ECGLiquidNet(ECG_TIMESTEPS, LATENT, LATENT, ECG_CLASSES; N=LNN_STEPS)
    params = count_params(net)

    seg_len = ECG_TIMESTEPS ÷ LNN_STEPS
    println("\n  Architecture:")
    println("    Projection: $ECG_TIMESTEPS -> $LATENT")
    println("    LiquidODE: $LNN_STEPS integration steps, segment_len=$seg_len")
    println("    Classifier: $LATENT -> $ECG_CLASSES")
    println("    Total Parameters: $params")
    println("  " * "-"^70)

    epochs = 30
    batch_size = 128
    step = 0
    best_acc = 0.0
    best_f1 = 0.0

    t0 = now()

    for epoch in 1:epochs
        lr = 3e-3 * 0.5 * (1 + cos(pi * epoch / epochs))

        perm = randperm(ds_train.n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in 1:batch_size:ds_train.n_samples
            step += 1
            stop = min(start + batch_size - 1, ds_train.n_samples)
            idx = perm[start:stop]

            X_batch, y_batch = load_batch(ds_train, idx)
            y_oh = onehot_ecg(y_batch)

            y_pred = forward!(net, X_batch)
            backward!(net, y_oh, y_pred; l2=2e-4, clip=1.0)
            adam_step!(net, lr, step)

            epoch_loss += ecg_xent(y_pred, y_oh)
            n_batches += 1
        end

        avg_loss = epoch_loss / n_batches

        if epoch == 1 || epoch % 3 == 0 || epoch == epochs
            y_pred_test = forward!(net, X_test)
            m = ecg_metrics(y_pred_test, y_test)

            improved = ""
            if m.acc > best_acc
                best_acc = m.acc
                best_f1 = m.macro_f1
                improved = " *"
            end

            @printf("  Ep %2d | Loss %.4f | Acc %5.1f%% | MacF1 %.3f | WgtF1 %.3f | LR %.1e%s\n",
                    epoch, avg_loss, m.acc*100, m.macro_f1, m.weighted_f1, lr, improved)
        end
    end

    elapsed = Dates.value(now() - t0) / 1000.0

    y_pred_final = forward!(net, X_test)
    m = ecg_metrics(y_pred_final, y_test)

    println("\n" * "-"^80)
    println("   ECG LIQUID NETWORK TRAINING COMPLETE")
    println("-"^80)
    @printf("  Training Time:    %.1fs\n", elapsed)
    @printf("  Best Accuracy:    %.2f%%\n", best_acc * 100)
    @printf("  Final Accuracy:   %.2f%%\n", m.acc * 100)
    @printf("  Macro F1:         %.4f\n", m.macro_f1)
    @printf("  Weighted F1:      %.4f\n", m.weighted_f1)

    class_names = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]
    println("\n  Per-Class Performance:")
    println("  " * "-"^55)
    for c in 1:ECG_CLASSES
        @printf("  %s: Prec=%.3f  Rec=%.3f  F1=%.3f\n",
                rpad(class_names[c], 18), m.per_class_prec[c], m.per_class_rec[c], m.per_class_f1[c])
    end

    print_confusion_matrix(m.preds, y_test)

    mkpath(joinpath(ECG_PROJECT_DIR, "results"))
    open(joinpath(ECG_PROJECT_DIR, "results", "ecg_liquid_results.txt"), "w") do io
        println(io, "ECG Liquid Time-Constant Network Results -- $(now())")
        println(io, "="^60)
        println(io, "Dataset: MIT-BIH Arrhythmia (109K samples, 5 classes)")
        println(io, "Architecture: Proj($ECG_TIMESTEPS->$LATENT) -> LiquidODE($LNN_STEPS steps) -> Head($LATENT->$ECG_CLASSES)")
        println(io, "Parameters: $params")
        @printf(io, "\nBest Accuracy: %.4f\n", best_acc)
        @printf(io, "Final Accuracy: %.4f\n", m.acc)
        @printf(io, "Macro F1: %.4f\n", m.macro_f1)
        @printf(io, "Weighted F1: %.4f\n", m.weighted_f1)
        println(io, "\nPer-Class F1:")
        for c in 1:ECG_CLASSES
            @printf(io, "  %s: %.4f\n", class_names[c], m.per_class_f1[c])
        end
    end

    println("\n  Results saved to: results/ecg_liquid_results.txt")
end

run_ecg_liquid()
