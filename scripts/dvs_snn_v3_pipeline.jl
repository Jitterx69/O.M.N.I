include("../src/dvs_loader.jl")

const SURROGATE_SCALE = 5.0
const V_THRESHOLD = 1.0
const V_LEAK = 0.9
const SNN_T_STEPS = DVS_TIME_BINS
const SPIKE_DROPOUT = 0.2

lrelu(x) = x > 0 ? x : 0.01x
lrelu_d(x) = x > 0 ? 1.0 : 0.01

function surrogate_grad(x)
    1.0 / (1.0 + SURROGATE_SCALE * abs(x))^2
end

function softmax_s(X)
    mx = maximum(X, dims=1)
    e = exp.(X .- mx)
    e ./ sum(e, dims=1)
end

mutable struct SpikeDense
    ni::Int; no::Int
    W::Matrix{Float64}; b::Vector{Float64}
    dW::Matrix{Float64}; db::Vector{Float64}
    mW::Matrix{Float64}; vW::Matrix{Float64}
    mb::Vector{Float64}; vb::Vector{Float64}
end

function SpikeDense(ni, no)
    SpikeDense(ni, no,
        randn(no, ni) .* sqrt(2.0 / ni), zeros(no),
        zeros(no, ni), zeros(no),
        zeros(no, ni), zeros(no, ni),
        zeros(no), zeros(no))
end

mutable struct LIFBlock
    layer::SpikeDense
    U::Array{Float64, 3}
    S::Array{Float64, 3}
    inp::Array{Float64, 3}
end

function LIFBlock(ni, no)
    LIFBlock(SpikeDense(ni, no), zeros(0,0,0), zeros(0,0,0), zeros(0,0,0))
end

function fwd_lif!(blk::LIFBlock, X_seq::Array{Float64, 3}; training=false)
    no = blk.layer.no
    B = size(X_seq, 2)
    T = size(X_seq, 3)
    blk.inp = X_seq

    blk.U = zeros(no, B, T + 1)
    blk.S = zeros(no, B, T)

    for t in 1:T
        curr = blk.layer.W * X_seq[:, :, t] .+ blk.layer.b
        blk.U[:, :, t+1] = V_LEAK .* blk.U[:, :, t] .+ (1.0 - V_LEAK) .* curr
        spikes = Float64.(blk.U[:, :, t+1] .> V_THRESHOLD)
        if training
            mask = rand(size(spikes)...) .> SPIKE_DROPOUT
            spikes .*= mask
        end
        blk.S[:, :, t] = spikes
        blk.U[:, :, t+1] .-= spikes .* V_THRESHOLD
    end

    return blk.S
end

function bwd_lif!(blk::LIFBlock, dS::Array{Float64, 3}, lambda, clip=1.0)
    l = blk.layer
    no, B, T = size(dS)

    l.dW .= 0.0; l.db .= 0.0
    dU = zeros(no, B)
    dX = zeros(size(blk.inp))

    for t in T:-1:1
        grad_s = surrogate_grad.(blk.U[:, :, t+1] .- V_THRESHOLD)
        dU_curr = dS[:, :, t] .+ dU .* V_LEAK
        dU_fire = dU_curr .* grad_s
        dI = dU_fire .* (1.0 - V_LEAK)

        l.dW .+= (dI * blk.inp[:, :, t]') ./ B
        l.db .+= vec(sum(dI, dims=2)) ./ B
        dX[:, :, t] = l.W' * dI
        dU = dU_curr .- dU_fire .* V_THRESHOLD
    end

    l.dW .+= lambda .* l.W
    gnorm = sqrt(sum(l.dW.^2) + sum(l.db.^2))
    if gnorm > clip
        s = clip / gnorm; l.dW .*= s; l.db .*= s
    end
    return dX
end

mutable struct TemporalAttention
    w::Vector{Float64}
    dw::Vector{Float64}
    mw::Vector{Float64}
    vw::Vector{Float64}
end

function TemporalAttention(T::Int)
    TemporalAttention(zeros(T), zeros(T), zeros(T), zeros(T))
end

function apply_temporal_attention(attn::TemporalAttention, S::Array{Float64, 3})
    no, B, T = size(S)
    mx = maximum(attn.w)
    ew = exp.(attn.w .- mx)
    alpha = ew ./ sum(ew)

    out = zeros(no, B)
    for t in 1:T
        out .+= alpha[t] .* S[:, :, t]
    end
    return out, alpha
end

mutable struct GestureSNNv3
    proj::SpikeDense
    lif1::LIFBlock
    lif2::LIFBlock
    lif3::LIFBlock
    attn::TemporalAttention
    head::SpikeDense
    training::Bool
end

function GestureSNNv3(in_dim, proj_dim, h1, h2, h3, n_classes, T)
    GestureSNNv3(
        SpikeDense(in_dim, proj_dim),
        LIFBlock(proj_dim, h1),
        LIFBlock(h1, h2),
        LIFBlock(h2, h3),
        TemporalAttention(T),
        SpikeDense(h3, n_classes),
        true)
end

function reshape_voxel_temporal_v3(X::Matrix{Float64}, n_bins::Int)
    feat_dim, B = size(X)
    R = DVS_RES_DOWN
    spatial_dim = R * R
    channel_per_bin = 2
    flat_per_bin = channel_per_bin * spatial_dim

    X_seq = zeros(flat_per_bin, B, n_bins)
    for t in 1:n_bins
        for pol in 0:1
            ch = pol * n_bins + (t - 1)
            src_start = ch * spatial_dim + 1
            src_end = src_start + spatial_dim - 1
            if src_end > feat_dim; continue; end
            dst_start = pol * spatial_dim + 1
            dst_end = dst_start + spatial_dim - 1
            X_seq[dst_start:dst_end, :, t] = X[src_start:src_end, :]
        end
    end
    return X_seq
end

function forward_snn_v3!(net::GestureSNNv3, X::Matrix{Float64})
    B = size(X, 2)
    X_seq = reshape_voxel_temporal_v3(X, SNN_T_STEPS)
    spatial_in = size(X_seq, 1)

    proj_seq = zeros(net.lif1.layer.ni, B, SNN_T_STEPS)
    for t in 1:SNN_T_STEPS
        proj_seq[:, :, t] = lrelu.(net.proj.W * X_seq[:, :, t] .+ net.proj.b)
    end

    h1 = fwd_lif!(net.lif1, proj_seq; training=net.training)
    h2 = fwd_lif!(net.lif2, h1; training=net.training)
    h3 = fwd_lif!(net.lif3, h2; training=net.training)

    readout, _ = apply_temporal_attention(net.attn, h3)
    return softmax_s(net.head.W * readout .+ net.head.b)
end

function backward_snn_v3!(net::GestureSNNv3, X::Matrix{Float64}, y_oh, y_pred, class_weights; l2=1e-4, clip=1.0)
    B = size(y_oh, 2)
    d = y_pred .- y_oh

    for i in 1:B
        cls = argmax(y_oh[:, i])
        d[:, i] .*= class_weights[cls]
    end

    readout, alpha = apply_temporal_attention(net.attn, net.lif3.S)
    net.head.dW = d * readout' ./ B .+ l2 .* net.head.W
    net.head.db = vec(sum(d, dims=2)) ./ B

    gnorm = sqrt(sum(net.head.dW.^2) + sum(net.head.db.^2))
    if gnorm > clip
        s = clip / gnorm; net.head.dW .*= s; net.head.db .*= s
    end

    d_readout = net.head.W' * d

    no, _, T = size(net.lif3.S)
    net.attn.dw .= 0.0
    for t in 1:T
        net.attn.dw[t] = sum(d_readout .* net.lif3.S[:, :, t]) / B
    end

    dS3 = zeros(size(net.lif3.S))
    for t in 1:T
        dS3[:, :, t] = d_readout .* alpha[t]
    end

    dS2 = bwd_lif!(net.lif3, dS3, l2, clip)
    dS1 = bwd_lif!(net.lif2, dS2, l2, clip)
    dX_proj = bwd_lif!(net.lif1, dS1, l2, clip)

    net.proj.dW .= 0.0; net.proj.db .= 0.0
    X_seq = reshape_voxel_temporal_v3(X, SNN_T_STEPS)

    for t in 1:SNN_T_STEPS
        pre_act = net.proj.W * X_seq[:, :, t] .+ net.proj.b
        d_proj = dX_proj[:, :, t] .* lrelu_d.(pre_act)
        net.proj.dW .+= (d_proj * X_seq[:, :, t]') ./ B
        net.proj.db .+= vec(sum(d_proj, dims=2)) ./ B
    end
    net.proj.dW .+= l2 .* net.proj.W

    gnorm_proj = sqrt(sum(net.proj.dW.^2) + sum(net.proj.db.^2))
    if gnorm_proj > clip
        s = clip / gnorm_proj; net.proj.dW .*= s; net.proj.db .*= s
    end
end

function adam_snn_v3!(net::GestureSNNv3, lr, t; b1=0.9, b2=0.999, eps=1e-8)
    bc1 = 1.0 - b1^t; bc2 = 1.0 - b2^t

    function upd!(l::SpikeDense)
        @. l.mW = b1*l.mW + (1-b1)*l.dW; @. l.vW = b2*l.vW + (1-b2)*l.dW^2
        @. l.W -= lr*(l.mW/bc1)/(sqrt(l.vW/bc2)+eps)
        @. l.mb = b1*l.mb + (1-b1)*l.db; @. l.vb = b2*l.vb + (1-b2)*l.db^2
        @. l.b -= lr*(l.mb/bc1)/(sqrt(l.vb/bc2)+eps)
    end

    upd!(net.proj)
    upd!(net.lif1.layer)
    upd!(net.lif2.layer)
    upd!(net.lif3.layer)
    upd!(net.head)

    a = net.attn
    @. a.mw = b1*a.mw + (1-b1)*a.dw; @. a.vw = b2*a.vw + (1-b2)*a.dw^2
    @. a.w -= lr*(a.mw/bc1)/(sqrt(a.vw/bc2)+eps)
end

function dvs_onehot(y::Vector{Int}, n_classes::Int)
    oh = zeros(n_classes, length(y))
    for (i, v) in enumerate(y)
        oh[v, i] = 1.0
    end
    oh
end

function compute_class_weights(y::Vector{Int}, n_classes::Int)
    counts = zeros(n_classes)
    for v in y; counts[v] += 1.0; end
    total = length(y)
    weights = zeros(n_classes)
    for c in 1:n_classes
        weights[c] = total / (n_classes * max(counts[c], 1.0))
    end
    mx = maximum(weights)
    weights ./= mx
    return weights
end

function run_dvs_snn_v3()
    println("\n" * "█"^80)
    println("   NEUROMORPHIC GESTURE RECOGNITION — SNN v3 (Hyper-Tuned)")
    println("   Spatial Downsampling | Temporal Attention | Class-Weighted Loss")
    println("█"^80)

    train_list = read_trial_list(joinpath(DVS_DATA_DIR, "trials_to_train.txt"))
    test_list = read_trial_list(joinpath(DVS_DATA_DIR, "trials_to_test.txt"))

    println("\n  Loading training trials ($(length(train_list)) files)...")
    X_train, y_train = load_split(train_list, DVS_TIME_BINS)
    println("  Loading test trials ($(length(test_list)) files)...")
    X_test, y_test = load_split(test_list, DVS_TIME_BINS)

    class_weights = compute_class_weights(y_train, DVS_CLASSES)

    R = DVS_RES_DOWN
    spatial_in = 2 * R * R
    PROJ_DIM = 512
    H1 = 256
    H2 = 128
    H3 = 64

    println("\n  Dataset: $(length(y_train)) train, $(length(y_test)) test")
    println("  Feature Dim: $(size(X_train, 1)) (2 pol x $DVS_TIME_BINS bins x $(R)x$(R))")
    println("  Spatial In per Timestep: $spatial_in")
    println("\n  Architecture:")
    println("    Projection:  $spatial_in -> $PROJ_DIM")
    println("    LIF-1:       $PROJ_DIM -> $H1")
    println("    LIF-2:       $H1 -> $H2")
    println("    LIF-3:       $H2 -> $H3")
    println("    Attn Readout: T=$SNN_T_STEPS temporal bins")
    println("    Head:        $H3 -> $DVS_CLASSES")

    print("  Class Weights: ")
    for c in 1:DVS_CLASSES
        @printf("%.2f ", class_weights[c])
    end
    println()
    println("  " * "-"^70)

    net = GestureSNNv3(spatial_in, PROJ_DIM, H1, H2, H3, DVS_CLASSES, SNN_T_STEPS)

    n = length(y_train)
    epochs = 80
    batch_size = 16
    step = 0
    best_acc = 0.0
    best_ep = 0

    t0 = now()

    for epoch in 1:epochs
        base_lr = 2e-3
        if epoch <= 5
            lr = base_lr * (epoch / 5.0)
        else
            lr = base_lr * 0.5 * (1 + cos(pi * (epoch - 5) / (epochs - 5)))
        end

        perm = randperm(n)
        epoch_correct = 0
        epoch_total = 0
        net.training = true

        for start in 1:batch_size:n
            step += 1
            stop = min(start + batch_size - 1, n)
            idx = perm[start:stop]

            X_b = X_train[:, idx]
            y_b = y_train[idx]
            y_oh = dvs_onehot(y_b, DVS_CLASSES)

            y_pred = forward_snn_v3!(net, X_b)
            backward_snn_v3!(net, X_b, y_oh, y_pred, class_weights; l2=1e-4, clip=1.0)
            adam_snn_v3!(net, lr, step)

            preds = [argmax(y_pred[:, i]) for i in 1:size(y_pred, 2)]
            epoch_correct += sum(preds .== y_b)
            epoch_total += length(y_b)
        end

        train_acc = epoch_correct / epoch_total

        if epoch <= 5 || epoch % 5 == 0 || epoch == epochs
            net.training = false
            test_pred = forward_snn_v3!(net, X_test)
            test_preds = [argmax(test_pred[:, i]) for i in 1:size(test_pred, 2)]
            test_acc = mean(test_preds .== y_test)

            improved = ""
            if test_acc > best_acc
                best_acc = test_acc
                best_ep = epoch
                improved = " *"
            end

            @printf("  Ep %2d | TrAcc %5.1f%% | TeAcc %5.1f%% | LR %.1e%s\n",
                    epoch, train_acc*100, test_acc*100, lr, improved)
        end
    end

    elapsed = Dates.value(now() - t0) / 1000.0

    net.training = false
    final_pred = forward_snn_v3!(net, X_test)
    final_preds = [argmax(final_pred[:, i]) for i in 1:size(final_pred, 2)]

    println("\n" * "═"^80)
    println("   DVS SNN v3 TRAINING COMPLETE")
    println("═"^80)
    @printf("  Training Time:   %.1fs\n", elapsed)
    @printf("  Best Test Acc:   %.2f%% (epoch %d)\n", best_acc * 100, best_ep)

    println("\n  Per-Class Accuracy:")
    class_names = ["Clap", "R-Wave", "L-Wave", "R-CW", "R-CCW",
                   "L-CW", "L-CCW", "Arm Roll", "Drums", "Guitar", "Other"]
    for c in 1:DVS_CLASSES
        mask = y_test .== c
        n_c = sum(mask)
        if n_c > 0
            correct_c = sum(final_preds[mask] .== c)
            @printf("    %2d %-10s: %d/%d = %5.1f%%\n", c, class_names[c], correct_c, n_c, correct_c/n_c*100)
        end
    end

    _, alpha = apply_temporal_attention(net.attn, net.lif3.S)
    println("\n  Temporal Attention Weights:")
    print("    ")
    for t in 1:SNN_T_STEPS
        @printf("%.3f ", alpha[t])
    end
    println()

    mkpath(joinpath(DVS_PROJECT_DIR, "results"))
    open(joinpath(DVS_PROJECT_DIR, "results", "dvs_snn_v3_results.txt"), "w") do io
        println(io, "DVS128 Gesture SNN v3 Results -- $(now())")
        println(io, "="^60)
        @printf(io, "Best Test Accuracy: %.4f (epoch %d)\n", best_acc, best_ep)
        @printf(io, "Training Time: %.1fs\n", elapsed)
        println(io, "Downsampling: $(DVS_RES)x$(DVS_RES) -> $(R)x$(R)")
        println(io, "Architecture: $spatial_in->$PROJ_DIM->$H1->$H2->$H3->$DVS_CLASSES")
    end
    println("\n  Results saved to: results/dvs_snn_v3_results.txt")
end

run_dvs_snn_v3()
