
using DelimitedFiles, LinearAlgebra, Statistics, Random, Printf, Dates

const CORE_PROJECT_DIR = dirname(@__DIR__)


lrelu(x) = x > 0 ? x : 0.01x
lrelu_d(x) = x > 0 ? 1.0 : 0.01

function softmax_c(X)
    mx = maximum(X, dims=1)
    e = exp.(X .- mx)
    e ./ sum(e, dims=1)
end


mutable struct Dense
    W::Matrix{Float64}; b::Vector{Float64}
    dW::Matrix{Float64}; db::Vector{Float64}
    mW::Matrix{Float64}; vW::Matrix{Float64}
    mb::Vector{Float64}; vb::Vector{Float64}
    inp::Matrix{Float64}
end

Dense(ni, no) = Dense(
    randn(no, ni) .* sqrt(2.0 / ni), zeros(no),
    zeros(no, ni), zeros(no),
    zeros(no, ni), zeros(no, ni),
    zeros(no), zeros(no),
    zeros(0, 0)
)

function fwd!(l::Dense, X)
    l.inp = X
    l.W * X .+ l.b
end

function bwd!(l::Dense, d, λ, clip=1.0)
    m = size(d, 2)
    l.dW = d * l.inp' ./ m .+ λ .* l.W
    l.db = vec(sum(d, dims=2)) ./ m
    gnorm = sqrt(sum(l.dW .^ 2) + sum(l.db .^ 2))
    if gnorm > clip
        scale = clip / gnorm
        l.dW .*= scale
        l.db .*= scale
    end
    l.W' * d
end


mutable struct BN
    γ::Vector{Float64}; β::Vector{Float64}
    dγ::Vector{Float64}; dβ::Vector{Float64}
    mγ::Vector{Float64}; vγ::Vector{Float64}
    mβ::Vector{Float64}; vβ::Vector{Float64}
    rμ::Vector{Float64}; rσ²::Vector{Float64}
    xn::Matrix{Float64}; xc::Matrix{Float64}; si::Vector{Float64}
    training::Bool
end

BN(d) = BN(
    ones(d), zeros(d), zeros(d), zeros(d),
    zeros(d), zeros(d), zeros(d), zeros(d),
    zeros(d), ones(d),
    zeros(0, 0), zeros(0, 0), zeros(d),
    true
)

function fwd!(bn::BN, X)
    d, m = size(X)
    if bn.training && m > 1
        μ = vec(mean(X, dims=2))
        σ² = vec(var(X, dims=2, corrected=false)) .+ 1e-5
        bn.si = 1.0 ./ sqrt.(σ²)
        bn.xc = X .- μ
        bn.xn = bn.xc .* bn.si
        bn.rμ .= 0.9 .* bn.rμ .+ 0.1 .* μ
        bn.rσ² .= 0.9 .* bn.rσ² .+ 0.1 .* σ²
    else
        bn.xn = (X .- bn.rμ) ./ sqrt.(bn.rσ² .+ 1e-5)
    end
    bn.γ .* bn.xn .+ bn.β
end

function bwd!(bn::BN, d)
    m = size(d, 2)
    if !bn.training || size(bn.xn, 2) != m || size(bn.xc, 2) != m || size(bn.si, 1) != size(d, 1)
        # Approximate with evaluation mode derivatives when shapes don't match
        return d .* bn.γ ./ sqrt.(bn.rσ² .+ 1e-5)
    end
    bn.dγ = vec(sum(d .* bn.xn, dims=2))
    bn.dβ = vec(sum(d, dims=2))
    dxn = d .* bn.γ
    dvar = vec(sum(dxn .* bn.xc .* (-0.5 .* bn.si .^ 3), dims=2))
    dmu = vec(sum(-dxn .* bn.si, dims=2)) .+
          dvar .* vec(mean(-2.0 .* bn.xc, dims=2))
    dxn .* bn.si .+ (2.0 / m) .* dvar .* bn.xc .+ dmu ./ m
end


mutable struct Net
    layers::Vector{Dense}
    bns::Vector{BN}
    sizes::Vector{Int}
    drop::Float64
    masks::Vector{Matrix{Float64}}
    pre_act::Vector{Matrix{Float64}}
    training::Bool
end

function Net(sizes; drop=0.3)
    layers = [Dense(sizes[i], sizes[i+1]) for i in 1:length(sizes)-1]
    bns = [BN(sizes[i+1]) for i in 1:length(sizes)-2]
    Net(layers, bns, sizes, drop, Matrix{Float64}[], Matrix{Float64}[], true)
end

function set_mode!(n::Net, training::Bool)
    n.training = training
    for bn in n.bns; bn.training = training; end
end

function forward!(n::Net, X)
    empty!(n.pre_act); empty!(n.masks)
    h = X
    for i in 1:length(n.layers)-1
        z = fwd!(n.bns[i], fwd!(n.layers[i], h))
        push!(n.pre_act, z)
        h = lrelu.(z)
        if n.training && n.drop > 0
            mask = Float64.(rand(size(h)...) .> n.drop) ./ (1.0 - n.drop)
            push!(n.masks, mask); h = h .* mask
        else
            push!(n.masks, ones(size(h)))
        end
    end
    softmax_c(fwd!(n.layers[end], h))
end

function backward!(n::Net, y_onehot, y_pred; l2=1e-4, clip=1.0)
    d = y_pred .- y_onehot
    d = bwd!(n.layers[end], d, l2, clip)
    for i in (length(n.layers)-1):-1:1
        d = d .* n.masks[i] .* lrelu_d.(n.pre_act[i])
        d = bwd!(n.bns[i], d)
        d = bwd!(n.layers[i], d, l2, clip)
    end
end

function adam_step!(n::Net, lr, t; β1=0.9, β2=0.999, ε=1e-8)
    bc1 = 1.0 - β1^t; bc2 = 1.0 - β2^t
    for l in n.layers
        @. l.mW = β1 * l.mW + (1-β1) * l.dW
        @. l.vW = β2 * l.vW + (1-β2) * l.dW^2
        @. l.mb = β1 * l.mb + (1-β1) * l.db
        @. l.vb = β2 * l.vb + (1-β2) * l.db^2
        @. l.W -= lr * (l.mW/bc1) / (sqrt(l.vW/bc2) + ε)
        @. l.b -= lr * (l.mb/bc1) / (sqrt(l.vb/bc2) + ε)
    end
    for bn in n.bns
        @. bn.mγ = β1 * bn.mγ + (1-β1) * bn.dγ
        @. bn.vγ = β2 * bn.vγ + (1-β2) * bn.dγ^2
        @. bn.mβ = β1 * bn.mβ + (1-β1) * bn.dβ
        @. bn.vβ = β2 * bn.vβ + (1-β2) * bn.dβ^2
        @. bn.γ -= lr * (bn.mγ/bc1) / (sqrt(bn.vγ/bc2) + ε)
        @. bn.β -= lr * (bn.mβ/bc1) / (sqrt(bn.vβ/bc2) + ε)
    end
end

function count_params(net::Net)
    sum(length(l.W) + length(l.b) for l in net.layers) +
    sum(length(bn.γ) + length(bn.β) for bn in net.bns)
end


function onehot(y)
    oh = zeros(2, length(y))
    for (i, v) in enumerate(y); oh[v+1, i] = 1.0; end
    oh
end

xent(ŷ, yoh) = -mean(sum(yoh .* log.(clamp.(ŷ, 1e-8, 1.0)), dims=1))

function compute_auc(scores, labels)
    np = sum(labels .== 1); nn = sum(labels .== 0)
    (np == 0 || nn == 0) && return 0.5
    idx = sortperm(scores, rev=true)
    tp = 0; fp = 0; auc = 0.0; prev_fpr = 0.0
    for i in idx
        if labels[i] == 1; tp += 1
        else fp += 1; auc += (tp/np) * (fp/nn - prev_fpr); prev_fpr = fp/nn; end
    end; auc
end

function evaluate(net, X, y)
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


function select_top_features(X, y, k; verbose=true)
    nf = size(X, 1)
    cors = zeros(nf)
    yf = Float64.(y); yμ = mean(yf); yσ = std(yf)
    for i in 1:nf
        xi = vec(X[i, :]); xμ = mean(xi); xσ = std(xi)
        xσ == 0 && continue
        cors[i] = abs(mean((xi .- xμ) .* (yf .- yμ)) / (xσ * yσ))
    end
    top_idx = sortperm(cors, rev=true)[1:min(k, nf)]
    if verbose
        n_strong = sum(cors[top_idx] .> 0.1)
        println("    Selected top $k features by |correlation| with target")
        println("   Features with |corr| > 0.1: $n_strong")
        println("   Top-10 correlations: $([@sprintf("%.4f", cors[i]) for i in top_idx[1:min(10,length(top_idx))]])")
    end
    sort(top_idx)
end


function load_data(; top_k=300, verbose=true)
    if verbose
        println(" Loading datasets...")
    end

    tr = readdlm(joinpath(CORE_PROJECT_DIR, "data", "train_dataset.csv"), ',', Float64; header=true)[1]
    te = readdlm(joinpath(CORE_PROJECT_DIR, "data", "test_dataset.csv"), ',', Float64; header=true)[1]

    X_train = Matrix{Float64}(tr[:, 1:end-1]')
    y_train = Int.(tr[:, end])
    X_test = Matrix{Float64}(te[:, 1:end-1]')
    y_test = Int.(te[:, end])

    if verbose
        println("   Original: $(size(X_train, 1)) features × $(size(X_train, 2)) train / $(size(X_test, 2)) test")
    end

    sel = select_top_features(X_train, y_train, top_k; verbose=verbose)
    X_train = X_train[sel, :]; X_test = X_test[sel, :]

    μ = mean(X_train, dims=2); σ = std(X_train, dims=2); σ[σ .== 0] .= 1.0
    X_train = (X_train .- μ) ./ σ; X_test = (X_test .- μ) ./ σ

    if verbose
        println("   Reduced to $(length(sel)) features")
        println("   Classes: 0→$(sum(y_train .== 0)) | 1→$(sum(y_train .== 1))")
    end

    X_train, y_train, X_test, y_test
end


function quick_train!(net, X_train, y_train, X_test, y_test;
                      epochs=20, lr=5e-4, batch_size=32, l2=1e-4, clip=1.0)
    n = size(X_train, 2)
    step = 0
    best_acc = 0.0

    for epoch in 1:epochs
        current_lr = lr * 0.5 * (1 + cos(π * epoch / epochs))
        set_mode!(net, true)
        perm = randperm(n)
        for start in 1:batch_size:n
            step += 1
            idx = perm[start:min(start+batch_size-1, n)]
            ŷ = forward!(net, X_train[:, idx])
            backward!(net, onehot(y_train[idx]), ŷ; l2=l2, clip=clip)
            adam_step!(net, current_lr, step)
        end
        m = evaluate(net, X_test, y_test)
        best_acc = max(best_acc, m.acc)
    end

    m = evaluate(net, X_test, y_test)
    return (best_acc=best_acc, final_acc=m.acc, final_f1=m.f1, final_auc=m.auc, final_loss=m.loss)
end
