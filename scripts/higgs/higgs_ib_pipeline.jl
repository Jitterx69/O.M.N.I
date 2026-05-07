include("../../src/higgs_loader.jl")

# Minimal redundant inclusion of Stochastic Bottleneck logic for the Higgs scale
mutable struct StochasticBottleneck
    h_dim::Int
    z_dim::Int
    W_mu::Matrix{Float64}; b_mu::Vector{Float64}
    W_lv::Matrix{Float64}; b_lv::Vector{Float64}
    dW_mu::Matrix{Float64}; db_mu::Vector{Float64}
    dW_lv::Matrix{Float64}; db_lv::Vector{Float64}
    mW_mu::Matrix{Float64}; vW_mu::Matrix{Float64}
    mb_mu::Vector{Float64}; vb_mu::Vector{Float64}
    mW_lv::Matrix{Float64}; vW_lv::Matrix{Float64}
    mb_lv::Vector{Float64}; vb_lv::Vector{Float64}
    inp::Matrix{Float64}; mu::Matrix{Float64}; lv::Matrix{Float64}
    eps::Matrix{Float64}; z::Matrix{Float64}
end

function StochasticBottleneck(h_dim::Int, z_dim::Int)
    StochasticBottleneck(
        h_dim, z_dim,
        randn(z_dim, h_dim) .* sqrt(2.0/h_dim), zeros(z_dim),
        randn(z_dim, h_dim) .* 0.01, zeros(z_dim),
        zeros(z_dim, h_dim), zeros(z_dim), zeros(z_dim, h_dim), zeros(z_dim),
        zeros(z_dim, h_dim), zeros(z_dim, h_dim), zeros(z_dim), zeros(z_dim),
        zeros(z_dim, h_dim), zeros(z_dim, h_dim), zeros(z_dim), zeros(z_dim),
        zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0), zeros(0,0)
    )
end

function fwd_ib!(sb::StochasticBottleneck, h; training=true)
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

function bwd_ib!(sb::StochasticBottleneck, dz, β, λ, clip=1.0)
    B = size(dz, 2)
    dmu = dz .+ β .* sb.mu
    sigma = exp.(0.5 .* sb.lv)
    dlv = dz .* 0.5 .* sigma .* sb.eps .+ β .* 0.5 .* (exp.(sb.lv) .- 1.0)
    
    sb.dW_mu = dmu * sb.inp' ./ B .+ λ .* sb.W_mu
    sb.db_mu = vec(sum(dmu, dims=2)) ./ B
    sb.dW_lv = dlv * sb.inp' ./ B .+ λ .* sb.W_lv
    sb.db_lv = vec(sum(dlv, dims=2)) ./ B
    
    gnorm = sqrt(sum(sb.dW_mu.^2) + sum(sb.db_mu.^2) + sum(sb.dW_lv.^2) + sum(sb.db_lv.^2))
    if gnorm > clip
        s = clip / gnorm
        sb.dW_mu .*= s; sb.db_mu .*= s; sb.dW_lv .*= s; sb.db_lv .*= s
    end
    return sb.W_mu' * dmu .+ sb.W_lv' * dlv
end

mutable struct Dense
    W::Matrix{Float64}; b::Vector{Float64}
    dW::Matrix{Float64}; db::Vector{Float64}
    mW::Matrix{Float64}; vW::Matrix{Float64}
    mb::Vector{Float64}; vb::Vector{Float64}
    inp::Matrix{Float64}
end

function Dense(ni, no)
    Dense(
        randn(no, ni) .* sqrt(2.0/ni), zeros(no),
        zeros(no, ni), zeros(no),
        zeros(no, ni), zeros(no, ni),
        zeros(no), zeros(no),
        zeros(0,0)
    )
end

function fwd_d!(l::Dense, X)
    l.inp = X
    l.W * X .+ l.b
end

function bwd_d!(l::Dense, d, λ, clip=1.0)
    m = size(d, 2)
    l.dW = d * l.inp' ./ m .+ λ .* l.W
    l.db = vec(sum(d, dims=2)) ./ m
    gnorm = sqrt(sum(l.dW.^2) + sum(l.db.^2))
    if gnorm > clip
        s = clip / gnorm
        l.dW .*= s; l.db .*= s
    end
    l.W' * d
end

function softmax(X)
    mx = maximum(X, dims=1)
    e = exp.(X .- mx)
    e ./ sum(e, dims=1)
end

function xent(y_pred, y_oh)
    -mean(sum(y_oh .* log.(clamp.(y_pred, 1e-8, 1.0)), dims=1))
end

lrelu(x) = x > 0 ? x : 0.01x
lrelu_d(x) = x > 0 ? 1.0 : 0.01

function run_higgs_ib()
    println("\n" * "═"^80)
    println("   HIGGS BOSON DISCOVERY PIPELINE — Variational Information Bottleneck")
    println("   Data Scale: 11,000,000 samples | Feature Count: 28")
    println("═"^80)

    higgs_path = joinpath(HIGGS_PROJECT_DIR, "data", "higgs", "HIGGS.csv")
    ds = index_higgs(higgs_path)
    
    println("\n  Dataset Stats:")
    println("    Total Samples: $(ds.n_samples)")
    @printf("    Signal (Higgs): %d (%.1f%%)\n", ds.signal_count, ds.signal_count/ds.n_samples*100)
    @printf("    Weights: Signal=%.3f, Background=%.3f\n", ds.class_weights[2], ds.class_weights[1])

    # Split for validation (last 500k rows)
    train_indices = collect(1:(ds.n_samples - 500000))
    val_indices = collect((ds.n_samples - 500000 + 1):ds.n_samples)
    
    H_DIM = 128
    Z_DIM = 16
    enc = Dense(HIGGS_FEATURES, H_DIM)
    ib = StochasticBottleneck(H_DIM, Z_DIM)
    cls = Dense(Z_DIM, 2)
    
    println("\n Architecture: Dense(28 -> $H_DIM) -> IB($H_DIM -> $Z_DIM) -> Dense($Z_DIM -> 2)")
    println("  " * "-"^70)

    batch_size = 256
    epochs = 5
    step = 0
    t0 = now()
    
    for epoch in 1:epochs
        lr = 1e-3 * 0.5 * (1 + cos(pi * epoch / epochs))
        β = 1e-4 * (epoch / epochs) # Annealing beta
        
        perm = randperm(length(train_indices))
        epoch_loss = 0.0
        n_batches = 0
        
        for start in 1:batch_size:length(train_indices)
            step += 1
            idx = train_indices[perm[start:min(start+batch_size-1, length(train_indices))]]
            
            X, y = load_higgs_batch(ds, idx)
            y_oh = higgs_onehot(y)
            
            # Fwd
            h = lrelu.(fwd_d!(enc, X))
            z = fwd_ib!(ib, h)
            y_pred = softmax(fwd_d!(cls, z))
            
            # Bwd
            d = y_pred .- y_oh
            dz = bwd_d!(cls, d, 1e-4)
            dh = bwd_ib!(ib, dz, β, 1e-4)
            dh = dh .* lrelu_d.(enc.W * enc.inp .+ enc.b)
            bwd_d!(enc, dh, 1e-4)
            
            # Optimizer (Adam simplified)
            function adam_upd!(l, lr, t)
                bc1, bc2 = 1.0 - 0.9^t, 1.0 - 0.999^t
                @. l.mW = 0.9 * l.mW + 0.1 * l.dW
                @. l.vW = 0.999 * l.vW + 0.001 * l.dW^2
                @. l.W -= lr * (l.mW/bc1) / (sqrt(l.vW/bc2) + 1e-8)
                @. l.mb = 0.9 * l.mb + 0.1 * l.db
                @. l.vb = 0.999 * l.vb + 0.001 * l.db^2
                @. l.b -= lr * (l.mb/bc1) / (sqrt(l.vb/bc2) + 1e-8)
            end
            
            adam_upd!(enc, lr, step)
            adam_upd!(cls, lr, step)
            # IB Adam
            bc1, bc2 = 1.0-0.9^step, 1.0-0.999^step
            @. ib.mW_mu = 0.9*ib.mW_mu + 0.1*ib.dW_mu; @. ib.vW_mu = 0.999*ib.vW_mu + 0.001*ib.dW_mu^2
            @. ib.W_mu -= lr*(ib.mW_mu/bc1)/(sqrt(ib.vW_mu/bc2)+1e-8)
            @. ib.mb_mu = 0.9*ib.mb_mu + 0.1*ib.db_mu; @. ib.vb_mu = 0.999*ib.vb_mu + 0.001*ib.db_mu^2
            @. ib.b_mu -= lr*(ib.mb_mu/bc1)/(sqrt(ib.vb_mu/bc2)+1e-8)
            @. ib.mW_lv = 0.9*ib.mW_lv + 0.1*ib.dW_lv; @. ib.vW_lv = 0.999*ib.vW_lv + 0.001*ib.dW_lv^2
            @. ib.W_lv -= lr*(ib.mW_lv/bc1)/(sqrt(ib.vW_lv/bc2)+1e-8)
            @. ib.mb_lv = 0.9*ib.mb_lv + 0.1*ib.db_lv; @. ib.vb_lv = 0.999*ib.vb_lv + 0.001*ib.db_lv^2
            @. ib.b_lv -= lr*(ib.mb_lv/bc1)/(sqrt(ib.vb_lv/bc2)+1e-8)

            epoch_loss += xent(y_pred, y_oh)
            n_batches += 1
            
            if step % 2000 == 0
                @printf("  Batch %7d | Loss: %.4f | Beta: %.1e\n", step, epoch_loss/n_batches, β)
            end
        end
        
        # Validation
        X_val, y_val = load_higgs_batch(ds, val_indices[1:10000]) # Sample val for speed
        h_val = lrelu.(fwd_d!(enc, X_val))
        z_val = fwd_ib!(ib, h_val, training=false)
        y_pred_val = softmax(fwd_d!(cls, z_val))
        m = higgs_metrics(y_pred_val, y_val)
        
        @printf("\n  Epoch %d Complete | Val Acc: %.2f%% | Val F1: %.4f\n\n", epoch, m.acc*100, m.f1)
    end
    
    elapsed = Dates.value(now() - t0) / 1000.0
    println("-"^78)
    @printf("  HIGGS DISCOVERY COMPLETE | Time: %.1fs\n", elapsed)
    println("-"^78)
end

run_higgs_ib()
