mutable struct CAJILayer
    n_jets::Int
    jet_dim::Int
    d_k::Int
    Wq::Matrix{Float64}; Wk::Matrix{Float64}; Wv::Matrix{Float64}
    Wo::Matrix{Float64}; bo::Vector{Float64}
    dWq::Matrix{Float64}; dWk::Matrix{Float64}; dWv::Matrix{Float64}
    dWo::Matrix{Float64}; dbo::Vector{Float64}
    mWq::Matrix{Float64}; vWq::Matrix{Float64}
    mWk::Matrix{Float64}; vWk::Matrix{Float64}
    mWv::Matrix{Float64}; vWv::Matrix{Float64}
    mWo::Matrix{Float64}; vWo::Matrix{Float64}
    mbo::Vector{Float64}; vbo::Vector{Float64}
    Q::Matrix{Float64}; K::Matrix{Float64}; V::Matrix{Float64}
    attn::Matrix{Float64}
    jets_inp::Matrix{Float64}
    ctx::Matrix{Float64}
end

function CAJILayer(n_jets::Int, jet_dim::Int, d_k::Int)
    total_in = n_jets * jet_dim
    out_dim = n_jets * d_k
    sc = sqrt(2.0 / jet_dim)
    CAJILayer(
        n_jets, jet_dim, d_k,
        randn(d_k, jet_dim) .* sc, randn(d_k, jet_dim) .* sc, randn(d_k, jet_dim) .* sc,
        randn(out_dim, out_dim) .* sqrt(2.0 / out_dim), zeros(out_dim),
        zeros(d_k, jet_dim), zeros(d_k, jet_dim), zeros(d_k, jet_dim),
        zeros(out_dim, out_dim), zeros(out_dim),
        zeros(d_k, jet_dim), zeros(d_k, jet_dim),
        zeros(d_k, jet_dim), zeros(d_k, jet_dim),
        zeros(d_k, jet_dim), zeros(d_k, jet_dim),
        zeros(out_dim, out_dim), zeros(out_dim, out_dim),
        zeros(out_dim), zeros(out_dim),
        zeros(0,0), zeros(0,0), zeros(0,0),
        zeros(0,0), zeros(0,0), zeros(0,0))
end

function extract_jets(X::Matrix{Float64})
    jet1 = X[6:9, :]
    jet2 = X[10:13, :]
    jet3 = X[14:17, :]
    jet4 = X[18:21, :]
    return vcat(jet1, jet2, jet3, jet4)
end

function fwd_caji!(l::CAJILayer, X::Matrix{Float64})
    B = size(X, 2)
    jets = extract_jets(X)
    l.jets_inp = jets

    n = l.n_jets
    d = l.jet_dim
    dk = l.d_k

    Q_all = zeros(dk, n, B)
    K_all = zeros(dk, n, B)
    V_all = zeros(dk, n, B)

    for j in 1:n
        s = (j-1)*d + 1
        e = j*d
        jet_slice = jets[s:e, :]
        Q_all[:, j, :] = l.Wq * jet_slice
        K_all[:, j, :] = l.Wk * jet_slice
        V_all[:, j, :] = l.Wv * jet_slice
    end

    out = zeros(n * dk, B)
    l.attn = zeros(n * n, B)

    for b in 1:B
        Q_b = Q_all[:, :, b]
        K_b = K_all[:, :, b]
        V_b = V_all[:, :, b]

        scores = Q_b' * K_b ./ sqrt(Float64(dk))
        mx = maximum(scores, dims=2)
        ex = exp.(scores .- mx)
        A = ex ./ sum(ex, dims=2)

        l.attn[:, b] = vec(A)

        ctx = A * V_b'
        out[:, b] = vec(ctx')
    end

    l.Q = reshape(Q_all, dk * n, B)
    l.K = reshape(K_all, dk * n, B)
    l.V = reshape(V_all, dk * n, B)
    l.ctx = out

    result = l.Wo * out .+ l.bo
    return result
end

function bwd_caji!(l::CAJILayer, d_out::Matrix{Float64}, lambda::Float64, clip::Float64=1.0)
    B = size(d_out, 2)
    n = l.n_jets
    d = l.jet_dim
    dk = l.d_k

    l.dWo = d_out * l.ctx' ./ B .+ lambda .* l.Wo
    l.dbo = vec(sum(d_out, dims=2)) ./ B

    d_ctx = l.Wo' * d_out

    l.dWq .= 0.0; l.dWk .= 0.0; l.dWv .= 0.0
    d_jets = zeros(size(l.jets_inp))

    for b in 1:B
        A = reshape(l.attn[:, b], n, n)

        d_ctx_b = reshape(d_ctx[:, b], dk, n)

        d_V_b = A' * d_ctx_b'
        d_A = d_ctx_b * reshape(l.V[1:dk*n, b:b], dk, n)'
        d_A = d_A' .* A .* (1.0 .- A)

        Q_b = reshape(l.Q[:, b], dk, n)
        K_b = reshape(l.K[:, b], dk, n)

        d_Q_b = K_b * d_A' ./ sqrt(Float64(dk))
        d_K_b = Q_b * d_A ./ sqrt(Float64(dk))

        for j in 1:n
            s = (j-1)*d + 1
            e = j*d
            jet_j = l.jets_inp[s:e, b:b]
            l.dWq .+= d_Q_b[:, j:j] * jet_j' ./ B
            l.dWk .+= d_K_b[:, j:j] * jet_j' ./ B
            l.dWv .+= d_V_b[j:j, :]' * jet_j' ./ B
            d_jets[s:e, b] .+= vec(l.Wq' * d_Q_b[:, j:j] .+
                                    l.Wk' * d_K_b[:, j:j] .+
                                    l.Wv' * d_V_b[j:j, :]')
        end
    end

    l.dWq .+= lambda .* l.Wq
    l.dWk .+= lambda .* l.Wk
    l.dWv .+= lambda .* l.Wv

    gnorm = sqrt(sum(l.dWq.^2) + sum(l.dWk.^2) + sum(l.dWv.^2) +
                 sum(l.dWo.^2) + sum(l.dbo.^2))
    if gnorm > clip
        s = clip / gnorm
        l.dWq .*= s; l.dWk .*= s; l.dWv .*= s
        l.dWo .*= s; l.dbo .*= s
    end

    d_X = zeros(size(d_out, 1) > 0 ? size(d_jets, 1) : 0, B)
    return d_jets
end

function adam_caji!(l::CAJILayer, lr, t; b1=0.9, b2=0.999, eps=1e-8)
    bc1 = 1.0 - b1^t; bc2 = 1.0 - b2^t

    function upd!(mW, vW, W, dW)
        @. mW = b1*mW + (1-b1)*dW
        @. vW = b2*vW + (1-b2)*dW^2
        @. W -= lr*(mW/bc1)/(sqrt(vW/bc2)+eps)
    end

    upd!(l.mWq, l.vWq, l.Wq, l.dWq)
    upd!(l.mWk, l.vWk, l.Wk, l.dWk)
    upd!(l.mWv, l.vWv, l.Wv, l.dWv)
    upd!(l.mWo, l.vWo, l.Wo, l.dWo)
    @. l.mbo = b1*l.mbo + (1-b1)*l.dbo
    @. l.vbo = b2*l.vbo + (1-b2)*l.dbo^2
    @. l.bo -= lr*(l.mbo/bc1)/(sqrt(l.vbo/bc2)+eps)
end

function count_caji_params(l::CAJILayer)
    length(l.Wq) + length(l.Wk) + length(l.Wv) + length(l.Wo) + length(l.bo)
end

function compute_pairwise_physics(X::Matrix{Float64})
    B = size(X, 2)
    n_pairs = 6
    pair_feats = zeros(n_pairs * 3, B)

    jet_indices = [(6,9), (10,13), (14,17), (18,21)]
    pair_idx = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]

    for (p, (i, j)) in enumerate(pair_idx)
        si, ei = jet_indices[i]
        sj, ej = jet_indices[j]

        pt_i = X[si, :]; eta_i = X[si+1, :]; phi_i = X[si+2, :]
        pt_j = X[sj, :]; eta_j = X[sj+1, :]; phi_j = X[sj+2, :]

        deta = eta_i .- eta_j
        dphi = phi_i .- phi_j
        dr = sqrt.(deta.^2 .+ dphi.^2)

        m_inv = sqrt.(abs.(2.0 .* pt_i .* pt_j .* (cosh.(deta) .- cos.(dphi))))

        pt_ratio = pt_i ./ (pt_j .+ 1e-8)

        off = (p-1)*3
        pair_feats[off+1, :] = dr
        pair_feats[off+2, :] = m_inv
        pair_feats[off+3, :] = log.(pt_ratio .+ 1e-8)
    end
    return pair_feats
end
