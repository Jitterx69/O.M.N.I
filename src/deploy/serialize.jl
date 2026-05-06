using Printf

const OMNI_MAGIC = UInt8[0x4F, 0x4D, 0x4E, 0x49]
const OMNI_VERSION = UInt32(1)

const LAYER_DENSE    = UInt8(0x01)
const LAYER_BN       = UInt8(0x02)
const LAYER_KAN      = UInt8(0x03)
const LAYER_PLASTIC  = UInt8(0x04)

struct ModelHeader
    magic::Vector{UInt8}
    version::UInt32
    model_type::UInt8
    n_layers::UInt32
    input_dim::UInt32
    output_dim::UInt32
end

function write_matrix(io::IO, M::Matrix{Float64})
    rows, cols = size(M)
    write(io, UInt32(rows))
    write(io, UInt32(cols))
    write(io, M)
end

function write_vector(io::IO, v::Vector{Float64})
    write(io, UInt32(length(v)))
    write(io, v)
end

function read_matrix(io::IO)
    rows = read(io, UInt32)
    cols = read(io, UInt32)
    M = zeros(Float64, rows, cols)
    read!(io, M)
    return M
end

function read_vector(io::IO)
    n = read(io, UInt32)
    v = zeros(Float64, n)
    read!(io, v)
    return v
end

function serialize_net(net, path::String; model_type::UInt8=LAYER_DENSE)
    open(path, "w") do io
        write(io, OMNI_MAGIC)
        write(io, OMNI_VERSION)
        write(io, model_type)

        if model_type == LAYER_DENSE
            write(io, UInt32(length(net.layers)))
            write(io, UInt32(size(net.layers[1].W, 2)))
            write(io, UInt32(size(net.layers[end].W, 1)))

            for l in net.layers
                write(io, LAYER_DENSE)
                write_matrix(io, l.W)
                write_vector(io, l.b)
            end

            for i in 1:length(net.layers)-1
                bn = net.bns[i]
                write(io, LAYER_BN)
                write_vector(io, bn.gamma)
                write_vector(io, bn.beta)
                write_vector(io, bn.mu_run)
                write_vector(io, bn.sig_run)
            end

        elseif model_type == LAYER_KAN
            write(io, UInt32(length(net.layers)))
            write(io, UInt32(net.layers[1].in_dim))
            write(io, UInt32(net.layers[end].out_dim))

            for l in net.layers
                write(io, LAYER_KAN)
                write(io, UInt32(l.in_dim))
                write(io, UInt32(l.out_dim))
                write(io, UInt32(l.degree))
                write_matrix(io, l.W_base)
                write_matrix(io, l.C_flat)
            end

        elseif model_type == LAYER_PLASTIC
            write(io, UInt32(3))
            write(io, UInt32(size(net.encoder.W, 2)))
            write(io, UInt32(size(net.head.W, 1)))

            write(io, LAYER_DENSE)
            write_matrix(io, net.encoder.W)
            write_vector(io, net.encoder.b)

            write(io, LAYER_PLASTIC)
            write_matrix(io, net.plastic.W)
            write_vector(io, net.plastic.b)
            write_matrix(io, net.plastic.alpha)
            write_matrix(io, net.plastic.H)

            write(io, LAYER_DENSE)
            write_matrix(io, net.head.W)
            write_vector(io, net.head.b)
        end
    end

    fsize = filesize(path)
    unit = fsize > 1024*1024 ? @sprintf("%.2f MB", fsize/1024/1024) : @sprintf("%.1f KB", fsize/1024)
    @printf("  Serialized to %s (%s)\n", path, unit)
end

function deserialize_net(path::String)
    open(path, "r") do io
        magic = Vector{UInt8}(undef, 4)
        read!(io, magic)
        magic != OMNI_MAGIC && error("Invalid .omni file: bad magic bytes")

        version = read(io, UInt32)
        model_type = read(io, UInt8)
        n_layers = read(io, UInt32)
        in_dim = read(io, UInt32)
        out_dim = read(io, UInt32)

        if model_type == LAYER_DENSE
            layers = Dense[]
            for _ in 1:n_layers
                _ = read(io, UInt8)
                W = read_matrix(io)
                b = read_vector(io)
                l = Dense(size(W, 2), size(W, 1))
                l.W .= W
                l.b .= b
                push!(layers, l)
            end

            bns = BN[]
            for _ in 1:n_layers-1
                _ = read(io, UInt8)
                gamma = read_vector(io)
                beta = read_vector(io)
                mu_run = read_vector(io)
                sig_run = read_vector(io)
                bn = BN(length(gamma))
                bn.gamma .= gamma
                bn.beta .= beta
                bn.mu_run .= mu_run
                bn.sig_run .= sig_run
                push!(bns, bn)
            end

            net = Net(vcat([in_dim], [size(l.W, 1) for l in layers]); drop=0.0)
            for (i, l) in enumerate(layers)
                net.layers[i].W .= l.W
                net.layers[i].b .= l.b
            end
            for (i, bn) in enumerate(bns)
                net.bns[i].gamma .= bn.gamma
                net.bns[i].beta .= bn.beta
                net.bns[i].mu_run .= bn.mu_run
                net.bns[i].sig_run .= bn.sig_run
            end
            set_mode!(net, false)
            return net, :mlp

        elseif model_type == LAYER_KAN
            layers = ChebKANLayer[]
            for _ in 1:n_layers
                _ = read(io, UInt8)
                l_in = read(io, UInt32)
                l_out = read(io, UInt32)
                deg = read(io, UInt32)
                W_base = read_matrix(io)
                C_flat = read_matrix(io)
                l = ChebKANLayer(Int(l_in), Int(l_out), Int(deg))
                l.W_base .= W_base
                l.C_flat .= C_flat
                push!(layers, l)
            end
            net = KANNet(layers, false)
            return net, :kan

        elseif model_type == LAYER_PLASTIC
            _ = read(io, UInt8)
            enc_W = read_matrix(io)
            enc_b = read_vector(io)

            _ = read(io, UInt8)
            pla_W = read_matrix(io)
            pla_b = read_vector(io)
            pla_alpha = read_matrix(io)
            pla_H = read_matrix(io)

            _ = read(io, UInt8)
            head_W = read_matrix(io)
            head_b = read_vector(io)

            net = PlasticNet(Int(in_dim), size(pla_W, 1), Int(out_dim))
            net.encoder.W .= enc_W
            net.encoder.b .= enc_b
            net.plastic.W .= pla_W
            net.plastic.b .= pla_b
            net.plastic.alpha .= pla_alpha
            net.plastic.H .= pla_H
            net.head.W .= head_W
            net.head.b .= head_b
            set_mode!(net, false)
            return net, :hebbian
        end
    end
end
