using Printf, Dates

include("../src/core.jl")
include("../src/experimental/kan.jl")
include("../src/experimental/hebbian.jl")
include("../src/deploy/serialize.jl")

function inference_loop(model_path::String)
    net, species = deserialize_model(model_path)
    @printf("  Loaded %s model from %s\n", uppercase(string(species)), model_path)

    println("  Inference server ready. Send input vectors (space-separated floats).")
    println("  Type 'quit' to exit, 'bench N' to run N-sample benchmark.")

    while !eof(stdin)
        line = readline(stdin)
        isempty(line) && continue
        strip(line) == "quit" && break

        if startswith(line, "bench")
            parts = split(line)
            n = length(parts) > 1 ? parse(Int, parts[2]) : 1000
            run_benchmark(net, species, n)
            continue
        end

        vals = tryparse.(Float64, split(line))
        if any(isnothing, vals)
            println("  ERROR: Invalid input. Expected space-separated floats.")
            continue
        end

        x = Float64.(vals)
        X = reshape(x, length(x), 1)

        t0 = time_ns()
        if species == :mlp
            set_mode!(net, false)
            yp = forward!(net, X)
        elseif species == :kan
            set_mode!(net, false)
            yp = forward!(net, X)
        elseif species == :hebbian
            set_mode!(net, false)
            yp = forward!(net, X)
        end
        dt = (time_ns() - t0) / 1000.0

        pred = argmax(yp[:, 1]) - 1
        conf = maximum(yp[:, 1])
        @printf("  Prediction: %d | Confidence: %.4f | Latency: %.1f us\n", pred, conf, dt)
    end
end

function run_benchmark(net, species, n::Int)
    in_dim = if species == :mlp
        size(net.layers[1].W, 2)
    elseif species == :kan
        net.layers[1].in_dim
    elseif species == :hebbian
        size(net.encoder.W, 2)
    end

    X_rand = randn(in_dim, n)
    latencies = zeros(n)

    for i in 1:n
        Xi = X_rand[:, i:i]
        t0 = time_ns()
        if species == :mlp
            forward!(net, Xi)
        elseif species == :kan
            forward!(net, Xi)
        elseif species == :hebbian
            forward!(net, Xi)
        end
        latencies[i] = (time_ns() - t0) / 1000.0
    end

    sort!(latencies)
    p50 = latencies[round(Int, n * 0.50)]
    p95 = latencies[round(Int, n * 0.95)]
    p99 = latencies[round(Int, n * 0.99)]
    mn = mean(latencies)

    println("\n  ┌────────────────────────────────────────────┐")
    println("  │        INFERENCE BENCHMARK RESULTS         │")
    println("  ├────────────────────────────────────────────┤")
    @printf("  │  Samples:    %-28d │\n", n)
    @printf("  │  Mean:       %-24.1f us  │\n", mn)
    @printf("  │  P50:        %-24.1f us  │\n", p50)
    @printf("  │  P95:        %-24.1f us  │\n", p95)
    @printf("  │  P99:        %-24.1f us  │\n", p99)
    @printf("  │  Throughput:  %-20.0f inf/sec  │\n", 1e6 / mn)
    println("  └────────────────────────────────────────────┘")
end

if length(ARGS) >= 1
    inference_loop(ARGS[1])
else
    println("  Usage: julia src/deploy/inference_server.jl <model.omni>")
end
