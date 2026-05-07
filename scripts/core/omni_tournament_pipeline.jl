using Statistics, Random, Printf, Dates

include("../../src/core.jl")
include("../../src/experimental/kan.jl")
include("../../src/experimental/hebbian.jl")
include("../../src/experimental/info_bottleneck.jl")

const TOURNEY_POP = 25
const TOURNEY_GENS = 8
const TOURNEY_ELITE = 3
const TOURNEY_K = 3
const TOURNEY_MUT = 0.35
const TOURNEY_CROSS = 0.7
const TOURNEY_BUDGET = 25

const SPECIES_POOL = [:mlp, :kan, :hebbian]

mutable struct AgentGenome
    species::Symbol
    hidden_widths::Vector{Int}
    dropout::Float64
    lr::Float64
    l2::Float64
    kan_degree::Int
    trace_decay::Float64

    fitness::Float64
    acc::Float64
    f1::Float64
    auc::Float64
    generation::Int
    params::Int
end

function random_agent(gen::Int)
    sp = rand(SPECIES_POOL)
    n_layers = rand(1:3)
    base_w = rand(32:256)
    widths = [clamp(round(Int, base_w * (1 - 0.3*(i-1)/max(n_layers-1,1)) + randn()*15), 16, 512) for i in 1:n_layers]
    sort!(widths, rev=true)

    dr = clamp(rand() * 0.5 + 0.05, 0.05, 0.55)
    lr = exp(log(1e-5) + rand() * (log(5e-3) - log(1e-5)))
    l2 = exp(log(1e-6) + rand() * (log(1e-2) - log(1e-6)))
    deg = rand(2:5)
    td = clamp(0.9 + randn()*0.03, 0.85, 0.99)

    AgentGenome(sp, widths, dr, lr, l2, deg, td,
                0.0, 0.0, 0.0, 0.0, gen, 0)
end

function agent_string(g::AgentGenome)
    arch = join(g.hidden_widths, "->")
    tag = uppercase(string(g.species))
    @sprintf("[%s] %s dr=%.2f lr=%.1e", tag, arch, g.dropout, g.lr)
end

function build_and_evaluate!(g::AgentGenome, X_tr, y_tr, X_te, y_te, in_dim)
    try
        if g.species == :mlp
            sizes = vcat([in_dim], g.hidden_widths, [2])
            net = Net(sizes; drop=g.dropout)
            g.params = count_params(net)
            res = quick_train!(net, X_tr, y_tr, X_te, y_te;
                               epochs=TOURNEY_BUDGET, lr=g.lr, batch_size=32, l2=g.l2, clip=1.0)
            g.fitness = 0.6 * res.best_acc + 0.4 * res.final_f1
            g.acc = res.final_acc
            g.f1 = res.final_f1
            g.auc = res.final_auc

        elseif g.species == :kan
            sizes = vcat([in_dim], g.hidden_widths[1:1], [2])
            net = KANNet(sizes, g.kan_degree)
            g.params = count_params(net)

            n = size(X_tr, 2)
            step = 0
            for ep in 1:TOURNEY_BUDGET
                lr_t = g.lr * 0.5 * (1 + cos(pi * ep / TOURNEY_BUDGET))
                perm = randperm(n)
                for s in 1:32:n
                    step += 1
                    idx = perm[s:min(s+31, n)]
                    yp = forward!(net, X_tr[:, idx])
                    backward!(net, onehot(y_tr[idx]), yp; l2=g.l2, clip=1.0)
                    adam_step!(net, lr_t, step)
                end
            end

            set_mode!(net, false)
            yp = forward!(net, X_te)
            preds = [argmax(yp[:, i])-1 for i in 1:size(yp, 2)]
            g.acc = mean(preds .== y_te)
            tp = sum((preds .== 1) .& (y_te .== 1))
            fp = sum((preds .== 1) .& (y_te .== 0))
            fn = sum((preds .== 0) .& (y_te .== 1))
            prec = tp / max(tp+fp, 1)
            rec = tp / max(tp+fn, 1)
            g.f1 = 2*prec*rec / max(prec+rec, 1e-8)
            g.auc = g.acc
            g.fitness = 0.6 * g.acc + 0.4 * g.f1

        elseif g.species == :hebbian
            h = length(g.hidden_widths) > 0 ? g.hidden_widths[1] : 64
            pnet = PlasticNet(in_dim, h, 2)
            g.params = count_params(pnet)

            n = size(X_tr, 2)
            step = 0
            for ep in 1:TOURNEY_BUDGET
                lr_t = g.lr * 0.5 * (1 + cos(pi * ep / TOURNEY_BUDGET))
                reset_traces!(pnet)
                set_mode!(pnet, true)
                perm = randperm(n)
                for s in 1:32:n
                    step += 1
                    idx = perm[s:min(s+31, n)]
                    yp = forward!(pnet, X_tr[:, idx])
                    backward!(pnet, onehot(y_tr[idx]), yp; l2=g.l2, clip=1.0)
                    adam_step!(pnet, lr_t, step)
                end
            end

            m = evaluate(pnet, X_te, y_te)
            g.acc = m.acc
            g.f1 = m.f1
            g.auc = m.auc
            g.fitness = 0.6 * g.acc + 0.4 * g.f1
        end
    catch e
        g.fitness = 0.0
        g.acc = 0.0
        g.f1 = 0.0
        g.auc = 0.0
    end
    return g.fitness
end

function tourney_select(pop::Vector{AgentGenome})
    cands = rand(pop, TOURNEY_K)
    deepcopy(cands[argmax([c.fitness for c in cands])])
end

function agent_crossover(p1::AgentGenome, p2::AgentGenome, gen::Int)
    if p1.species != p2.species
        child = deepcopy(rand() < 0.5 ? p1 : p2)
        child.generation = gen
        child.fitness = 0.0
        return child
    end

    if rand() > TOURNEY_CROSS
        child = deepcopy(rand() < 0.5 ? p1 : p2)
        child.generation = gen
        child.fitness = 0.0
        return child
    end

    max_len = max(length(p1.hidden_widths), length(p2.hidden_widths))
    cw = Int[]
    for i in 1:max_len
        if i <= length(p1.hidden_widths) && i <= length(p2.hidden_widths)
            push!(cw, rand() < 0.5 ? p1.hidden_widths[i] : p2.hidden_widths[i])
        elseif i <= length(p1.hidden_widths)
            rand() < 0.5 && push!(cw, p1.hidden_widths[i])
        else
            rand() < 0.5 && push!(cw, p2.hidden_widths[i])
        end
    end
    isempty(cw) && push!(cw, rand(32:128))
    sort!(cw, rev=true)

    dr = rand() < 0.5 ? p1.dropout : p2.dropout
    lr = rand() < 0.5 ? p1.lr : p2.lr
    l2 = rand() < 0.5 ? p1.l2 : p2.l2
    deg = rand() < 0.5 ? p1.kan_degree : p2.kan_degree
    td = rand() < 0.5 ? p1.trace_decay : p2.trace_decay

    AgentGenome(p1.species, cw, dr, lr, l2, deg, td,
                0.0, 0.0, 0.0, 0.0, gen, 0)
end

function agent_mutate!(g::AgentGenome)
    if rand() < TOURNEY_MUT
        for i in eachindex(g.hidden_widths)
            if rand() < 0.4
                g.hidden_widths[i] = clamp(round(Int, g.hidden_widths[i] * (1 + randn()*0.3)), 16, 512)
            end
        end
        sort!(g.hidden_widths, rev=true)
    end

    if rand() < TOURNEY_MUT * 0.4
        if rand() < 0.5 && length(g.hidden_widths) < 4
            push!(g.hidden_widths, clamp(round(Int, minimum(g.hidden_widths) * rand(0.5:0.1:1.0)), 16, 256))
            sort!(g.hidden_widths, rev=true)
        elseif length(g.hidden_widths) > 1
            deleteat!(g.hidden_widths, argmin(g.hidden_widths))
        end
    end

    rand() < TOURNEY_MUT && (g.dropout = clamp(g.dropout + randn()*0.08, 0.05, 0.55))
    rand() < TOURNEY_MUT && (g.lr = clamp(g.lr * exp(randn()*0.4), 1e-5, 5e-3))
    rand() < TOURNEY_MUT && (g.l2 = clamp(g.l2 * exp(randn()*0.4), 1e-6, 1e-2))
    rand() < TOURNEY_MUT && (g.kan_degree = clamp(g.kan_degree + rand(-1:1), 2, 6))
    rand() < TOURNEY_MUT && (g.trace_decay = clamp(g.trace_decay + randn()*0.02, 0.85, 0.99))

    g.fitness = 0.0
end

function species_census(pop::Vector{AgentGenome})
    counts = Dict{Symbol, Int}()
    fits = Dict{Symbol, Vector{Float64}}()
    for g in pop
        counts[g.species] = get(counts, g.species, 0) + 1
        if !haskey(fits, g.species); fits[g.species] = Float64[]; end
        push!(fits[g.species], g.fitness)
    end
    return counts, fits
end

function run_tournament()
    println("\n" * "█"^80)
    println("   O.M.N.I. MULTI-AGENT TOURNAMENT")
    println("   Species: MLP | KAN | HEBBIAN")
    println("   Survival of the Fittest Architecture")
    println("█"^80)

    X_tr, y_tr, X_te, y_te = load_data(; top_k=300, verbose=false)
    in_dim = size(X_tr, 1)

    println("\n  Configuration:")
    @printf("    Population:  %d agents\n", TOURNEY_POP)
    @printf("    Generations: %d\n", TOURNEY_GENS)
    @printf("    Species:     %s\n", join(string.(SPECIES_POOL), ", "))
    @printf("    Budget:      %d epochs per evaluation\n", TOURNEY_BUDGET)
    println("  " * "-"^70)

    t0 = now()
    pop = [random_agent(0) for _ in 1:TOURNEY_POP]
    all_time_best = nothing
    gen_history = []

    for gen in 0:TOURNEY_GENS
        @printf("\n  Generation %d: Evaluating...\n", gen)

        for (i, g) in enumerate(pop)
            if g.fitness == 0.0
                build_and_evaluate!(g, X_tr, y_tr, X_te, y_te, in_dim)
            end
            if i % 5 == 0
                @printf("    %d/%d evaluated\r", i, length(pop))
            end
        end

        sort!(pop, by=g->g.fitness, rev=true)

        if all_time_best === nothing || pop[1].fitness > all_time_best.fitness
            all_time_best = deepcopy(pop[1])
        end

        counts, fits = species_census(pop)
        push!(gen_history, (gen=gen, best=pop[1].fitness, avg=mean(g.fitness for g in pop)))

        println("\n  ┌──────┬────────┬────────────────────────────┬───────┬───────┬────────┐")
        println("  │ Rank │ Species│ Architecture               │  Acc  │  F1   │ Params │")
        println("  ├──────┼────────┼────────────────────────────┼───────┼───────┼────────┤")
        for (i, g) in enumerate(pop[1:min(8, length(pop))])
            arch = join(g.hidden_widths, "->")
            if length(arch) > 26; arch = arch[1:23] * "..."; end
            tag = uppercase(string(g.species))[1:3]
            pk = g.params >= 1000 ? @sprintf("%dK", g.params/1000) : string(g.params)
            @printf("  │  %2d  │ %-6s │ %-26s │ %5.1f │ %.3f │ %-6s │\n",
                    i, tag, arch, g.acc*100, g.f1, pk)
        end
        println("  └──────┴────────┴────────────────────────────┴───────┴───────┴────────┘")

        print("  Species Census: ")
        for (sp, c) in counts
            avg_f = mean(fits[sp])
            @printf(" %s=%d(%.1f%%)", uppercase(string(sp)), c, avg_f*100)
        end
        println()

        gen >= TOURNEY_GENS && break

        new_pop = AgentGenome[]
        for i in 1:TOURNEY_ELITE
            push!(new_pop, deepcopy(pop[i]))
        end

        while length(new_pop) < TOURNEY_POP
            p1 = tourney_select(pop)
            p2 = tourney_select(pop)
            child = agent_crossover(p1, p2, gen+1)
            agent_mutate!(child)
            push!(new_pop, child)
        end
        pop = new_pop[1:TOURNEY_POP]
    end

    total_time = Dates.value(now() - t0) / 1000.0

    println("\n" * "═"^80)
    println("  TOURNAMENT CHAMPION")
    println("═"^80)
    @printf("  Species:      %s\n", uppercase(string(all_time_best.species)))
    println("  Architecture: $(join(all_time_best.hidden_widths, " -> "))")
    @printf("  Accuracy:     %.2f%%\n", all_time_best.acc * 100)
    @printf("  F1 Score:     %.4f\n", all_time_best.f1)
    @printf("  Fitness:      %.4f\n", all_time_best.fitness)
    @printf("  Parameters:   %d\n", all_time_best.params)
    @printf("  Born:         Generation %d\n", all_time_best.generation)

    println("\n  Evolution Timeline:")
    for h in gen_history
        bar_len = round(Int, h.best * 50)
        bar = "█"^bar_len * "░"^(50 - bar_len)
        @printf("    Gen %2d │ Best: %5.2f%% │ Avg: %5.2f%% │%s│\n",
                h.gen, h.best*100, h.avg*100, bar)
    end

    @printf("\n  Total Time: %.1fs\n", total_time)

    println("\n  FULL TRAINING — Champion Genome (200 epochs)")
    println("  " * "-"^70)

    if all_time_best.species == :mlp
        sizes = vcat([in_dim], all_time_best.hidden_widths, [2])
        champ = Net(sizes; drop=all_time_best.dropout)
        n = size(X_tr, 2)
        step = 0
        best_acc = 0.0
        best_ep = 0
        for ep in 1:200
            lr_t = all_time_best.lr * 0.5 * (1 + cos(pi * ep / 200))
            set_mode!(champ, true)
            perm = randperm(n)
            for s in 1:32:n
                step += 1
                idx = perm[s:min(s+31, n)]
                yp = forward!(champ, X_tr[:, idx])
                backward!(champ, onehot(y_tr[idx]), yp; l2=all_time_best.l2, clip=1.0)
                adam_step!(champ, lr_t, step)
            end
            m = evaluate(champ, X_te, y_te)
            if m.acc > best_acc; best_acc = m.acc; best_ep = ep; end
            if ep % 40 == 0
                @printf("    Ep %3d | TeAcc %5.2f%% | F1 %.4f\n", ep, m.acc*100, m.f1)
            end
        end
        final_m = evaluate(champ, X_te, y_te)
        @printf("\n  CHAMPION FINAL: Acc %.2f%% | F1 %.4f | AUC %.4f\n",
                final_m.acc*100, final_m.f1, final_m.auc)
        return all_time_best, champ, final_m

    elseif all_time_best.species == :kan
        sizes = vcat([in_dim], all_time_best.hidden_widths[1:1], [2])
        champ = KANNet(sizes, all_time_best.kan_degree)
        n = size(X_tr, 2)
        step = 0
        best_acc = 0.0
        for ep in 1:200
            lr_t = all_time_best.lr * 0.5 * (1 + cos(pi * ep / 200))
            perm = randperm(n)
            for s in 1:32:n
                step += 1
                idx = perm[s:min(s+31, n)]
                yp = forward!(champ, X_tr[:, idx])
                backward!(champ, onehot(y_tr[idx]), yp; l2=all_time_best.l2, clip=1.0)
                adam_step!(champ, lr_t, step)
            end
            set_mode!(champ, false)
            yp = forward!(champ, X_te)
            preds = [argmax(yp[:, i])-1 for i in 1:size(yp, 2)]
            acc = mean(preds .== y_te)
            if acc > best_acc; best_acc = acc; end
            if ep % 40 == 0
                @printf("    Ep %3d | TeAcc %5.2f%%\n", ep, acc*100)
            end
        end
        @printf("\n  CHAMPION FINAL: Best Acc %.2f%%\n", best_acc*100)
        return all_time_best, champ, (acc=best_acc,)

    elseif all_time_best.species == :hebbian
        h = all_time_best.hidden_widths[1]
        champ = PlasticNet(in_dim, h, 2)
        n = size(X_tr, 2)
        step = 0
        best_acc = 0.0
        for ep in 1:200
            lr_t = all_time_best.lr * 0.5 * (1 + cos(pi * ep / 200))
            reset_traces!(champ)
            set_mode!(champ, true)
            perm = randperm(n)
            for s in 1:32:n
                step += 1
                idx = perm[s:min(s+31, n)]
                yp = forward!(champ, X_tr[:, idx])
                backward!(champ, onehot(y_tr[idx]), yp; l2=all_time_best.l2, clip=1.0)
                adam_step!(champ, lr_t, step)
            end
            m = evaluate(champ, X_te, y_te)
            if m.acc > best_acc; best_acc = m.acc; end
            if ep % 40 == 0
                @printf("    Ep %3d | TeAcc %5.2f%% | F1 %.4f\n", ep, m.acc*100, m.f1)
            end
        end
        final_m = evaluate(champ, X_te, y_te)
        @printf("\n  CHAMPION FINAL: Acc %.2f%% | F1 %.4f | AUC %.4f\n",
                final_m.acc*100, final_m.f1, final_m.auc)
        return all_time_best, champ, final_m
    end
end

champion_genome, champion_model, champion_metrics = run_tournament()

mkpath("results")
open("results/evolution/tournament_results.txt", "w") do io
    println(io, "O.M.N.I. Multi-Agent Tournament Results -- $(now())")
    println(io, "="^60)
    @printf(io, "Champion Species: %s\n", uppercase(string(champion_genome.species)))
    println(io, "Architecture: $(join(champion_genome.hidden_widths, " -> "))")
    @printf(io, "Dropout: %.3f\n", champion_genome.dropout)
    @printf(io, "LR: %.2e\n", champion_genome.lr)
    @printf(io, "L2: %.2e\n", champion_genome.l2)
    if champion_genome.species == :kan
        @printf(io, "KAN Degree: %d\n", champion_genome.kan_degree)
    end
    if champion_genome.species == :hebbian
        @printf(io, "Trace Decay: %.3f\n", champion_genome.trace_decay)
    end
end

println("\n  Tournament results saved to results/tournament_results.txt")
