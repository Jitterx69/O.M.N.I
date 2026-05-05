#=
================================================================================
   NEUROEVOLUTION — Genetic Algorithm Architecture Search
  ==========================================================
  Evolves optimal neural network architectures using a from-scratch
  genetic algorithm. Zero external dependencies.

  Genome encodes:
    • Number of hidden layers (1–5)
    • Width of each hidden layer (16–512)
    • Dropout rate (0.05–0.6)
    • Learning rate (1e-5 – 5e-3)
    • L2 regularization (1e-6 – 1e-2)

  GA operators:
    • Tournament selection (size 3)
    • Uniform crossover with layer-swap
    • Gaussian mutation (widths, hyperparams, add/remove layers)
    • Elitism (top 2 survive unchanged)

  Fitness: best test accuracy after short training (configurable epochs)

  Usage:
    julia neuroevolution.jl
================================================================================
=#

include("../core.jl")

# ─── GA Configuration ──────────────────────────────────────────────────────────
const POP_SIZE = 20           # Population size
const N_GENERATIONS = 10      # Number of evolutionary generations
const TOURNAMENT_SIZE = 3     # Tournament selection pressure
const ELITE_COUNT = 2         # Elitism: top N survive unchanged
const MUTATION_RATE = 0.3     # Probability of mutating each gene
const CROSSOVER_RATE = 0.7    # Probability of crossover vs cloning
const FITNESS_EPOCHS = 20     # Epochs to train each candidate
const FITNESS_BATCH = 32      # Batch size for fitness evaluation

# Genome bounds
const MIN_LAYERS = 1
const MAX_LAYERS = 5
const MIN_WIDTH = 16
const MAX_WIDTH = 512
const MIN_DROPOUT = 0.05
const MAX_DROPOUT = 0.6
const MIN_LR = 1e-5
const MAX_LR = 5e-3
const MIN_L2 = 1e-6
const MAX_L2 = 1e-2

# ─── Genome ─────────────────────────────────────────────────────────────────────
mutable struct Genome
    hidden_widths::Vector{Int}    # widths of hidden layers
    dropout::Float64              # dropout rate
    lr::Float64                   # learning rate
    l2::Float64                   # L2 regularization
    fitness::Float64              # best test accuracy
    final_acc::Float64
    final_f1::Float64
    final_auc::Float64
    generation::Int               # born in which generation
end

function random_genome(gen::Int)
    n_layers = rand(MIN_LAYERS:MAX_LAYERS)
    # Generate widths that generally decrease (funnel shape)
    max_w = rand(64:MAX_WIDTH)
    widths = Int[]
    for i in 1:n_layers
        w = max(MIN_WIDTH, round(Int, max_w * (1 - 0.5 * (i-1) / max(n_layers-1, 1)) + randn() * 20))
        w = clamp(w, MIN_WIDTH, MAX_WIDTH)
        push!(widths, w)
    end
    sort!(widths, rev=true)  # wider layers first

    dropout = clamp(rand() * (MAX_DROPOUT - MIN_DROPOUT) + MIN_DROPOUT, MIN_DROPOUT, MAX_DROPOUT)
    lr = exp(log(MIN_LR) + rand() * (log(MAX_LR) - log(MIN_LR)))  # log-uniform
    l2 = exp(log(MIN_L2) + rand() * (log(MAX_L2) - log(MIN_L2)))

    Genome(widths, dropout, lr, l2, 0.0, 0.0, 0.0, 0.0, gen)
end

function genome_to_sizes(g::Genome, input_dim::Int)
    vcat([input_dim], g.hidden_widths, [2])
end

function genome_string(g::Genome)
    arch = join(g.hidden_widths, "→")
    @sprintf("[%s] dr=%.2f lr=%.1e l2=%.1e", arch, g.dropout, g.lr, g.l2)
end

function genome_params(g::Genome, input_dim::Int)
    sizes = genome_to_sizes(g, input_dim)
    total = 0
    for i in 1:length(sizes)-1
        total += sizes[i] * sizes[i+1] + sizes[i+1]  # Dense W + b
        if i < length(sizes) - 1
            total += 2 * sizes[i+1]  # BN γ + β
        end
    end
    total
end

# ─── Fitness Evaluation ────────────────────────────────────────────────────────
function evaluate_fitness!(g::Genome, X_train, y_train, X_test, y_test, input_dim)
    sizes = genome_to_sizes(g, input_dim)
    net = Net(sizes; drop=g.dropout)

    result = quick_train!(net, X_train, y_train, X_test, y_test;
                          epochs=FITNESS_EPOCHS, lr=g.lr, batch_size=FITNESS_BATCH,
                          l2=g.l2, clip=1.0)

    g.fitness = result.best_acc
    g.final_acc = result.final_acc
    g.final_f1 = result.final_f1
    g.final_auc = result.final_auc
    return g.fitness
end

# ─── Selection: Tournament ──────────────────────────────────────────────────────
function tournament_select(population::Vector{Genome})
    candidates = rand(population, TOURNAMENT_SIZE)
    return deepcopy(candidates[argmax([c.fitness for c in candidates])])
end

# ─── Crossover: Uniform with Layer Swap ─────────────────────────────────────────
function crossover(p1::Genome, p2::Genome, gen::Int)
    if rand() > CROSSOVER_RATE
        child = deepcopy(rand() < 0.5 ? p1 : p2)
        child.generation = gen
        return child
    end

    # Crossover hidden widths
    max_len = max(length(p1.hidden_widths), length(p2.hidden_widths))
    child_widths = Int[]
    for i in 1:max_len
        if i <= length(p1.hidden_widths) && i <= length(p2.hidden_widths)
            push!(child_widths, rand() < 0.5 ? p1.hidden_widths[i] : p2.hidden_widths[i])
        elseif i <= length(p1.hidden_widths)
            rand() < 0.5 && push!(child_widths, p1.hidden_widths[i])
        else
            rand() < 0.5 && push!(child_widths, p2.hidden_widths[i])
        end
    end

    if isempty(child_widths)
        child_widths = [rand(MIN_WIDTH:MAX_WIDTH)]
    end
    sort!(child_widths, rev=true)

    # Crossover hyperparameters
    dropout = rand() < 0.5 ? p1.dropout : p2.dropout
    lr = rand() < 0.5 ? p1.lr : p2.lr
    l2 = rand() < 0.5 ? p1.l2 : p2.l2

    Genome(child_widths, dropout, lr, l2, 0.0, 0.0, 0.0, 0.0, gen)
end

# ─── Mutation ───────────────────────────────────────────────────────────────────
function mutate!(g::Genome)
    # Mutate layer widths
    if rand() < MUTATION_RATE
        for i in eachindex(g.hidden_widths)
            if rand() < 0.4
                # Scale by ±30%
                factor = 1.0 + randn() * 0.3
                g.hidden_widths[i] = clamp(round(Int, g.hidden_widths[i] * factor), MIN_WIDTH, MAX_WIDTH)
            end
        end
        sort!(g.hidden_widths, rev=true)
    end

    # Add or remove a layer
    if rand() < MUTATION_RATE * 0.5
        if rand() < 0.5 && length(g.hidden_widths) < MAX_LAYERS
            # Add a layer
            new_w = clamp(round(Int, minimum(g.hidden_widths) * rand(0.5:0.1:1.0)), MIN_WIDTH, MAX_WIDTH)
            push!(g.hidden_widths, new_w)
            sort!(g.hidden_widths, rev=true)
        elseif length(g.hidden_widths) > MIN_LAYERS
            # Remove the smallest layer
            deleteat!(g.hidden_widths, argmin(g.hidden_widths))
        end
    end

    # Mutate dropout
    if rand() < MUTATION_RATE
        g.dropout = clamp(g.dropout + randn() * 0.1, MIN_DROPOUT, MAX_DROPOUT)
    end

    # Mutate learning rate (log-space)
    if rand() < MUTATION_RATE
        g.lr = clamp(g.lr * exp(randn() * 0.5), MIN_LR, MAX_LR)
    end

    # Mutate L2 (log-space)
    if rand() < MUTATION_RATE
        g.l2 = clamp(g.l2 * exp(randn() * 0.5), MIN_L2, MAX_L2)
    end

    g.fitness = 0.0  # reset fitness after mutation
end

# ─── Pretty Printing ───────────────────────────────────────────────────────────
function print_header()
    println()
    println("╔" * "═"^78 * "╗")
    println("║" * " "^12 * " NEUROEVOLUTION — Architecture Search" * " "^26 * "║")
    println("║" * " "^12 * "Genetic Algorithm for Neural Network Design" * " "^23 * "║")
    println("║" * " "^12 * "From Scratch • Julia Standard Library Only" * " "^23 * "║")
    println("╚" * "═"^78 * "╝")
end

function print_population(pop::Vector{Genome}, gen::Int, input_dim::Int; top_n=5)
    sorted = sort(pop, by=g->g.fitness, rev=true)
    best = sorted[1]
    avg_fit = mean(g.fitness for g in pop)
    worst = sorted[end]

    println()
    println("┌─────────────────────────────────────────────────────────────────────────────┐")
    @printf("│  Generation %2d                                                              │\n", gen)
    println("├─────┬──────────────────────────────────────────────┬──────┬──────┬───────────┤")
    println("│ Rk  │ Architecture                                 │ Acc  │ F1   │ Params    │")
    println("├─────┼──────────────────────────────────────────────┼──────┼──────┼───────────┤")

    for (i, g) in enumerate(sorted[1:min(top_n, length(sorted))])
        arch = join(g.hidden_widths, "→")
        if length(arch) > 44
            arch = arch[1:41] * "..."
        end
        params = genome_params(g, input_dim)
        param_str = params >= 1000 ? @sprintf("%dK", params÷1000) : string(params)
        marker = i == 1 ? "" : "  "
        @printf("│ %s%d │ %-44s │ %4.1f │ %.3f│ %-9s │\n",
                marker, i, arch, g.fitness*100, g.final_f1, param_str)
    end

    println("├─────┴──────────────────────────────────────────────┴──────┴──────┴───────────┤")
    @printf("│  Best: %5.2f%%  │  Avg: %5.2f%%  │  Worst: %5.2f%%  │  Pop: %d            │\n",
            best.fitness*100, avg_fit*100, worst.fitness*100, length(pop))
    println("└──────────────────────────────────────────────────────────────────────────────┘")
end

function print_genome_detail(g::Genome, input_dim::Int, label::String)
    sizes = genome_to_sizes(g, input_dim)
    params = genome_params(g, input_dim)

    println()
    println("  $label")
    println("  " * "─"^60)
    println("  Architecture:   $(join(sizes, " → "))")
    println("  Hidden layers:  $(length(g.hidden_widths))")
    println("  Parameters:     $(params)")
    @printf("  Dropout:        %.3f\n", g.dropout)
    @printf("  Learning rate:  %.2e\n", g.lr)
    @printf("  L2 reg:         %.2e\n", g.l2)
    @printf("  Fitness (acc):  %.2f%%\n", g.fitness * 100)
    @printf("  F1 Score:       %.4f\n", g.final_f1)
    @printf("  AUC:            %.4f\n", g.final_auc)
    println("  Born:           Generation $(g.generation)")
end

# ─── Main Evolution Loop ───────────────────────────────────────────────────────
function evolve()
    print_header()

    # Load data
    println("\n Loading and preparing data...")
    X_train, y_train, X_test, y_test = load_data(; top_k=300, verbose=true)
    input_dim = size(X_train, 1)

    println("\n️  GA Configuration:")
    println("   Population:     $POP_SIZE")
    println("   Generations:    $N_GENERATIONS")
    println("   Tournament:     $TOURNAMENT_SIZE")
    println("   Elitism:        $ELITE_COUNT")
    println("   Mutation rate:  $MUTATION_RATE")
    println("   Crossover rate: $CROSSOVER_RATE")
    println("   Fitness epochs: $FITNESS_EPOCHS")

    t0 = now()

    # ── Initialize population ───────────────────────────────────────────────
    println("\n Initializing population of $POP_SIZE random architectures...")
    population = [random_genome(0) for _ in 1:POP_SIZE]

    # Track all-time best
    all_time_best = nothing
    generation_bests = Genome[]

    for gen in 0:N_GENERATIONS
        gen_start = now()

        # ── Evaluate fitness ────────────────────────────────────────────────
        if gen == 0
            println("\n Evaluating initial population...")
        else
            @printf("\n Evaluating generation %d...\n", gen)
        end

        n_eval = 0
        for (i, g) in enumerate(population)
            if g.fitness == 0.0  # not yet evaluated
                n_eval += 1
                try
                    evaluate_fitness!(g, X_train, y_train, X_test, y_test, input_dim)
                catch e
                    # If architecture fails (e.g., dim mismatch), assign 0 fitness
                    g.fitness = 0.0
                    g.final_acc = 0.0
                    g.final_f1 = 0.0
                    g.final_auc = 0.0
                end

                # Progress indicator
                if n_eval % 5 == 0 || n_eval == count(g->g.fitness == 0.0, population) || i == length(population)
                    @printf("   Evaluated %d/%d candidates...\r", i, length(population))
                end
            end
        end
        println()  # clear progress line

        # ── Track best ──────────────────────────────────────────────────────
        gen_best = sort(population, by=g->g.fitness, rev=true)[1]
        push!(generation_bests, deepcopy(gen_best))

        if all_time_best === nothing || gen_best.fitness > all_time_best.fitness
            all_time_best = deepcopy(gen_best)
        end

        # ── Display ─────────────────────────────────────────────────────────
        gen_elapsed = Dates.value(now() - gen_start) / 1000
        print_population(population, gen, input_dim)
        @printf("   ⏱ Generation time: %.1fs\n", gen_elapsed)

        # Stop after last generation (don't create offspring)
        gen >= N_GENERATIONS && break

        # ── Create next generation ──────────────────────────────────────────
        sort!(population, by=g->g.fitness, rev=true)

        new_pop = Genome[]

        # Elitism: carry over top individuals unchanged
        for i in 1:min(ELITE_COUNT, length(population))
            elite = deepcopy(population[i])
            push!(new_pop, elite)
        end

        # Fill rest with offspring
        while length(new_pop) < POP_SIZE
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            child = crossover(parent1, parent2, gen + 1)
            mutate!(child)
            push!(new_pop, child)
        end

        population = new_pop[1:POP_SIZE]
    end

    total_time = Dates.value(now() - t0) / 1000

    # ── Final Results ───────────────────────────────────────────────────────
    println()
    println("╔" * "═"^78 * "╗")
    println("║" * " "^20 * " EVOLUTION COMPLETE" * " "^38 * "║")
    println("╚" * "═"^78 * "╝")

    print_genome_detail(all_time_best, input_dim, " ALL-TIME BEST ARCHITECTURE")

    # Show evolution progress
    println("\n   Evolution Progress:")
    println("  " * "─"^60)
    for (i, g) in enumerate(generation_bests)
        bar_len = round(Int, g.fitness * 40)
        bar = "█"^bar_len * "░"^(40 - bar_len)
        @printf("   Gen %2d: %5.2f%% │%s│\n", i-1, g.fitness*100, bar)
    end

    @printf("\n  ⏱  Total evolution time: %.1fs\n", total_time)
    @printf("   Architectures evaluated: ~%d\n", POP_SIZE + (N_GENERATIONS * (POP_SIZE - ELITE_COUNT)))

    # ── Full training of best architecture ──────────────────────────────────
    println("\n" * "="^78)
    println("   FULL TRAINING — Best Evolved Architecture")
    println("="^78)

    best_sizes = genome_to_sizes(all_time_best, input_dim)
    println("\n  Architecture: $(join(best_sizes, " → "))")
    println("  Training with full epochs (150) and evolved hyperparameters...")

    best_net = Net(best_sizes; drop=all_time_best.dropout)
    n = size(X_train, 2)
    step = 0
    best_acc = 0.0
    best_weights = nothing
    best_ep = 0

    full_epochs = 150
    for epoch in 1:full_epochs
        lr = all_time_best.lr * 0.5 * (1 + cos(π * ((epoch-1) % 50) / 50))
        set_mode!(best_net, true)
        perm = randperm(n)
        for start in 1:32:n
            step += 1
            idx = perm[start:min(start+31, n)]
            ŷ = forward!(best_net, X_train[:, idx])
            backward!(best_net, onehot(y_train[idx]), ŷ; l2=all_time_best.l2, clip=1.0)
            adam_step!(best_net, lr, step)
        end

        m = evaluate(best_net, X_test, y_test)
        if m.acc > best_acc
            best_acc = m.acc
            best_ep = epoch
            best_weights = [(copy(l.W), copy(l.b)) for l in best_net.layers]
        end

        if epoch == 1 || epoch % 25 == 0 || epoch == full_epochs
            train_m = evaluate(best_net, X_train, y_train)
            @printf("  Ep %3d │ TrAcc %5.1f%% │ TeAcc %5.1f%% │ F1 %.3f │ AUC %.3f │ LR %.1e\n",
                    epoch, train_m.acc*100, m.acc*100, m.f1, m.auc, lr)
        end
    end

    # Restore best
    if best_weights !== nothing
        for (i, (W, b)) in enumerate(best_weights)
            best_net.layers[i].W .= W
            best_net.layers[i].b .= b
        end
    end

    final_m = evaluate(best_net, X_test, y_test)
    c = final_m.cm

    println("\n" * "─"^78)
    println("  FINAL RESULTS — Evolved Architecture")
    println("─"^78)
    @printf("\n  Accuracy:   %6.2f%%\n", final_m.acc * 100)
    @printf("  Precision:  %6.2f%%\n", final_m.prec * 100)
    @printf("  Recall:     %6.2f%%\n", final_m.rec * 100)
    @printf("  F1 Score:   %6.4f\n", final_m.f1)
    @printf("  ROC-AUC:    %6.4f\n", final_m.auc)

    println("\n  Confusion Matrix:")
    println("  ┌──────────┬──────────┬──────────┐")
    println("  │          │ Pred: 0  │ Pred: 1  │")
    println("  ├──────────┼──────────┼──────────┤")
    @printf("  │ True: 0  │  %4d    │  %4d    │\n", c.tn, c.fp)
    @printf("  │ True: 1  │  %4d    │  %4d    │\n", c.fn, c.tp)
    println("  └──────────┴──────────┴──────────┘")

    # Save results
    mkpath(joinpath(CORE_PROJECT_DIR, "results"))
    open(joinpath(CORE_PROJECT_DIR, "results", "neuroevolution_results.txt"), "w") do io
        println(io, "Neuroevolution Results — $(now())")
        println(io, "="^60)
        println(io, "\nBest Architecture: $(join(best_sizes, " → "))")
        println(io, "Parameters: $(count_params(best_net))")
        @printf(io, "Dropout: %.3f\n", all_time_best.dropout)
        @printf(io, "Learning rate: %.2e\n", all_time_best.lr)
        @printf(io, "L2: %.2e\n", all_time_best.l2)
        @printf(io, "\nAccuracy:  %.4f\n", final_m.acc)
        @printf(io, "Precision: %.4f\n", final_m.prec)
        @printf(io, "Recall:    %.4f\n", final_m.rec)
        @printf(io, "F1:        %.4f\n", final_m.f1)
        @printf(io, "AUC:       %.4f\n", final_m.auc)
        println(io, "\nEvolution History:")
        for (i, g) in enumerate(generation_bests)
            @printf(io, "  Gen %2d: acc=%.4f arch=[%s] dr=%.3f lr=%.2e l2=%.2e\n",
                    i-1, g.fitness, join(g.hidden_widths, ","), g.dropout, g.lr, g.l2)
        end
    end

    println("\n   Results saved to: results/neuroevolution_results.txt")

    println("\n" * "╔" * "═"^78 * "╗")
    @printf("║   NEUROEVOLUTION COMPLETE                                                ║\n")
    @printf("║  Best architecture: %-56s  ║\n", join(best_sizes, " → "))
    @printf("║  Accuracy: %5.2f%% │ F1: %.4f │ AUC: %.4f                              ║\n",
            final_m.acc*100, final_m.f1, final_m.auc)
    println("╚" * "═"^78 * "╝")

    return all_time_best, best_net, final_m
end

# Run
evolve()
