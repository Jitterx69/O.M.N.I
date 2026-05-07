using Dates, Printf

println("\n" * "═"^80)
println("   STARTING ADVANCED JULIA ML PIPELINE")
println("  Generated: $(now())")
println("═"^80 * "\n")

if !isdir("data") || !isfile("data/train_dataset.csv") || !isfile("data/test_dataset.csv")
    println("  Data not found! Running generator first...")
    run(`python3 scripts/generate_dataset.py`)
end

println("\n" * "" * "  PHASE 1: NEUROEVOLUTION ARCHITECTURE SEARCH")
println("  " * "─"^76)
include("../../src/modules/neuroevolution.jl")

println("\n\n" * "" * "  PHASE 2: NEURAL PRUNING (COMPRESSION)")
println("  " * "─"^76)
include("../../src/modules/pruning.jl")

println("\n\n" * "" * "  PHASE 3: EXPLAINABILITY ENGINE (XAI)")
println("  " * "─"^76)
include("../../src/modules/explainability.jl")

println("\n\n" * "" * "  PHASE 4: BAYESIAN UNCERTAINTY QUANTIFICATION")
println("  " * "─"^76)
include("../../src/modules/uncertainty.jl")

println("\n" * "═"^80)
println("   ENTIRE ADVANCED PIPELINE COMPLETED SUCCESSFULLY!")
println("  All reports saved to: ./results/")
println("═"^80 * "\n")
