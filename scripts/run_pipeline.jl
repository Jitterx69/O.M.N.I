#=
================================================================================
   ADVANCED JULIA ML — Master Pipeline Orchestrator
  ==============================================================================
  Runs the entire advanced pipeline end-to-end:
    1. Neuroevolution (find best architecture)
    2. Neural Pruning (compress best architecture)
    3. Explainability Engine (feature importance)
    4. Bayesian Uncertainty (confidence scoring)

  Usage:
    julia run_pipeline.jl
================================================================================
=#

using Dates, Printf

println("\n" * "═"^80)
println("   STARTING ADVANCED JULIA ML PIPELINE")
println("  Generated: $(now())")
println("═"^80 * "\n")

# Make sure we have the data
if !isdir("data") || !isfile("data/train_dataset.csv") || !isfile("data/test_dataset.csv")
    println("  Data not found! Running generator first...")
    run(`python3 scripts/generate_dataset.py`)
end

# 1. Evolution
println("\n" * "" * "  PHASE 1: NEUROEVOLUTION ARCHITECTURE SEARCH")
println("  " * "─"^76)
include("../src/modules/neuroevolution.jl")

# 2. Pruning
println("\n\n" * "" * "  PHASE 2: NEURAL PRUNING (COMPRESSION)")
println("  " * "─"^76)
include("../src/modules/pruning.jl")

# 3. Explainability
println("\n\n" * "" * "  PHASE 3: EXPLAINABILITY ENGINE (XAI)")
println("  " * "─"^76)
include("../src/modules/explainability.jl")

# 4. Uncertainty
println("\n\n" * "" * "  PHASE 4: BAYESIAN UNCERTAINTY QUANTIFICATION")
println("  " * "─"^76)
include("../src/modules/uncertainty.jl")

println("\n" * "═"^80)
println("   ENTIRE ADVANCED PIPELINE COMPLETED SUCCESSFULLY!")
println("  All reports saved to: ./results/")
println("═"^80 * "\n")
