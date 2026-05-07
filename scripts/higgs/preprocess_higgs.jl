include("../../src/higgs_loader.jl")

function preprocess_higgs()
    println("\n" * "█"^80)
    println("   HIGGS BINARY DATA ENGINE — Pre-processing 11 Million Samples")
    println("   Converting CSV -> Float32 Binary (Physics Augmented)")
    println("█"^80)

    csv_path = joinpath(HIGGS_PROJECT_DIR, "data", "higgs", "HIGGS.csv")
    bin_path = joinpath(HIGGS_PROJECT_DIR, "data", "higgs", "HIGGS.bin")

    if !isfile(csv_path)
        println("  ERROR: HIGGS.csv not found at $csv_path")
        return
    end

    n_samples = 11000000
    # 1 label + 38 features (28 raw + 10 physics augmented) = 39 Float32 per sample
    floats_per_sample = 39 
    
    println("  Source: $csv_path")
    println("  Target: $bin_path")
    println("  Expected Size: ~1.7 GB")
    println("  Starting stream conversion...")

    t0 = now()
    
    open(bin_path, "w") do bin_f
        open(csv_path, "r") do csv_f
            count = 0
            while !eof(csv_f)
                line = readline(csv_f)
                length(line) == 0 && continue
                
                # parse_higgs_line returns (vcat(f, aug), label)
                features, label = parse_higgs_line(line)
                
                # Write label first, then 38 features
                write(bin_f, Float32(label))
                for f in features
                    write(bin_f, Float32(f))
                end
                
                count += 1
                if count % 500000 == 0
                    @printf("    Processed %d million samples...\n", count / 1000000)
                end
                if count >= n_samples; break; end
            end
            println("\n  Success! Processed $count samples.")
        end
    end

    elapsed = Dates.value(now() - t0) / 1000.0
    @printf("  Pre-processing Complete | Time: %.1fs\n", elapsed)
    println("█"^80 * "\n")
end

preprocess_higgs()
