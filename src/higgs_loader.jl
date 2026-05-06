using DelimitedFiles, Statistics, Random, Printf, Dates, Mmap

const HIGGS_PROJECT_DIR = dirname(@__DIR__)
const HIGGS_FEATURES = 28

mutable struct HiggsDataset
    path::String
    n_samples::Int
    offsets::Vector{Int64}
    signal_count::Int
    class_weights::Vector{Float64}
end

function index_higgs(path::String)
    offsets = Int64[]
    signal_count = 0
    total_count = 0
    
    println("  Indexing 11 million rows (this may take a minute)...")
    
    # We use a buffered reader for speed
    open(path, "r") do f
        while !eof(f)
            pos = position(f)
            line = readline(f)
            length(line) == 0 && continue
            
            push!(offsets, pos)
            # The label is the first character '1' or '0'
            if line[1] == '1'
                signal_count += 1
            end
            total_count += 1
            
            if total_count % 1000000 == 0
                @printf("    Indexed %d million rows...\n", total_count ÷ 1000000)
            end
        end
    end
    
    n = length(offsets)
    background_count = n - signal_count
    
    # Balancing weights
    w_sig = n / (2.0 * signal_count)
    w_bg = n / (2.0 * background_count)
    
    HiggsDataset(path, n, offsets, signal_count, [w_bg, w_sig])
end

function parse_higgs_line(line::String)
    parts = split(line, ',')
    label = round(Int, parse(Float64, parts[1]))
    features = [parse(Float64, parts[i]) for i in 2:HIGGS_FEATURES+1]
    return features, label
end

function load_higgs_batch(ds::HiggsDataset, indices::Vector{Int})
    B = length(indices)
    X = zeros(HIGGS_FEATURES, B)
    y = zeros(Int, B)
    
    open(ds.path, "r") do f
        for (j, idx) in enumerate(indices)
            seek(f, ds.offsets[idx])
            line = readline(f)
            feat, lab = parse_higgs_line(line)
            X[:, j] = feat
            y[j] = lab
        end
    end
    
    return X, y
end

# Multi-class helpers adapted for binary HIGGS
function higgs_onehot(y::Vector{Int})
    oh = zeros(2, length(y))
    for (i, v) in enumerate(y)
        oh[v + 1, i] = 1.0
    end
    oh
end

function higgs_metrics(y_pred_mat, y_true::Vector{Int})
    preds = [argmax(y_pred_mat[:, i]) - 1 for i in 1:size(y_pred_mat, 2)]
    acc = mean(preds .== y_true)
    
    tp = sum((preds .== 1) .& (y_true .== 1))
    fp = sum((preds .== 1) .& (y_true .== 0))
    fn = sum((preds .== 0) .& (y_true .== 1))
    tn = sum((preds .== 0) .& (y_true .== 0))
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    
    return (acc=acc, f1=f1, prec=prec, rec=rec, tp=tp, fp=fp, fn=fn, tn=tn)
end
