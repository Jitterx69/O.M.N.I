using DelimitedFiles, Statistics, Random, Printf, Dates, Mmap

const ECG_PROJECT_DIR = dirname(@__DIR__)
const ECG_CLASSES = 5
const ECG_TIMESTEPS = 187

mutable struct ECGDataset
    path::String
    n_samples::Int
    offsets::Vector{Int}
    label_counts::Vector{Int}
    class_weights::Vector{Float64}
end

function index_csv(path::String)
    offsets = Int[]
    label_counts = zeros(Int, ECG_CLASSES)

    open(path, "r") do f
        while !eof(f)
            push!(offsets, position(f))
            line = readline(f)
            length(line) == 0 && continue
            last_comma = findlast(',', line)
            label = round(Int, parse(Float64, line[last_comma+1:end]))
            label_counts[label + 1] += 1
        end
    end

    filter!(i -> i < filesize(path), offsets)
    n = length(offsets)

    total = sum(label_counts)
    class_weights = zeros(ECG_CLASSES)
    for c in 1:ECG_CLASSES
        if label_counts[c] > 0
            class_weights[c] = total / (ECG_CLASSES * label_counts[c])
        end
    end

    ECGDataset(path, n, offsets, label_counts, class_weights)
end

function parse_line(line::String)
    parts = split(line, ',')
    signal = [parse(Float64, parts[i]) for i in 1:ECG_TIMESTEPS]
    label = round(Int, parse(Float64, parts[end]))
    return signal, label
end

function load_batch(ds::ECGDataset, indices::Vector{Int})
    B = length(indices)
    X = zeros(ECG_TIMESTEPS, B)
    y = zeros(Int, B)

    open(ds.path, "r") do f
        for (j, idx) in enumerate(indices)
            seek(f, ds.offsets[idx])
            line = readline(f)
            sig, lab = parse_line(line)
            X[:, j] = sig
            y[j] = lab
        end
    end

    return X, y
end

function load_full(ds::ECGDataset)
    X = zeros(ECG_TIMESTEPS, ds.n_samples)
    y = zeros(Int, ds.n_samples)

    open(ds.path, "r") do f
        for i in 1:ds.n_samples
            seek(f, ds.offsets[i])
            line = readline(f)
            sig, lab = parse_line(line)
            X[:, i] = sig
            y[i] = lab
        end
    end

    return X, y
end

struct BatchIterator
    ds::ECGDataset
    batch_size::Int
    shuffle::Bool
end

function iterate_batches(iter::BatchIterator)
    perm = iter.shuffle ? randperm(iter.ds.n_samples) : collect(1:iter.ds.n_samples)
    batches = Tuple{Matrix{Float64}, Vector{Int}}[]

    for start in 1:iter.batch_size:iter.ds.n_samples
        stop = min(start + iter.batch_size - 1, iter.ds.n_samples)
        idx = perm[start:stop]
        X, y = load_batch(iter.ds, idx)
        push!(batches, (X, y))
    end

    return batches
end

function onehot_ecg(y::Vector{Int})
    oh = zeros(ECG_CLASSES, length(y))
    for (i, v) in enumerate(y)
        oh[v + 1, i] = 1.0
    end
    oh
end

function ecg_softmax(X)
    mx = maximum(X, dims=1)
    e = exp.(X .- mx)
    e ./ sum(e, dims=1)
end

function ecg_xent(y_pred, y_oh; weights=nothing)
    if weights === nothing
        return -mean(sum(y_oh .* log.(clamp.(y_pred, 1e-8, 1.0)), dims=1))
    end

    B = size(y_oh, 2)
    loss = 0.0
    for j in 1:B
        c = argmax(y_oh[:, j])
        w = weights[c]
        loss += -w * sum(y_oh[:, j] .* log.(clamp.(y_pred[:, j], 1e-8, 1.0)))
    end
    loss / B
end

function ecg_metrics(y_pred_mat, y_true::Vector{Int})
    preds = [argmax(y_pred_mat[:, i]) - 1 for i in 1:size(y_pred_mat, 2)]
    acc = mean(preds .== y_true)

    per_class_prec = zeros(ECG_CLASSES)
    per_class_rec = zeros(ECG_CLASSES)
    per_class_f1 = zeros(ECG_CLASSES)

    for c in 0:ECG_CLASSES-1
        tp = sum((preds .== c) .& (y_true .== c))
        fp = sum((preds .== c) .& (y_true .!= c))
        fn = sum((preds .!= c) .& (y_true .== c))

        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-8)

        per_class_prec[c+1] = p
        per_class_rec[c+1] = r
        per_class_f1[c+1] = f
    end

    macro_f1 = mean(per_class_f1)
    weighted_f1 = 0.0
    total = length(y_true)
    for c in 0:ECG_CLASSES-1
        n_c = sum(y_true .== c)
        weighted_f1 += per_class_f1[c+1] * n_c / total
    end

    (acc=acc, macro_f1=macro_f1, weighted_f1=weighted_f1,
     per_class_f1=per_class_f1, per_class_prec=per_class_prec,
     per_class_rec=per_class_rec, preds=preds)
end

function print_dataset_info(ds::ECGDataset, name::String)
    println("  $name: $(ds.n_samples) samples")
    class_names = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]
    for c in 1:ECG_CLASSES
        pct = ds.label_counts[c] / ds.n_samples * 100
        @printf("    Class %d (%s): %6d samples (%5.1f%%) | weight: %.3f\n",
                c-1, rpad(class_names[c], 16), ds.label_counts[c], pct, ds.class_weights[c])
    end
end

function print_confusion_matrix(preds, y_true)
    class_names = ["Norm", "Supra", "Vent", "Fus", "Unkn"]
    println("\n  Confusion Matrix:")
    print("  " * lpad("", 8))
    for c in 1:ECG_CLASSES
        print(lpad(class_names[c], 7))
    end
    println()
    println("  " * "-"^(8 + 7 * ECG_CLASSES))
    for true_c in 0:ECG_CLASSES-1
        print("  " * rpad(class_names[true_c+1], 8))
        for pred_c in 0:ECG_CLASSES-1
            count = sum((preds .== pred_c) .& (y_true .== true_c))
            print(lpad(string(count), 7))
        end
        println()
    end
end
