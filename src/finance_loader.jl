using Statistics, Random, Printf, Dates

const FINANCE_PROJECT_DIR = dirname(@__DIR__)
const FINANCE_DATA_PATH = joinpath(FINANCE_PROJECT_DIR, "data", "finance", "market_data.csv")

# Fractional Differentiation helper (Fixed Window Method)
function fractional_diff(x::Vector{Float64}, d::Float64, threshold::Float64=1e-4)
    w = [1.0]
    for k in 1:100 # Window size for weights
        push!(w, -w[end] * (d - k + 1) / k)
        if abs(w[end]) < threshold; break; end
    end
    
    n = length(x)
    nw = length(w)
    res = zeros(n)
    for i in nw:n
        res[i] = sum(w .* x[i:-1:i-nw+1])
    end
    return res[nw:end]
end

mutable struct FinanceLoader
    data::Matrix{Float64}
    dates::Vector{Date}
    assets::Vector{String}
    means::Vector{Float64}
    stds::Vector{Float64}
end

function load_finance_data(path::String)
    lines = readlines(path)
    header = split(lines[1], ',')
    assets = String.(header[2:end])
    
    dates = Date[]
    rows = Vector{Float64}[]
    
    for line in lines[2:end]
        parts = split(line, ',')
        push!(dates, Date(parts[1]))
        
        row = Float64[]
        for p in parts[2:end]
            val = tryparse(Float64, p)
            push!(row, isnan(val === nothing ? NaN : val) ? 0.0 : val)
        end
        push!(rows, row)
    end
    
    data_mat = hcat(rows...)' |> Matrix
    
    # Forward fill zeros (simple missing value handling for finance)
    for j in 1:size(data_mat, 2)
        last_val = 0.0
        for i in 1:size(data_mat, 1)
            if data_mat[i, j] == 0.0
                data_mat[i, j] = last_val
            else
                last_val = data_mat[i, j]
            end
        end
    end
    
    # Fractional Differentiation (d=0.4 preserves memory)
    n_assets = size(data_mat, 2)
    frac_cols = []
    for j in 1:n_assets
        # FracDiff on log prices
        push!(frac_cols, fractional_diff(log.(max.(data_mat[:, j], 1e-6)), 0.4))
    end
    
    # Trim dates and data to match frac_diff delay
    min_len = minimum(length.(frac_cols))
    data_mat = hcat([c[end-min_len+1:end] for c in frac_cols]...)
    valid_dates = dates[end-min_len+1:end]
    
    # Feature Engineering: Add 5-day Volatility and 5-day Momentum
    n_rows, n_assets = size(data_mat)
    vol = zeros(n_rows, n_assets)
    mom = zeros(n_rows, n_assets)
    
    for j in 1:n_assets
        for i in 6:n_rows
            vol[i, j] = std(data_mat[i-5:i, j])
            mom[i, j] = sum(data_mat[i-5:i, j])
        end
    end
    
    # Concatenate: [FracDiff | Volatility | Momentum]
    enhanced_data = hcat(data_mat, vol, mom)
    
    # Standardize
    means = vec(mean(enhanced_data, dims=1))
    stds = vec(std(enhanced_data, dims=1))
    norm_data = (enhanced_data .- means') ./ (stds' .+ 1e-8)
    
    return FinanceLoader(norm_data, valid_dates, assets, means, stds)
end

function create_windows(loader::FinanceLoader, window_size::Int, target_asset_idx::Int)
    n_samples = size(loader.data, 1) - window_size
    n_features = size(loader.data, 2)
    
    X = zeros(window_size * n_features, n_samples)
    y = zeros(1, n_samples)
    
    for i in 1:n_samples
        # Flat window: [asset1_t1, asset2_t1, ..., assetN_t1, asset1_t2, ...]
        window = loader.data[i:i+window_size-1, :]
        X[:, i] = vec(window')
        
        # Target: Is next day return positive? (Binary classification for stability)
        y[1, i] = loader.data[i+window_size, target_asset_idx] > 0 ? 1.0 : 0.0
    end
    
    return X, y
end

function split_chronological(X, y, train_ratio=0.7)
    n = size(X, 2)
    split_idx = floor(Int, n * train_ratio)
    
    X_train = X[:, 1:split_idx]
    y_train = y[:, 1:split_idx]
    
    X_test = X[:, split_idx+1:end]
    y_test = y[:, split_idx+1:end]
    
    return X_train, y_train, X_test, y_test
end
