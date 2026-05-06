using Statistics, Random, Printf, Dates

const FINANCE_PROJECT_DIR = dirname(@__DIR__)
const FINANCE_DATA_PATH = joinpath(FINANCE_PROJECT_DIR, "data", "finance", "market_data.csv")

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
    
    # Convert to Log Returns: log(P_t / P_{t-1})
    returns = diff(log.(max.(data_mat, 1e-6)), dims=1)
    valid_dates = dates[2:end]
    
    # Standardize
    means = vec(mean(returns, dims=1))
    stds = vec(std(returns, dims=1))
    norm_returns = (returns .- means') ./ (stds' .+ 1e-8)
    
    return FinanceLoader(norm_returns, valid_dates, assets, means, stds)
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
