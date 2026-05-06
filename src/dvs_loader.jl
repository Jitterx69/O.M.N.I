using Statistics, Random, Printf, Dates

const DVS_PROJECT_DIR = dirname(@__DIR__)
const DVS_DATA_DIR = joinpath(DVS_PROJECT_DIR, "data", "DVS  Gesture dataset", "DvsGesture")
const DVS_RES = 128
const DVS_DOWN = 4
const DVS_RES_DOWN = div(DVS_RES, DVS_DOWN)
const DVS_CLASSES = 11
const DVS_TIME_BINS = 20

struct DVSEvent
    x::Int
    y::Int
    pol::Int
    ts::Int64
end

function skip_aedat_header(io::IO)
    while !eof(io)
        pos = position(io)
        line = readline(io)
        if isempty(line) || line[1] != '#'
            seek(io, pos)
            return
        end
    end
end

function read_aedat_events(path::String; max_events::Int=5000000)
    events = DVSEvent[]
    sizehint!(events, 500000)

    open(path, "r") do io
        skip_aedat_header(io)

        while !eof(io) && length(events) < max_events
            if position(io) + 28 > filesize(path)
                break
            end

            event_type = read(io, UInt16)
            event_source = read(io, UInt16)
            event_size = read(io, UInt32)
            event_ts_offset = read(io, UInt32)
            event_ts_overflow = read(io, UInt32)
            event_capacity = read(io, UInt32)
            event_number = read(io, UInt32)
            event_valid = read(io, UInt32)

            if event_type != 1
                bytes_to_skip = event_number * event_size
                if position(io) + bytes_to_skip <= filesize(path)
                    seek(io, position(io) + bytes_to_skip)
                end
                continue
            end

            for _ in 1:event_number
                if eof(io) || position(io) + 8 > filesize(path)
                    break
                end
                if length(events) >= max_events
                    break
                end

                data = read(io, UInt32)
                timestamp = read(io, UInt32)

                x = Int((data >> 17) & 0x00001FFF)
                y = Int((data >> 2) & 0x00001FFF)
                pol = Int((data >> 1) & 0x00000001)

                if x < DVS_RES && y < DVS_RES
                    push!(events, DVSEvent(x, y, pol, Int64(timestamp)))
                end
            end
        end
    end

    return events
end

function load_gesture_labels(label_path::String)
    labels = Tuple{Int, Int64, Int64}[]
    lines = readlines(label_path)
    for line in lines[2:end]
        parts = split(strip(line), ',')
        length(parts) < 3 && continue
        cls = parse(Int, parts[1])
        t_start = parse(Int64, parts[2])
        t_end = parse(Int64, parts[3])
        push!(labels, (cls, t_start, t_end))
    end
    return labels
end

function extract_gesture_events(events::Vector{DVSEvent}, t_start::Int64, t_end::Int64)
    filtered = DVSEvent[]
    for e in events
        if e.ts >= t_start && e.ts <= t_end
            push!(filtered, e)
        end
    end
    return filtered
end

function events_to_voxel_grid(events::Vector{DVSEvent}, n_bins::Int)
    R = DVS_RES_DOWN
    grid = zeros(Float64, 2 * n_bins * R * R)

    length(events) == 0 && return grid

    t_min = minimum(e.ts for e in events)
    t_max = maximum(e.ts for e in events)
    dt = max(t_max - t_min, 1)

    for e in events
        bin = min(floor(Int, (e.ts - t_min) / dt * n_bins), n_bins - 1)
        xd = div(e.x, DVS_DOWN)
        yd = div(e.y, DVS_DOWN)
        xd = clamp(xd, 0, R - 1)
        yd = clamp(yd, 0, R - 1)
        ch = e.pol * n_bins + bin
        idx = ch * R * R + yd * R + xd + 1
        if idx >= 1 && idx <= length(grid)
            grid[idx] += 1.0
        end
    end

    mx = maximum(grid)
    if mx > 0
        grid ./= mx
    end

    return grid
end

function load_trial_gestures(aedat_path::String, label_path::String, n_bins::Int)
    events = read_aedat_events(aedat_path)
    labels = load_gesture_labels(label_path)

    X_list = Vector{Float64}[]
    y_list = Int[]

    for (cls, t_start, t_end) in labels
        cls > DVS_CLASSES && continue
        gesture_events = extract_gesture_events(events, t_start, t_end)
        length(gesture_events) < 50 && continue

        voxel = events_to_voxel_grid(gesture_events, n_bins)
        push!(X_list, voxel)
        push!(y_list, cls)
    end

    return X_list, y_list
end

function load_split(trial_list::Vector{String}, n_bins::Int)
    X_all = Vector{Float64}[]
    y_all = Int[]

    for trial_name in trial_list
        aedat_path = joinpath(DVS_DATA_DIR, trial_name)
        label_name = replace(trial_name, ".aedat" => "_labels.csv")
        label_path = joinpath(DVS_DATA_DIR, label_name)

        if !isfile(aedat_path) || !isfile(label_path)
            continue
        end

        X_t, y_t = load_trial_gestures(aedat_path, label_path, n_bins)
        append!(X_all, X_t)
        append!(y_all, y_t)
    end

    feat_dim = 2 * n_bins * DVS_RES_DOWN * DVS_RES_DOWN
    X = zeros(feat_dim, length(X_all))
    for (i, v) in enumerate(X_all)
        X[:, i] = v
    end

    return X, y_all
end

function read_trial_list(path::String)
    lines = readlines(path)
    return [String(strip(l)) for l in lines if length(strip(l)) > 0]
end
