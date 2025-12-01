function open_params()
    filepath = joinpath(@__DIR__, "model_params.json")
    if !isfile(filepath)
        error("Arquivo de parâmetros não encontrado: $filepath")
    end

    json_data = JSON.parsefile(filepath)
    model = json_data[1]

    return (
        coef = model["coefficients"],
        mean = model["mean"],
        std  = model["std"],
        features = model["features"]
    )
end

function get_bp_gain(
    pwldr::PWLDR,
    variable::JuMP.VariableRef,
    k̂::Int;
    n_samples::Int = 100
)
    params = open_params()

    coef = params.coef
    mean = params.mean
    std  = params.std
    features = params.features

    V_full = vector_representation(pwldr, variable; n_samples)
    X = Float64[]

    for f in features
        if f == "k"
            push!(X, k̂)

        elseif occursin("_k", f)
            idx = parse(Int, replace(replace(f, "_k" => ""), "v" => ""))
            push!(X, V_full[idx] * k̂)

        else
            idx = parse(Int, replace(f, "v" => ""))
            push!(X, V_full[idx])
        end
    end

    X_norm = [(X[i] - mean[features[i]]) / std[features[i]] for i in eachindex(X)]
    ŷ = coef["intercept"]

    for i in eachindex(X_norm)
        ŷ += coef[features[i]] * X_norm[i]
    end

    return ŷ
end