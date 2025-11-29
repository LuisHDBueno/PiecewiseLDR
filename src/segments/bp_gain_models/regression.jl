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
    variable::JuMP.VariableRef;
    n_samples::Int = 100
)
    params = open_params()

    coef = params.coef
    mean = params.mean
    std  = params.std
    features = params.features

    V_full = vector_representation(pwldr, variable; n_samples)
    V = [V_full[parse(Int, replace(f, "v" => ""))] for f in features]
    V_norm = [(V[i] - mean[features[i]]) / std[features[i]] for i in eachindex(V)]

    ŷ = coef["intercept"]

    for i in eachindex(V_norm)
        ŷ += coef[features[i]] * V_norm[i]
    end

    return ŷ
end