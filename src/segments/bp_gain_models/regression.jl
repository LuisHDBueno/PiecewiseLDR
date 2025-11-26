function open_params()
    filepath = joinpath(@__DIR__, "params.json") 
    if !isfile(filepath)
        error("Arquivo de parâmetros não encontrado: $filepath")
    end
    
    json_data = JSON.parsefile(filepath)
    return Vector{Float64}(json_data)
end

function get_bp_gain(
    pwldr::PWLDR,
    variable::JuMP.VariableRef;
    n_samples::Int = 100
)
    β = open_params()
    V = vector_representation(pwldr, variable; n_samples)
    return sum(β .* V)
end