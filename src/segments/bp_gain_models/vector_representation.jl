""""
    function get_statistics(
        v::Vector{Float64}
    )

    Get 5 statistics about the given vector

    # Arguments
    - v::Vector{Float64}: Vector to get statistics

    #Returns
    @NamedTuple{min::Float64, max::Float64, mean::Float64, median::Float64, std::Float64}
"""
function get_statistics(
    v::Vector{Float64}
)
    return (
        min = minimum(v),
        max = maximum(v),
        mean = Statistics.mean(v),
        median = Statistics.median(v),
        std = Statistics.std(v; corrected=false)
    )
end

function vector_representation(
    pwldr::PWLDR,
    variable::JuMP.VariableRef;
    n_samples::Int = 100
)

    dist_idx, inner_idx = pwldr.uncertainty_to_distribution[variable]
    dist = pwldr.PWRV_list[dist_idx].distribution
    ABC = pwldr.ldr_model.ext[:_LDR_ABC]

    A = vcat(ABC.Au, ABC.Al, ABC.Ae)
    V_A = zeros(5)
    
    for line in size(A, 1)
        a_i = Vector(A[line,1:end])
        a_i = a_i / maximum(a_i)
        V_A .+= values(get_statistics(a_i))
    end
    V_A = V_A / size(A, 1)

    B = vcat(ABC.Bu, ABC.Bl, ABC.Be)
    C = ABC.C

    ξ = [1.0]
    for pwvr in pwldr.PWRV_list
        push!(ξ, rand(pwvr.distribution))
    end

    V_B = zeros(5)
    V_C = zeros(5)
    for _ in 1:n_samples
        sample = rand(dist)
        ξ[dist_idx] = sample
        b = values(get_statistics(B * ξ))
        b = b ./ maximum(b)
        V_B .+= 
        c = values(get_statistics(C * ξ))
        c = c ./ maximum(c)
        V_C .+= c
    end

    V_B = V_B/n_samples
    V_C = V_C/n_samples

    return vcat(V_C, V_B, V_A)
end

function vector_representation(
    pwldr::PWLDR,
    variable_list::Vector{JuMP.VariableRef};
    n_samples::Int = 100
)

    dist_list = []
    for var in variable_list
        dist_idx, inner_idx = pwldr.uncertainty_to_distribution[variable]
        dist = pwldr.PWRV_list[dist_idx].distribution
        push!(dist_list, (idx = dist_idx, distribution = dist))
    end
    ABC = pwldr.ldr_model.ext[:_LDR_ABC]

    A = vcat(ABC.Au, ABC.Al, ABC.Ae)
    V_A = zeros(5)
    
    for line in size(A, 1)
        V_A .+= values(get_statistics(Vector(A[line,1:end])))
    end
    V_A = V_A / size(A, 1)

    B = vcat(ABC.Bu, ABC.Bl, ABC.Be)
    C = ABC.C

    ξ = [1.0]
    for pwvr in pwldr.PWRV_list
        push!(ξ, rand(pwvr.distribution))
    end

    V_B = zeros(5)
    V_C = zeros(5)
    for _ in 1:n_samples
        for var in dist_list
            sample = rand(var.distribution)
            ξ[var.dist_idx] = sample
        end
        V_B .+= values(get_statistics(B * ξ))
        V_C .+= values(get_statistics(C * ξ))
    end

    V_B = V_B/n_samples
    V_C = V_C/n_samples

    return vcat(V_C, V_B, V_A)
end


