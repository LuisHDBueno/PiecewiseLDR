"""
    _build_second_moment_matrix(
        n_segments::Vector{Int},
        PWRV_list::Vector{PWVR}
    )
    
    Build the second moment matrix associated to the PWRV_list

    # Arguments
    - n_segments::Vector{Int}: Vector that contains the number of segments
        at each piecewise variable
    - PWLR_list::Vector{PWVR}: Vector of all piecewise variables at the correct
        order

    # Returns
    ::Matrix{Float64}: Second moment matrix
"""
function _build_second_moment_matrix(
    n_segments::Vector{Int},
    PWRV_list::Vector{PWVR}
)

    n_cols = Int(sum(n_segments)) + 1
    M = zeros(n_cols, n_cols)

    M[1,1] = 1

    line_indices = Int[]
    line = 2
    for pwvr in PWRV_list
        μ = mean(pwvr)
        n = pwvr.n_breakpoints + 1

        M[1, line:line+n-1] .= μ
        M[line:line+n-1, 1] .= μ

        cov_d = cov(pwvr)
        Σ = cov_d .+ μ * μ'
        M[line:line+n-1, line:line+n-1] .= Σ

        push!(line_indices, line)
        line += n
    end

    for i in 1:length(PWRV_list)-1
        for j in i+1:length(PWRV_list)
            pwvr_i = PWRV_list[i]
            pwvr_j = PWRV_list[j]

            lines_i = line_indices[i] : line_indices[i] + pwvr_i.n_breakpoints
            lines_j = line_indices[j] : line_indices[j] + pwvr_j.n_breakpoints

            μ_i = mean(pwvr_i)
            μ_j = mean(pwvr_j)

            Σ_ij = cov(pwvr_i, pwvr_j) .+ μ_i * μ_j'

            M[lines_i, lines_j] .= Σ_ij
            M[lines_j, lines_i] .= Σ_ij'
        end
    end

    return M
end