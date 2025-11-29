"""
    _build_B(
        B::SparseArrays.SparseMatrixCSC{Float64, Int64},
        n_segments::Vector{Int}
    )

    Transform the B matrix from the original LDR problem to the respective
    piecewise format
    
    # Arguments
    - B::SparseArrays.SparseMatrixCSC{Float64, Int64}: B matrix from the 
        original LDR
    - n_segments::Vector{Int}: Vector that contains the number of segments at each
        piecewise variable

    # Returns
    ::SparseArrays.SparseMatrixCSC{Float64, Int64}: Piecewise representation of
        B matrix
"""
function _build_B(
    B::SparseArrays.SparseMatrixCSC{Float64, Int64},
    n_segments::Vector{Int}
)

    col_indices = vcat([fill(i+1, n_segments[i]) for i in 1:length(n_segments)]...)
    B_new = hcat(B[:, 1], B[:, col_indices])

    return B_new
end

"""
    _build_C(
        C::SparseArrays.SparseMatrixCSC{Float64, Int64},
        n_segments::Vector{Int}
    )

    Transform the C matrix from the original LDR problem to the respective
    piecewise format
    
    # Arguments
    - C::SparseArrays.SparseMatrixCSC{Float64, Int64}: C matrix from the
        original LDR
    - n_segments::Vector{Int}: Vector that contains the number of segments at each
        piecewise variable

    # Returns
    ::SparseArrays.SparseMatrixCSC{Float64, Int64}: Piecewise representation of
        C matrix
"""
function _build_C(
    C::SparseArrays.SparseMatrixCSC{Float64, Int64},
    n_segments::Vector{Int}
    )
    return _build_B(C, n_segments)
end

"""
    _build_h(
        PWRV_list::Vector{PWVR}
    )

    Build right side vector for the restriction W η ≥ h
    
    # Arguments
    - PWLR_list::Vector{PWVR}: Vector of all piecewise variables at the correct
        order

    # Returns
    ::Vector{Float64}: right side vector for the restriction W η ≥ h
"""
function _build_h(
    PWRV_list::Vector{PWVR}
)
    ub = Float64[]
    lb = Float64[]
    hu = Float64[]

    for pwvr in PWRV_list
        
        push!(lb, pwvr.η_vec[1])
        append!(lb, zeros(Float64, pwvr.n_breakpoints))

        push!(ub, pwvr.η_vec[2])
        for i in 2:(pwvr.n_breakpoints + 1)
            push!(ub, pwvr.η_vec[i + 1] - pwvr.η_vec[i])
        end

        if (pwvr.n_breakpoints == 0)
            continue
        end
        push!(hu, - last(pwvr.η_vec))
        push!(hu, pwvr.η_vec[1] * (pwvr.η_vec[3] - pwvr.η_vec[2]))
        append!(hu, zeros(Float64, max(pwvr.n_breakpoints - 1, 0)))
    end
    h = vcat([1.0], [-1.0], hu, -ub, lb)
    return h
end

"""
    _build_W(
        n_segments::Vector{Int},
        PWRV_list::Vector{PWVR}
    )

    Build the W matrix for the restriction W η ≥ h at compact format
    
    # Arguments
    - n_segments::Vector{Int}: Vector that contains the number of segments
        at each piecewise variable
    - PWLR_list::Vector{PWVR}: Vector of all piecewise variables at the correct
        order

    # Returns
    ::SparseArrays.SparseMatrixCSC{Float64, Int64}: W matrix for the restriction
        W η ≥ h at compact format
"""
function _build_W(
    n_segments::Vector{Int},
    PWRV_list::Vector{PWVR}
)
    dim_uncertainty = Int(sum(n_segments))
    
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    line = 1
    col = 1
    for pwvr in PWRV_list
        if (pwvr.n_breakpoints == 0)
            col += 1
            continue
        end

        if (pwvr.n_breakpoints > 0)
            for _ in 1:(pwvr.n_breakpoints + 1)
                push!(rows, line)
                push!(cols, col)
                push!(vals, 1)
                col += 1
            end
            col -= (pwvr.n_breakpoints + 1)
            line += 1
        end

        for i in 2:(pwvr.n_breakpoints + 1)
            push!(rows, line)
            push!(cols, col)
            push!(vals, -(pwvr.η_vec[i+1] - pwvr.η_vec[i]))

            push!(rows, line)
            push!(cols, col+1)
            push!(vals, pwvr.η_vec[i] - pwvr.η_vec[i-1])

            line += 1
            col += 1
        end
        col += 1
    end
    
    Wu = sparse(rows, cols, vals, line - 1, col - 1)

    nu = size(Wu, 1)
    top_block = [1  zeros(1, dim_uncertainty)
                -1 zeros(1, dim_uncertainty) ]

    middle_block = [zeros(nu, 1)   -Wu
                    zeros(dim_uncertainty, 1)  -SparseArrays.I(dim_uncertainty)
                    zeros(dim_uncertainty, 1)   SparseArrays.I(dim_uncertainty) ]

    W = [top_block;
        middle_block]

    return W
end