"""
    PWLDR

    Mutable struct that represents the piecewise linear problem

    # Fields
    - model::JuMP.Model: JuMP model at the piecewise form
    - ldr_model::LinearDecisionRules.LDRModel: Original LDR model
    - PWRV_list::Vector{PWVR}: List of existing piecewise variables
    - n_segments_vec::Vector{Int}: Vector of number of segments for each
        variable
    - W_constraints::Dict{Symbol, Matrix{JuMP.ConstraintRef}}: W matrix
        dependent constraints
    - h_constraints::Dict{Symbol, Vector{JuMP.ConstraintRef}}: h vector
        dependent constraints
    - reset_model::Bool: Flag to reset the JuMP model before optimize
    - uncertainty_to_distribution::Dict{JuMP.VariableRef, Tuple{Int64, Int64}}:
        Order of uncertainty distributions
"""
mutable struct PWLDR
    model::JuMP.Model
    ldr_model::LinearDecisionRules.LDRModel
    PWRV_list::Vector{PWVR}
    n_segments_vec::Vector{Int}
    W_constraints::Dict{Symbol, Matrix{JuMP.ConstraintRef}}
    h_constraints::Dict{Symbol, Vector{JuMP.ConstraintRef}}
    reset_model::Bool
    uncertainty_to_distribution::Dict{JuMP.VariableRef, Tuple{Int64, Int64}}
end

"""
    flatten_distributions_in_order(
        ldr_model::LinearDecisionRules.LDRModel
    )
    
    Flatten and order all the distributions of the original LDR problem into a
        single vector of distributions

    # Arguments
    - ldr_model::LinearDecisionRules.LDRModel: Original LDR model

    # Returns
    ::Vector{Distribution}: All distributions in order
"""
function flatten_distributions_in_order(
    ldr_model::LinearDecisionRules.LDRModel
)
    scalar_distributions = ldr_model.cache_model.scalar_distributions
    vector_distributions = ldr_model.cache_model.vector_distributions
    uncertainty_to_distribution = ldr_model.cache_model.uncertainty_to_distribution

    variables = all_variables(ldr_model.cache_model.model)
    all_distributions = Vector{Distribution}()

    scalar_idx = 1
    vector_idx = 1
    temp = []

    for var in variables
        if !(var in keys(uncertainty_to_distribution))
            continue
        end
        dist_idx, inner_idx = uncertainty_to_distribution[var]

        if inner_idx == 0
            push!(temp, (dist_idx, scalar_distributions[scalar_idx]))
            scalar_idx += 1
        else
            push!(temp, (dist_idx, vector_distributions[vector_idx].v[inner_idx]))
            if inner_idx == length(vector_distributions[vector_idx].v)
                vector_idx += 1
            end
        end
    end

    sorted_temp = sort(temp, by = x -> x[1])
    for (_, dist) in sorted_temp
        push!(all_distributions, dist)
    end

    return all_distributions
end

"""
    _build_init_PWRV_list(
        n_segments_vec::Vector{Int},
        η_min::Vector{Float64},
        η_max::Vector{Float64},
        distribution_vec::Vector{Distribution}
    )
    
    Build the list of piecewise random variables with evenly divided segments

    # Arguments
    - n_segments_vec::Vector{Int}: Number os segments at each piecewise random
        variable
    - η_min::Vector{Float64}: Lower bound of each piecewise random variable
    - η_max::Vector{Float64}: Upper bound of each piecewise random variable
    - distribution_vec::Vector{Distribution}: Distribution of each piecewise
        random variable

    # Returns
    ::Vector{PWVR}: Vector of piecewise random variables
"""
function _build_init_PWRV_list(
    n_segments_vec::Vector{Int},
    η_min::Vector{Float64},
    η_max::Vector{Float64},
    distribution_vec::Vector{Distribution}
)
    PWRV_list = Vector{PWVR}()
    for i in 1:length(η_min)
        n = n_segments_vec[i]
        push!(PWRV_list,
                PWVR(truncated(distribution_vec[i], η_min[i], η_max[i]),
                        η_min[i],
                        η_max[i],
                        fill(1/n, Int(n)))
                )
    end
    return PWRV_list
end

"""
    _build_problem(
        pwldr::PWLDR
    )

    Build the JuMP model that represents the piecewise linear problem and fill
        the information at the PWLDR struct

    # Arguments
    - pwldr::PWLDR: Struct with the minimum information to build the JuMP model.
            minimum information:
                - ldr_model
                - n_segments_vec
                - uncertainty_to_distribution
    
"""
function _build_problem!(
    pwldr::PWLDR
)
    ldr_model = pwldr.ldr_model
    n_segments_vec = pwldr.n_segments_vec
    ABC = ldr_model.ext[:_LDR_ABC]
    first_stage_index = ldr_model.ext[:_LDR_first_stage_indices]

    η_min = ABC.lb
    η_max = ABC.ub

    distribution_vec = flatten_distributions_in_order(ldr_model)
    PWRV_list = _build_init_PWRV_list(n_segments_vec, η_min, η_max, distribution_vec)

    # Model Init
    model = JuMP.Model(ldr_model.solver)
    set_silent(model)
    model.ext[:first_stage_index] = first_stage_index

    dim_X = size(ABC.Ae, 2)
    dim_ξ = Int(sum(n_segments_vec) + 1)

    @expression(model, X[1:dim_X, 1:dim_ξ], AffExpr(0.0))
    for i in 1:dim_X
        if i in first_stage_index
            X[i, 1] = @variable(model, base_name = "X[$i,1]")
        else
            for j in 1:dim_ξ
                X[i, j] = @variable(model, base_name = "X[$i,$j]")
            end
        end
    end

    #Reference each W dependent constraint                        
    W_constraints = Dict{Symbol, Matrix{JuMP.ConstraintRef}}()
    h_constraints = Dict{Symbol, Vector{JuMP.ConstraintRef}}()
    W = _build_W(n_segments_vec, PWRV_list)
    h = _build_h(PWRV_list)
    nW = size(W, 1)

    # Equality constraints
    if size(ABC.Be, 1) > 0
        Be = _build_B(ABC.Be, n_segments_vec)
        @constraint(model, ABC.Ae * X .== Be)
    end

    # Inequality contraints
    if size(ABC.Bu, 1) > 0
        Bu = _build_B(ABC.Bu, n_segments_vec)
        @variable(model, Su[1:size(Bu, 1), 1:dim_ξ])
        @constraint(model, ABC.Au * X .+ Su .== Bu)
        @variable(model, ΛSu[1:size(Bu, 1),1:nW] >= 0)

        #W, h dependence
        W_constraints[:upper_ineq] = @constraint(model, ΛSu * W .== Su)
        h_constraints[:upper_ineq] = @constraint(model, ΛSu * h .>= 0)
    end

    if size(ABC.Bl, 1) > 0
        Bl = _build_B(ABC.Bl, n_segments_vec)
        @variable(model, Sl[1:size(Bl, 1), 1:dim_ξ])
        @constraint(model, ABC.Al * X .- Sl .== Bl)
        @variable(model, ΛSl[1:size(Bl, 1),1:nW] >= 0)

        #W, h dependence
        W_constraints[:lower_ineq] = @constraint(model, ΛSl * W .== Sl)
        h_constraints[:lower_ineq] = @constraint(model, ΛSl * h .>= 0)
    end

    # Can only include rows where the bound is not +Inf
    idxs_xu = findall(x -> x != Inf, ABC.xu)
    if size(idxs_xu, 1) > 0
        @variable(model, Sxu[idxs_xu, 1:dim_ξ])
        @constraint(model, X[idxs_xu,1] .+ Sxu[idxs_xu,1] .== ABC.xu[idxs_xu])
        @constraint(model, X[idxs_xu,2:end] .+ Sxu[idxs_xu,2:end] .== 0)

        @variable(model, ΛSxu[idxs_xu,1:nW] .>= 0)

        #W, h dependence
        W_constraints[:upper_x] = @constraint(model, ΛSxu.data * W .== Sxu.data)
        h_constraints[:upper_x] = @constraint(model, ΛSxu.data * h .>= 0)
    end

    # Can only include rows where the bound is not -Inf
    idxs_xl = findall(x -> x != -Inf, ABC.xl)
    if size(idxs_xl, 1) > 0
        @variable(model, Sxl[idxs_xl, 1:dim_ξ])
        @constraint(model, X[idxs_xl,1] .- Sxl[idxs_xl,1] .== ABC.xl[idxs_xl])
        @constraint(model, X[idxs_xl,2:end] .- Sxl[idxs_xl,2:end] .== 0)

        @variable(model, ΛSxl[idxs_xl,1:nW] .>= 0)
        
        #W, h dependence
        W_constraints[:lower_x] = @constraint(model, ΛSxl.data * W .== Sxl.data)
        h_constraints[:lower_x] = @constraint(model, ΛSxl.data * h .>= 0)
    end

    C  = _build_C(ABC.C, n_segments_vec)

    model.ext[:C] = C
    M = _build_second_moment_matrix(n_segments_vec, PWRV_list)

    @expression(model, obj, LinearAlgebra.tr(C' * X * M))

    if ldr_model.ext[:_LDR_sense] == MOI.MIN_SENSE
        @objective(model, Min, obj)
    else
        @objective(model, Max, obj)
    end
    model.ext[:sense] = ldr_model.ext[:_LDR_sense]

    # Fill pwldr struct
    pwldr.model = model
    pwldr.PWRV_list = PWRV_list
    pwldr.W_constraints = W_constraints
    pwldr.h_constraints = h_constraints
    pwldr.reset_model = false
end

"""
    optimize!(
        model::PWLDR
    )

    Optimize the PWLDR problem and build the problem if the flag reset_model
        is true

    # Arguments
    - model::PWLDR: PWLDR problem to be optimized
"""
function optimize!(
    model::PWLDR
)
    if model.reset_model
        _build_problem!(model)
        JuMP.optimize!(model.model)
    else
        JuMP.optimize!(model.model)
    end
end

"""
    objective_value(
        model::PWLDR
    )

    Get the objective value of the problem

    # Arguments
    - model::PWLDR: PWLDR problem to get the objective value

    # Returns
    ::Float64: Objective value of the problem
"""
function objective_value(
    model::PWLDR
)
    return objective_value(model.model)
end

"""
    update_breakpoints!(
        pwldr::PWLDR,
        weight_vec::Vector{Vector{Float64}}
    )

    Update all properties of the PWLDR problem given a new vector of weights for
        the segments

    # Arguments
    - pwldr::PWLDR: PWLDR model to be updated
    - weight_vec::Vector{Vector{Float64}}: Vector of weights for each piecewise
        random variable
"""
function update_breakpoints!(
    pwldr::PWLDR,
    weight_vec::Vector{Vector{Float64}}
)

    function update_W_constraints_inplace!(
        constraints_matrix::Matrix{ConstraintRef},
        variables::Matrix{VariableRef}, 
        new_W::SparseArrays.SparseMatrixCSC{Float64, Int64}
    )
        
        n_sl, n_xi = size(constraints_matrix)
        n_w = size(variables, 2)
        
        for i in 1:n_sl
            for j in 1:n_xi
                con_ref = constraints_matrix[i, j]
                for k in 1:n_w
                    var_ref = variables[i, k]
                    new_coeff = new_W[k, j]
                    JuMP.set_normalized_coefficient(con_ref, var_ref, new_coeff)
                end
                
            end
        end
    end

    function update_h_constraints_inplace!(
        constraints_vector::Vector{ConstraintRef},
        variables::Matrix{VariableRef}, 
        new_h::Vector{Float64}
    )
        n_sl, n_w = size(variables)
        
        for i in 1:n_sl
            con_ref = constraints_vector[i]
            for k in 1:n_w
                var_ref = variables[i, k]
                new_coeff = new_h[k]
                JuMP.set_normalized_coefficient(con_ref, var_ref, new_coeff)
            end
        end
    end

    for i in 1:length(pwldr.PWRV_list)
        update_breakpoints!(pwldr.PWRV_list[i], weight_vec[i])
    end

    W = _build_W(pwldr.n_segments_vec, pwldr.PWRV_list)
    h = _build_h(pwldr.PWRV_list)

    model = pwldr.model

    if haskey(pwldr.W_constraints, :upper_ineq)
        ΛSu = model[:ΛSu]
        update_W_constraints_inplace!(
            pwldr.W_constraints[:upper_ineq], ΛSu, W)

        update_h_constraints_inplace!(
            pwldr.h_constraints[:upper_ineq], ΛSu, h
        )
    end

    if haskey(pwldr.W_constraints, :lower_ineq)
        ΛSl  = model[:ΛSl]

        update_W_constraints_inplace!(
            pwldr.W_constraints[:lower_ineq], ΛSl, W)

        update_h_constraints_inplace!(
            pwldr.h_constraints[:lower_ineq], ΛSl, h
        )
    end

    if haskey(pwldr.W_constraints, :upper_x)
        ΛSxu = model[:ΛSxu].data

        update_W_constraints_inplace!(
            pwldr.W_constraints[:upper_x], ΛSxu, W)

        update_h_constraints_inplace!(
            pwldr.h_constraints[:upper_x], ΛSxu, h
        )
    end

    if haskey(pwldr.W_constraints, :lower_x)
        ΛSxl = model[:ΛSxl].data
        update_W_constraints_inplace!(
            pwldr.W_constraints[:lower_x], ΛSxl, W)

        update_h_constraints_inplace!(
            pwldr.h_constraints[:lower_x], ΛSxl, h
        )

    end

    X = model[:X]
    C = model.ext[:C]

    M = _build_second_moment_matrix(pwldr.n_segments_vec, pwldr.PWRV_list)

    if model.ext[:sense] == MOI.MIN_SENSE
        @objective(model, Min, LinearAlgebra.tr(C' * X * M))
    else
        @objective(model, Max, LinearAlgebra.tr(C' * X * M))
    end
end

function evaluate_sample(
    PWRV_list,
    X,
    C,
    samples
)
    ξ = [1.0]
    for (pwvr, sp) in zip(PWRV_list, samples)
        ξ_ext = sample_vector(pwvr, sp)
        append!(ξ, ξ_ext)
    end
    value_ret = (C * ξ)' * (X * ξ)
    return value_ret
end

function getindex(
    model::PWLDR,
    indice::Symbol
)
    return getindex(model.model, indice)
end

function get_decision(
    pwldr::PWLDR,
    x::JuMP.VariableRef
)
    @assert !haskey(pwldr.uncertainty_to_distribution, x)
    model = pwldr.ldr_model
    var_to_column = model.ext[:_LDR_var_to_column]
    column_to_canonical = model.ext[:_LDR_column_to_canonical]
    i = column_to_canonical[var_to_column[x]]
    return value(pwldr.model[:X][i,1])
end

function get_decision(
    pwldr::PWLDR,
    x::JuMP.VariableRef,
    uncertainty_variable::JuMP.VariableRef
)
    @assert !haskey(pwldr.uncertainty_to_distribution, x)
    @assert haskey(pwldr.uncertainty_to_distribution, uncertainty_variable)
    model = pwldr.ldr_model
    var_to_column = model.ext[:_LDR_var_to_column]
    column_to_canonical = model.ext[:_LDR_column_to_canonical]
    i = column_to_canonical[var_to_column[x]]
    j_idx, _ = pwldr.uncertainty_to_distribution[uncertainty_variable]
    j_init = 1 + j_idx + sum(pwldr.n_segments_vec[1:j_idx - 1])
    j_end = j_init + pwldr.n_segments_vec[j_idx] - 1
    return (
        breakpoints = pwldr.PWRV_list[j_idx].η_vec,
        decision_vec = value.(pwldr.model[:X][i, j_init:j_end])
    )
end

"""
    _segments_number(
        ldr_model::LinearDecisionRules.LDRModel;
        fix_n::Int = 1
    )

    Build a vector with fixed segments number for each random variable of the
        ldr model

    # Arguments
    ldr_model::LinearDecisionRules.LDRModel: Original LDR problem
    fix_n::Int = 1: Number to be fixed

    # Returns
    ::Vector{Int}: Respective segments_number vector
"""
function _segments_number(
    ldr_model::LinearDecisionRules.LDRModel;
    fix_n::Int = 1
)
    ABC = ldr_model.ext[:_LDR_ABC]
    dim_ξ_ldr = size(ABC.Be, 2)

    n_segments_vec = ones(Int, dim_ξ_ldr - 1) * fix_n
    return n_segments_vec
end

"""
    set_breakpoint!(
        pwldr::PWLDR,
        variable::JuMP.VariableRef,
        n_breakpoints::Int
    )

    Set the number of breakpoints for the respective variable

    # Arguments
    - pwldr::PWLDR: PWLDR model
    - variable::JuMP.VariableRef: Name of the variable to be changed
    - n_breakpoints::Int: Number of breakpoints to be setted
"""
function set_breakpoint!(
    pwldr::PWLDR,
    variable::JuMP.VariableRef,
    n_breakpoints::Int
)
    dist_idx, inner_idx = pwldr.uncertainty_to_distribution[variable]
    pwldr.n_segments_vec[dist_idx] = n_breakpoints + 1
    pwldr.reset_model = true
end

"""
    PWLDR(
        ldr_model::LinearDecisionRules.LDRModel
    )

    Given a linear decision rule model, build the minimum structure to transform
        into a piecewise linear model
    
    # Arguments
    - ldr_model::LinearDecisionRules.LDRModel: Original ldr model

    # Returns
    ::PWLDR: minimum structure of piecewise model
"""
function PWLDR(
    ldr_model::LinearDecisionRules.LDRModel
)
    
    n_segments_vec = _segments_number(ldr_model)

    #Build empty model
    empty_model = PWLDR(JuMP.Model(),
                        ldr_model,
                        PWVR[],
                        n_segments_vec,
                        Dict{Symbol, Matrix{JuMP.ConstraintRef}}(),
                        Dict{Symbol, Vector{JuMP.ConstraintRef}}(),
                        true,
                        ldr_model.cache_model.uncertainty_to_distribution)

    return empty_model
end

