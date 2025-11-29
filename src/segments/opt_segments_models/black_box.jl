function _evaluate_segments(
    weights_limits_list::Vector{Float64},
    pwldr_model::PWLDR,
    sense::Int)

    weight_vec = Vector{Vector{Float64}}()
    weight_index = 1
    for i in 1:length(pwldr_model.PWRV_list)
        push!(weight_vec,
            weights_limits_list[weight_index:weight_index + Int(pwldr_model.n_segments_vec[i] - 1)]
            )
        weight_index += pwldr_model.n_segments_vec[i]
    end

    update_breakpoints!(pwldr_model, weight_vec)
    optimize!(pwldr_model)

    return sense * objective_value(pwldr_model)
end

function black_box!(
    pwldr_model::PWLDR
)
    η_weights_list = Vector{Tuple{Float64, Float64}}()
    for n in pwldr_model.n_segments_vec
        for _ in 1:n
            # Pesos de cada um dos limites
            push!(η_weights_list, (0.01, 1.0))
        end
    end

    if pwldr_model.model.ext[:sense] == MOI.MIN_SENSE
        sense = 1
    else
        sense = -1
    end
    max_eval = 200
    obj_func = hyperparam -> _evaluate_segments(hyperparam, pwldr_model, sense)
    res = bboptimize(obj_func, SearchRange = η_weights_list, MaxFuncEvals = max_eval)
    _evaluate_segments(best_candidate(res), pwldr_model, sense)
end
