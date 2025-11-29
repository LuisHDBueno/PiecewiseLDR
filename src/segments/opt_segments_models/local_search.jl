function _evaluate_local_search(
    pwldr_model::PWLDR,
    sense::Int
)
    optimize!(pwldr_model) 
    return sense * objective_value(pwldr_model)
end

function local_search!(
    pwldr_model::PWLDR
)

    if pwldr_model.model.ext[:sense] == MOI.MIN_SENSE
        sense = 1
    else
        sense = -1
    end

    value = _evaluate_local_search(pwldr_model, sense)

    weight_vec = Vector{Vector{Float64}}()
    for pwvr in pwldr_model.PWRV_list
        push!(weight_vec, pwvr.weight)
    end

    max_iter = 10000

    iter = 0
    improved = true
    while improved
        
        improved = false

        for (index, pwvr) in enumerate(pwldr_model.PWRV_list)
            step = 1/((pwvr.n_breakpoints + 1) * 10)
            weight = pwvr.weight
            for i in 1:(pwvr.n_breakpoints + 1)
                weight_copy = copy(weight)
                if (weight[i] - step > 0)
                    weight_copy[i] = weight[i] - step
                    weight_vec[index] = weight_copy
                    update_breakpoints!(pwldr_model, weight_vec)
                    value_lower = _evaluate_local_search(pwldr_model, sense)
                    if value_lower < value
                        improved = true
                        value = value_lower
                        weight = weight_copy
                        break
                    end
                end

                if (weight[i] + step <= 1)
                    weight_copy[i] = weight[i] + step
                    weight_vec[index] = weight_copy

                    update_breakpoints!(pwldr_model, weight_vec)
                    value_upper = _evaluate_local_search(pwldr_model, sense)
                    if value_upper < value
                        improved = true
                        value = value_upper
                        weight = weight_copy
                        break
                    end
                end

                weight_vec[index] = weight
            end
        end
        if iter >= max_iter
            improved = false
        end
        iter += 1
    end
    update_breakpoints!(pwldr_model, weight_vec)
end