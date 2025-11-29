using Distributions
using JuMP
using HiGHS
using LinearDecisionRules

include("../PiecewiseLDR.jl")
using .PiecewiseLDR

buy_cost = 10
return_value = 8
sell_value = 15

demand_max = 120
demand_min = 80
demand_dist = Uniform(demand_min, demand_max)

ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
set_silent(ldr)

@variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
@variable(ldr, sell >= 0)
@variable(ldr, ret >= 0)
@variable(ldr, demand in LinearDecisionRules.Uncertainty(
        distribution = demand_dist
    )
)

@constraint(ldr, sell + ret <= buy)
@constraint(ldr, sell <= demand)

@objective(ldr, Max,
    - buy_cost * buy
    + return_value * ret
    + sell_value * sell
)
optimize!(ldr)
buy_ldr = LinearDecisionRules.get_decision(ldr, buy)
@show objective_value(ldr)

model = PiecewiseLDR.PWLDR(ldr)
PiecewiseLDR.set_breakpoint!(model, demand, 2)
optimize!(model)
@show objective_value(model)

PiecewiseLDR.local_search!(model)
optimize!(model)

@show objective_value(model)
@show PiecewiseLDR.get_decision(model, sell)
@show PiecewiseLDR.get_decision(model, ret)
@show PiecewiseLDR.get_decision(model, buy)

@show PiecewiseLDR.get_decision(model, sell, demand)
@show PiecewiseLDR.get_decision(model, ret, demand)
@show PiecewiseLDR.get_decision(model, buy, demand)
buy_pwldr = PiecewiseLDR.get_decision(model, buy)

model_d = JuMP.Model(HiGHS.Optimizer)
set_silent(model_d)

@variable(model_d, buy >= 0)
@variable(model_d, sell >= 0)
@variable(model_d, ret >= 0)

@constraint(model_d, sell + ret <= buy)
@constraint(model_d, sell <= mean(demand_dist))

@objective(model_d, Max,
    - buy_cost * buy
    + return_value * ret
    + sell_value * sell
)
optimize!(model_d)

@show objective_value(model_d)
buy_d = value(model_d[:buy])

function newsvendor_2(
    buy,
    samples_test
)
    model_2 = JuMP.Model(HiGHS.Optimizer)
    set_silent(model_2)

    @variable(model_2, sell >= 0)
    @variable(model_2, ret >= 0)

    @constraint(model_2, sell + ret <= buy)
    constr = @constraint(model_2, sell <= mean(demand_dist))

    @objective(model_2, Max,
        - buy_cost * buy
        + return_value * ret
        + sell_value * sell
    )
    total_cost = 0.0
    for sample in samples_test
        set_normalized_rhs(constr, sample)
        optimize!(model_2)

        total_cost += objective_value(model_2)
    end

    return total_cost / length(samples_test)
end

sample_test = rand(demand_dist, 400)
@show "deterministic"
@show newsvendor_2(buy_d, sample_test)
@show "ldr"
@show newsvendor_2(buy_ldr, sample_test)
@show "pwldr"
@show newsvendor_2(buy_pwldr, sample_test)