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

ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
set_silent(ldr)

@variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
@variable(ldr, sell >= 0)
@variable(ldr, ret >= 0)
@variable(ldr, demand in LinearDecisionRules.Uncertainty(
        distribution = Uniform(demand_min, demand_max)
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