module TestMain

using Distributions
using JuMP
using LinearDecisionRules
using HiGHS
using Random

include("../src/PiecewiseLDR.jl")
using .PiecewiseLDR

include("../src/segments/train_problems/problem_setup.jl")
include("../src/segments/train_problems/shipment_planning.jl")
include("../src/segments/train_problems/capacity_expansion.jl")
include("../src/segments/train_problems/network_flow_allocation.jl")

using Test

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_build_ldr()
    optimizer = HiGHS.Optimizer
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

    pwldr = PiecewiseLDR.PWLDR(ldr)

    optimize!(pwldr)

    @test objective_value(ldr) == objective_value(pwldr)
    
    X_ldr = value.(ldr.primal_model[:X])
    X_pwldr = value.(pwldr.model[:X])

    @test isapprox(X_ldr, X_pwldr; atol=1e-6)

end

function test_build_distribution_set()
    optimizer = HiGHS.Optimizer
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

    distributions = [
        Uniform(demand_min, demand_max),
        Uniform(demand_min, demand_max),
        Uniform(demand_min, demand_max)
    ]

    @variable(ldr, demand[i = 1:3] in LinearDecisionRules.ScalarUncertainty(distributions[i]), base_name="demand")

    @constraint(ldr, sell + ret <= buy)

    for i in 1:3
        @constraint(ldr, sell <= demand[i])
    end

    @objective(ldr, Max,
        - buy_cost * buy
        + return_value * ret
        + sell_value * sell
    )
    optimize!(ldr)

    pwldr = PiecewiseLDR.PWLDR(ldr)

    optimize!(pwldr)

    @test objective_value(ldr) == objective_value(pwldr)
    
    X_ldr = value.(ldr.primal_model[:X])
    X_pwldr = value.(pwldr.model[:X])

    @test isapprox(X_ldr, X_pwldr; atol=1e-6)
end

function test_build_pwldr()
    optimizer = HiGHS.Optimizer
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

    n_breakpoints = 2
    pwldr = PiecewiseLDR.PWLDR(ldr)
    PiecewiseLDR.set_breakpoint!(pwldr, demand, n_breakpoints)
    optimize!(pwldr)

    @test objective_value(ldr) <= objective_value(pwldr)
    
    LinearDecisionRules.set_attribute(
            demand,
            LinearDecisionRules.BreakPoints(),
            n_breakpoints,)
    optimize!(ldr)
    
    @test objective_value(ldr) == objective_value(pwldr)
    
    X_ldr = value.(ldr.primal_model[:X])
    X_pwldr = value.(pwldr.model[:X])
    @test isapprox(X_ldr, X_pwldr; atol=1e-6)

    C_ldr = ldr.ext[:_LDR_ABC].C
    C_pwldr = pwldr.model.ext[:C]
    @test isapprox(C_ldr, C_pwldr; atol=1e-6)

    M_ldr = ldr.ext[:_LDR_M]
    M_pwldr = PiecewiseLDR._build_second_moment_matrix(pwldr.n_segments_vec, pwldr.PWVR_list)
    @test isapprox(M_ldr, M_pwldr; atol=1e-6)

end

function test_update_breakpoints()

    optimizer = HiGHS.Optimizer
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

    n_breakpoints = 2
    pwldr = PiecewiseLDR.PWLDR(ldr)
    PiecewiseLDR.set_breakpoint!(pwldr, demand, n_breakpoints)
    optimize!(pwldr)

    @test objective_value(ldr) <= objective_value(pwldr)

    # update breakpoints
    η_vec = [80.0, 90.0, 110.0, 120.0]
    PiecewiseLDR.update_breakpoints!(pwldr, [[1.0,2.0,1.0]])

    LinearDecisionRules.set_attribute(
        demand,
        LinearDecisionRules.BreakPoints(),
        η_vec[2:end-1],)

    optimize!(pwldr)
    optimize!(ldr)

    @test isapprox(objective_value(ldr), objective_value(pwldr);atol=1e-6)
    
    X_ldr = value.(ldr.primal_model[:X])
    X_pwldr = value.(pwldr.model[:X])
    @test isapprox(X_ldr, X_pwldr; atol=1e-6)

    C_ldr = ldr.ext[:_LDR_ABC].C
    C_pwldr = pwldr.model.ext[:C]
    @test isapprox(C_ldr, C_pwldr; atol=1e-6)

    M_ldr = ldr.ext[:_LDR_M]
    M_pwldr = PiecewiseLDR._build_second_moment_matrix(pwldr.n_segments_vec, pwldr.PWVR_list)
    @test isapprox(M_ldr, M_pwldr; atol=1e-6)

end

function test_black_box()
    optimizer = HiGHS.Optimizer
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

    n_breakpoints = 2
    pwldr = PiecewiseLDR.PWLDR(ldr)
    PiecewiseLDR.set_breakpoint!(pwldr, demand, n_breakpoints)
    optimize!(pwldr)
    before_opt_displace = objective_value(pwldr)

    PiecewiseLDR.black_box!(pwldr)
    optimize!(pwldr)
    after_opt_displace = objective_value(pwldr)

    @test before_opt_displace <= after_opt_displace
    
end

function test_local_search()
    optimizer = HiGHS.Optimizer
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

    n_breakpoints = 2
    pwldr = PiecewiseLDR.PWLDR(ldr)
    PiecewiseLDR.set_breakpoint!(pwldr, demand, n_breakpoints)
    optimize!(pwldr)
    before_opt_displace = objective_value(pwldr)

    PiecewiseLDR.local_search!(pwldr)
    optimize!(pwldr)
    after_opt_displace = objective_value(pwldr)

    @test before_opt_displace <= after_opt_displace
end

function test_statistics()
    data = [10.0, 2.0, 38.0, 23.0, 38.0, 23.0, 21.0]
    v = PiecewiseLDR.get_statistics(data)

    @test 2.0 == v.min == v[1]
    @test 38.0 == v.max == v[2]
    @test isapprox(22.143, v.mean, atol=1e-3)
    @test v.mean == v[3]
    @test v.median == 23.00 == v[4]
    @test isapprox(12.298, v.std, atol=1e-3)
    @test v.std == v[5]
end

function test_bp_gain()
    optimizer = HiGHS.Optimizer
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

    pwldr = PiecewiseLDR.PWLDR(ldr)
    optimize!(pwldr)

    result = PiecewiseLDR.get_bp_gain(pwldr, demand)
    @show result
end

function test_vector_representation()
    optimizer = HiGHS.Optimizer
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

    pwldr = PiecewiseLDR.PWLDR(ldr)
    optimize!(pwldr)

    result = PiecewiseLDR.vector_representation(pwldr, demand)
    @test result isa Vector{Float64}
    @test size(result) == (15,)
end

function test_shipment_planing()
    # Parameters
    optimizer = HiGHS.Optimizer
    setup = ShipmentPlanningSetup
    dist_list = [
                    Uniform(10, 90),
                    truncated(Normal(50, 15), 10, 90),
                    MixtureModel([
                        truncated(Normal(30, 8), 10, 90),
                        truncated(Normal(70, 8), 10, 90)
                    ]),
                    truncated(Normal(50, 40), 10, 90)
                ]
    n_samples_train = 100
    n_samples_test = 1000
    problem = setup.gen_metadata(dist_list, n_samples_train,
                                            n_samples_test, optimizer)
    std = setup.std(problem)
    reoptm_std = setup.second_stage(std, problem.samples_test)

    deterministic = setup.deterministic(problem)
    reoptm_deterministic = setup.second_stage(deterministic, problem.samples_test)

    ws = setup.ws(problem)

    @test ws <= reoptm_std <= reoptm_deterministic

    ldr_model = setup.ldr(problem)
    ldr_obj = ldr_model.objective_value

    pwldr = PiecewiseLDR.PWLDR(ldr_model.model)
    for (idx_v, variable) in enumerate(keys(pwldr.uncertainty_to_distribution))
        PiecewiseLDR.set_breakpoint!(pwldr, variable, 2)
    end
    optimize!(pwldr)
    pwldr_uni = objective_value(pwldr)
    PiecewiseLDR.local_search!(pwldr)
    optimize!(pwldr)
    pwldr_opt = objective_value(pwldr)
    @test ws <= pwldr_opt <= pwldr_uni < ldr_obj

    # Test implementation
    linear_decision_rules = ldr_model.model
    for (idx_v, variable) in enumerate(keys(linear_decision_rules.cache_model.uncertainty_to_distribution))
        LinearDecisionRules.set_attribute(
            variable,
            LinearDecisionRules.BreakPoints(),
            2,)
    end
    optimize!(linear_decision_rules)
    obj_linear_decision_rules = objective_value(linear_decision_rules)
    @test isapprox(obj_linear_decision_rules, pwldr_uni; atol=1e-6)
end

function test_capacity_expansion()
    # Parameters
    optimizer = HiGHS.Optimizer
    setup = CapacityExpansionSetup
    dist_list = [
                    Uniform(10, 90),
                    truncated(Normal(50, 15), 10, 90),
                    MixtureModel([
                        truncated(Normal(30, 8), 10, 90),
                        truncated(Normal(70, 8), 10, 90)
                    ]),
                    truncated(Normal(50, 40), 10, 90)
                ]
    n_samples_train = 100
    n_samples_test = 1000
    problem = setup.gen_metadata(dist_list, n_samples_train,
                                            n_samples_test, optimizer)
    std = setup.std(problem)
    reoptm_std = setup.second_stage(std, problem.samples_test)

    deterministic = setup.deterministic(problem)
    reoptm_deterministic = setup.second_stage(deterministic, problem.samples_test)

    ws = setup.ws(problem)

    @test ws <= reoptm_std <= reoptm_deterministic

    ldr_model = setup.ldr(problem)
    ldr_obj = ldr_model.objective_value

    pwldr = PiecewiseLDR.PWLDR(ldr_model.model)
    for (idx_v, variable) in enumerate(keys(pwldr.uncertainty_to_distribution))
        PiecewiseLDR.set_breakpoint!(pwldr, variable, 2)
    end
    optimize!(pwldr)
    pwldr_uni = objective_value(pwldr)
    PiecewiseLDR.local_search!(pwldr)
    optimize!(pwldr)
    pwldr_opt = objective_value(pwldr)
    @test ws <= pwldr_opt <= pwldr_uni < ldr_obj

    # Test implementation
    linear_decision_rules = ldr_model.model
    for (idx_v, variable) in enumerate(keys(linear_decision_rules.cache_model.uncertainty_to_distribution))
        LinearDecisionRules.set_attribute(
            variable,
            LinearDecisionRules.BreakPoints(),
            2,)
    end
    optimize!(linear_decision_rules)
    obj_linear_decision_rules = objective_value(linear_decision_rules)
    @test isapprox(obj_linear_decision_rules, pwldr_uni; atol=1e-6)
end

function test_network_flow_allocation()
    # Parameters
    optimizer = HiGHS.Optimizer
    setup = NetworkFlowAllocationSetup
    dist_list = [
                    Uniform(10, 90),
                    truncated(Normal(50, 15), 10, 90),
                    MixtureModel([
                        truncated(Normal(30, 8), 10, 90),
                        truncated(Normal(70, 8), 10, 90)
                    ]),
                    truncated(Normal(50, 40), 10, 90)
                ]
    n_samples_train = 100
    n_samples_test = 1000
    problem = setup.gen_metadata(dist_list, n_samples_train,
                                            n_samples_test, optimizer)
    std = setup.std(problem)
    reoptm_std = setup.second_stage(std, problem.samples_test)

    deterministic = setup.deterministic(problem)
    reoptm_deterministic = setup.second_stage(deterministic, problem.samples_test)

    ws = setup.ws(problem)

    @test ws <= reoptm_std <= reoptm_deterministic

    ldr_model = setup.ldr(problem)
    ldr_obj = ldr_model.objective_value

    pwldr = PiecewiseLDR.PWLDR(ldr_model.model)
    for (idx_v, variable) in enumerate(keys(pwldr.uncertainty_to_distribution))
        PiecewiseLDR.set_breakpoint!(pwldr, variable, 2)
    end
    optimize!(pwldr)
    pwldr_uni = objective_value(pwldr)
    PiecewiseLDR.local_search!(pwldr)
    optimize!(pwldr)
    pwldr_opt = objective_value(pwldr)
    @test ws <= pwldr_opt <= pwldr_uni < ldr_obj

    # Test implementation
    linear_decision_rules = ldr_model.model
    for (idx_v, variable) in enumerate(keys(linear_decision_rules.cache_model.uncertainty_to_distribution))
        LinearDecisionRules.set_attribute(
            variable,
            LinearDecisionRules.BreakPoints(),
            2,)
    end
    optimize!(linear_decision_rules)
    obj_linear_decision_rules = objective_value(linear_decision_rules)
    @test isapprox(obj_linear_decision_rules, pwldr_uni; atol=1e4)
end

#End Module
end

TestMain.runtests()