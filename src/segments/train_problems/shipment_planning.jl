struct ShipmentPlanningMetadata
    n_products::Int
    n_clients::Int
    prod_cost_1::Vector{Float64}
    prod_cost_2::Vector{Float64}
    client_cost::Matrix{Float64}
    demand_dist::Vector{Distribution{Univariate, Continuous}}
    samples_train
    samples_test
    optimizer
end

function sp_second_stage(
    problem_inst::ProblemInstance,
    samples_test
)

    problem = problem_inst.metadata
    # Solve with fixed first stage
    fixed_cost = sum(problem.prod_cost_1 .* problem_inst.first_stage_decision)

    # Create model
    recourse_model = JuMP.Model(problem.optimizer)
    set_silent(recourse_model)

    @variable(recourse_model, ship[1:problem.n_products, 1:problem.n_clients] .>= 0)
    @variable(recourse_model, buy_2[1:problem.n_products] .>= 0)

    for i in 1:problem.n_products
        @constraint(recourse_model, sum(ship[i, j] for j in 1:problem.n_clients) <= problem_inst.first_stage_decision[i] + buy_2[i])
    end

    demand_constraints = Vector{ConstraintRef}(undef, problem.n_clients)
    for j in 1:problem.n_clients
        demand_constraints[j] = @constraint(recourse_model, sum(ship[i, j] for i in 1:problem.n_products) >= 0.0)
    end

    @objective(recourse_model, Min,
            sum(problem.prod_cost_2 .* buy_2) +
            sum(sum(problem.client_cost .* ship))
    )

    total_cost = 0.0
    for sample in samples_test
        for j in 1:problem.n_clients
            set_normalized_rhs(demand_constraints[j], sample[j])
        end
        optimize!(recourse_model)

        scenario_cost = fixed_cost + objective_value(recourse_model)
        total_cost += scenario_cost
    end

    return total_cost / length(samples_test)
end

function sp_ldr(
    problem::ShipmentPlanningMetadata
)
    ldr = LinearDecisionRules.LDRModel(problem.optimizer)
    set_silent(ldr)

    @variable(ldr, buy_1[1:problem.n_products] .>= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, buy_2[1:problem.n_products] .>= 0)
    @variable(ldr, ship[1:problem.n_products, 1:problem.n_clients] .>= 0)

    @variable(ldr, demand[i = 1:problem.n_clients] in LinearDecisionRules.ScalarUncertainty(problem.demand_dist[i]))

    for j in 1:problem.n_clients
        @constraint(ldr, sum(ship[i, j] for i in 1:problem.n_products) >= demand[j])
    end
    for i in 1:problem.n_products
        @constraint(ldr, sum(ship[i, j] for j in 1:problem.n_clients) <= buy_1[i] + buy_2[i])
    end

    @objective(ldr, Min,
                + sum(problem.prod_cost_1 .* buy_1)
                + sum(problem.prod_cost_2 .* buy_2)
                + sum(sum(problem.client_cost .* ship)))
            
    optimize!(ldr)

    first_stage_decision = [LinearDecisionRules.get_decision(ldr, buy_1[i]) for i in 1:problem.n_products]
    obj_value = objective_value(ldr)

    return ProblemInstance(problem, first_stage_decision, ldr, obj_value, 0)
end

function sp_standard_form(
    problem::ShipmentPlanningMetadata
)
    S = length(problem.samples_train)

    model = Model(problem.optimizer)
    set_silent(model)

    # 1 Stage
    @variable(model, buy_1[1:problem.n_products] >= 0)

    # 2 Stage
    @variable(model, buy_2[1:problem.n_products, 1:S] >= 0)
    @variable(model, ship[1:problem.n_products, 1:problem.n_clients, 1:S] >= 0)

    for s in 1:S
        d_s = problem.samples_train[s]

        for j in 1:problem.n_clients
            @constraint(model, sum(ship[i,j,s] for i in 1:problem.n_products) >= d_s[j])
        end

        for i in 1:problem.n_products
            @constraint(model, sum(ship[i,j,s] for j in 1:problem.n_clients) <= buy_1[i] + buy_2[i,s])
        end
    end

    expr_first = sum(problem.prod_cost_1[i] * buy_1[i] for i in 1:problem.n_products)

    expr_second = 0.0
    probs = fill(1.0/S, S)
    for s in 1:S
        p = probs[s]
        expr_second += p * (
            sum(problem.prod_cost_2[i] * buy_2[i,s] for i in 1:problem.n_products) +
            sum(problem.client_cost[i,j] * ship[i,j,s] for i in 1:problem.n_products, j in 1:problem.n_clients)
        )
    end

    @objective(model, Min, expr_first + expr_second)

    optimize!(model)
    first_stage_decision = value.(model[:buy_1])
    obj_value = objective_value(model)

    return ProblemInstance(problem, first_stage_decision, model, obj_value, 0)
end

function sp_deterministic(
    problem::ShipmentPlanningMetadata
)
    model = JuMP.Model(problem.optimizer)
    set_silent(model)

    @variable(model, buy_1[1:problem.n_products] .>= 0)
    @variable(model, buy_2[1:problem.n_products] .>= 0)
    @variable(model, ship[1:problem.n_products, 1:problem.n_clients] .>= 0)

    for j in 1:problem.n_clients
        demand_j = Distributions.mean(problem.demand_dist[j])
        @constraint(model, sum(ship[i, j] for i in 1:problem.n_products) >= demand_j)
    end
    for i in 1:problem.n_products
        @constraint(model, sum(ship[i, j] for j in 1:problem.n_clients) <= buy_1[i] + buy_2[i])
    end

    @objective(model, Min,
        sum(problem.prod_cost_1 .* buy_1) +
        sum(problem.prod_cost_2 .* buy_2) +
        sum(sum(problem.client_cost .* ship))
    )

    optimize!(model)
    first_stage_decision = value.(model[:buy_1])
    obj_value = objective_value(model)

    return ProblemInstance(problem, first_stage_decision, model, obj_value, 0)
end

function sp_ws(
    problem::ShipmentPlanningMetadata
)
    samples_test = problem.samples_test
    # Wait and see
    model = JuMP.Model(problem.optimizer)
    set_silent(model)

    @variable(model, buy_1[1:problem.n_products] .>= 0)

    @variable(model, buy_2[1:problem.n_products] .>= 0)

    @variable(model, ship[1:problem.n_products, 1:problem.n_clients] .>= 0)

    for i in 1:problem.n_products
        @constraint(model, sum(ship[i, j] for j in 1:problem.n_clients) <= buy_1[i] + buy_2[i])
    end

    demand_constraints = Vector{ConstraintRef}(undef, problem.n_clients)
    for j in 1:problem.n_clients
        demand_constraints[j] = @constraint(model, sum(ship[i, j] for i in 1:problem.n_products) >= 0.0)
    end

    @objective(model, Min,
                + sum(problem.prod_cost_1 .* buy_1)
                + sum(problem.prod_cost_2 .* buy_2)
                + sum(sum(problem.client_cost .* ship)))
    total = 0
    for sample in samples_test
        for j in 1:problem.n_clients
            set_normalized_rhs(demand_constraints[j], sample[j])
        end
        
        optimize!(model)
        total += objective_value(model)
    end

    return total/length(samples_test)
end

function sp_gen_metadata(
    dist_list::Vector{Distribution{Univariate, Continuous}},
    n_samples_train::Int,
    n_samples_test::Int,
    optimizer
)
    n_products = 5
    n_clients = size(dist_list, 1)

    prod_cost_1 = rand(Uniform(75, 100), n_products)
    prod_cost_2 = prod_cost_1 .* 2
    client_cost = rand(Uniform(30, 50), n_products, n_clients)

    demand_dist = shuffle(dist_list)

    samples_demand_train = [rand.(demand_dist) for _ in 1:n_samples_train]
    samples_demand_test = [rand.(demand_dist) for _ in 1:n_samples_test]
    return ShipmentPlanningMetadata(
            n_products, n_clients, prod_cost_1, prod_cost_2, client_cost,
            demand_dist, samples_demand_train, samples_demand_test, optimizer
            )
end

ShipmentPlanningSetup = ProblemSetup("shipment_planning",
                            sp_gen_metadata, sp_second_stage, sp_ldr, sp_ws,
                            sp_standard_form, sp_deterministic
                            )
