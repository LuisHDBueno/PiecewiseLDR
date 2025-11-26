struct NFAMetadata
    n_warehouses::Int
    n_customers::Int
    invest_cost::Vector{Float64}
    penalty_cost::Vector{Float64}
    efficiency::Matrix{Float64}
    demand_dist::Vector{Distribution{Univariate, Continuous}} 
    op_cost_dist::Matrix{Distribution{Univariate, Continuous}}
    samples_train
    samples_test
    optimizer
end

function nfa_second_stage(
    problem_inst::ProblemInstance,
    samples_test
)
    problem = problem_inst.metadata

    # Solve with fixed first stage
    fixed_cost = sum(problem.invest_cost .* problem_inst.first_stage_decision)

    recourse_model = JuMP.Model(problem.optimizer)
    set_silent(recourse_model)

    @variable(recourse_model, flow[1:problem.n_warehouses, 1:problem.n_customers] >= 0)
    @variable(recourse_model, shortfall[1:problem.n_customers] >= 0)

    demand_constraint = Vector{ConstraintRef}(undef, problem.n_customers)
    for j in 1:problem.n_customers
        demand_constraint[j] = @constraint(recourse_model, 
            sum(flow[i, j] for i in 1:problem.n_warehouses) + shortfall[j] >= 0.0
        )
    end

    for i in 1:problem.n_warehouses
        @constraint(recourse_model,
            sum(problem.efficiency[i, j] * flow[i, j] for j in 1:problem.n_customers) <= problem_inst.first_stage_decision[i]
        )
    end

    @objective(recourse_model, Min, 0.0)

    total_cost = 0.0
    n_d = problem.n_customers
    n_o = problem.n_warehouses * problem.n_customers
    for sample in samples_test 
        demand_scenario = sample[1:n_d]
        op_cost_scenario_flat = sample[n_d + 1 : n_d + n_o]
        op_cost_scenario = reshape(op_cost_scenario_flat, (problem.n_warehouses, problem.n_customers))

        for j in 1:problem.n_customers
            set_normalized_rhs(demand_constraint[j], demand_scenario[j])
        end
        
        expr = sum(problem.penalty_cost .* shortfall) +
               sum(op_cost_scenario .* flow)

        set_objective_function(recourse_model, expr)
        optimize!(recourse_model)
        
        scenario_cost = fixed_cost + objective_value(recourse_model)
        total_cost += scenario_cost
    end

    return total_cost / length(samples_test)
end

function nfa_ldr(
    problem::NFAMetadata
)
    ldr = LinearDecisionRules.LDRModel(problem.optimizer)
    set_silent(ldr)

    @variable(ldr, capacity[1:problem.n_warehouses] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, flow[1:problem.n_warehouses, 1:problem.n_customers] >= 0)
    @variable(ldr, shortfall[1:problem.n_customers] >= 0)

    @variable(ldr, demand[j = 1:problem.n_customers] 
        in LinearDecisionRules.ScalarUncertainty(problem.demand_dist[j]))
    
    @variable(ldr, op_cost[i = 1:problem.n_warehouses, j = 1:problem.n_customers]
        in LinearDecisionRules.ScalarUncertainty(problem.op_cost_dist[i, j]))

    for j in 1:problem.n_customers
        @constraint(ldr, 
            sum(flow[i, j] for i in 1:problem.n_warehouses) + shortfall[j] >= demand[j]
        )
    end

    for i in 1:problem.n_warehouses
        @constraint(ldr,
            sum(problem.efficiency[i, j] * flow[i, j] for j in 1:problem.n_customers) <= capacity[i]
        )
    end
    
    @objective(ldr, Min,
        sum(problem.invest_cost .* capacity) +
        sum(problem.penalty_cost .* shortfall) +
        sum(op_cost .* flow)
    )

    optimize!(ldr)

    first_stage_decision = [LinearDecisionRules.get_decision(ldr, capacity[i]) for i in 1:problem.n_warehouses]
    obj_value = objective_value(ldr)

    return ProblemInstance(problem, first_stage_decision, ldr, obj_value, 0)
end

function nfa_standard_form(
    problem::NFAMetadata
)
    S = length(problem.samples_train)

    model = Model(problem.optimizer)
    set_silent(model)

    @variable(model, capacity[1:problem.n_warehouses] >= 0)

    @variable(model, flow[1:problem.n_warehouses, 1:problem.n_customers, 1:S] >= 0)
    @variable(model, shortfall[1:problem.n_customers, 1:S] >= 0)                    

    n_d = problem.n_customers
    n_o = problem.n_warehouses * problem.n_customers
    
    for (s, sample) in enumerate(problem.samples_train)
        demand_scenario = sample[1:n_d]
        
        for j in 1:problem.n_customers
            @constraint(model, 
                sum(flow[i, j, s] for i in 1:problem.n_warehouses) + shortfall[j, s] >= demand_scenario[j]
            )
        end

        for i in 1:problem.n_warehouses
            @constraint(model,
                sum(problem.efficiency[i, j] * flow[i, j, s] for j in 1:problem.n_customers) <= capacity[i]
            )
        end
    end

    expr_first = sum(problem.invest_cost .* capacity)
    expr_second = 0.0
    p = 1.0 / S

    for (s, sample) in enumerate(problem.samples_train)
        op_cost_scenario_flat = sample[n_d + 1 : n_d + n_o]
        op_cost_scenario = reshape(op_cost_scenario_flat, (problem.n_warehouses, problem.n_customers))

        expr_second += p * (
            sum(problem.penalty_cost .* shortfall[1:end, s]) +
            sum(op_cost_scenario .* flow[1:end, 1:end, s])
        )
    end

    @objective(model, Min, expr_first + expr_second)
    optimize!(model)
    
    first_stage_decision = value.(model[:capacity])
    obj_value = objective_value(model)

    return ProblemInstance(problem, first_stage_decision, model, obj_value, 0)
end

function nfa_deterministic(
    problem::NFAMetadata
)
    model = JuMP.Model(problem.optimizer)
    set_silent(model)

    @variable(model, capacity[1:problem.n_warehouses] >= 0)
    @variable(model, flow[1:problem.n_warehouses, 1:problem.n_customers] >= 0)
    @variable(model, shortfall[1:problem.n_customers] >= 0) 

    mean_demands = Distributions.mean.(problem.demand_dist)
    for j in 1:problem.n_customers
        @constraint(model, 
            sum(flow[i, j] for i in 1:problem.n_warehouses) + shortfall[j] >= mean_demands[j]
        )
    end

    for i in 1:problem.n_warehouses
        @constraint(model,
            sum(problem.efficiency[i, j] * flow[i, j] for j in 1:problem.n_customers) <= capacity[i]
        )
    end

    mean_op_costs = Distributions.mean.(problem.op_cost_dist)
    @objective(model, Min,
        sum(problem.invest_cost .* capacity) +
        sum(mean_op_costs .* flow) +
        sum(problem.penalty_cost .* shortfall)
    )

    optimize!(model)
    first_stage_decision = value.(model[:capacity])
    obj_value = objective_value(model)

    return ProblemInstance(problem, first_stage_decision, model, obj_value, 0)
end

function nfa_ws(
    problem::NFAMetadata
)
    samples_test = problem.samples_test
    model = JuMP.Model(problem.optimizer)
    set_silent(model)

    @variable(model, capacity[1:problem.n_warehouses] >= 0)
    
    @variable(model, flow[1:problem.n_warehouses, 1:problem.n_customers] >= 0)
    @variable(model, shortfall[1:problem.n_customers] >= 0)

    for i in 1:problem.n_warehouses
        @constraint(model,
            sum(problem.efficiency[i, j] * flow[i, j] for j in 1:problem.n_customers) <= capacity[i]
        )
    end

    demand_constraint = Vector{ConstraintRef}(undef, problem.n_customers)
    for j in 1:problem.n_customers
        demand_constraint[j] = @constraint(model, 
            sum(flow[i, j] for i in 1:problem.n_warehouses) + shortfall[j] >= 0.0
        )
    end

    expr_first = sum(problem.invest_cost .* capacity)
    @objective(model, Min, 0.0)

    total_cost = 0.0
    n_d = problem.n_customers
    n_o = problem.n_warehouses * problem.n_customers
    for sample in samples_test 
        demand_scenario = sample[1:n_d]
        op_cost_scenario_flat = sample[n_d + 1 : n_d + n_o]
        op_cost_scenario = reshape(op_cost_scenario_flat, (problem.n_warehouses, problem.n_customers))

        for j in 1:problem.n_customers
            set_normalized_rhs(demand_constraint[j], demand_scenario[j])
        end
        
        expr_second = sum(problem.penalty_cost .* shortfall) +
                      sum(op_cost_scenario .* flow)

        set_objective_function(model, expr_first + expr_second)
        optimize!(model)
        
        total_cost += objective_value(model)
    end
    return total_cost / length(samples_test)
end

function nfa_gen_metadata(
    dist_list::Vector{Distribution{Univariate, Continuous}},
    n_samples_train::Int,
    n_samples_test::Int,
    optimizer
)

    n_warehouses = 2
    n_customers = size(dist_list, 1)

    demand_dists = shuffle(dist_list)
    op_cost_dists = Matrix{Distribution{Univariate, Continuous}}(undef, n_warehouses, n_customers)
    dist_list_adapt_costs = [
        Uniform(30, 60),
        truncated(Normal(45, 5), 30, 60),
        MixtureModel([
            truncated(Normal(37, 3), 30, 60),
            truncated(Normal(52, 3), 30, 60) 
        ]),

        truncated(Normal(45, 15), 30, 60)
    ]
    for i in 1:n_warehouses
        op_cost_dists[i, :] = shuffle(dist_list_adapt_costs)
    end

    invest_cost = rand(Uniform(500, 750), n_warehouses)
    penalty_cost = fill(1500, n_customers)
    efficiency = rand(Uniform(1, 1.25), n_warehouses, n_customers)

    samples_train = Vector{Vector{Float64}}(undef, n_samples_train)
    for i in 1:n_samples_train
        d_sample = rand.(demand_dists)
        o_sample = rand.(op_cost_dists)

        samples_train[i] = vcat(d_sample, vec(o_sample))
    end

    samples_test = Vector{Vector{Float64}}(undef, n_samples_test)
    for i in 1:n_samples_test
        d_sample = rand.(demand_dists)
        o_sample = rand.(op_cost_dists)
        samples_test[i] = vcat(d_sample, vec(o_sample))
    end

    return NFAMetadata(
        n_warehouses, n_customers, invest_cost, penalty_cost,
        efficiency,
        demand_dists, op_cost_dists,
        samples_train, samples_test, optimizer
    )
end

NetworkFlowAllocationSetup = ProblemSetup("network_flow_allocation",
                                nfa_gen_metadata, nfa_second_stage, nfa_ldr,
                                nfa_ws, nfa_standard_form, nfa_deterministic)