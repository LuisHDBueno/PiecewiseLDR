include("../PiecewiseLDR.jl")
using .PiecewiseLDR

using Distributions
using JuMP
using LinearDecisionRules
using CSV
using DataFrames
using HiGHS
using Random
using StaticArrays

include("train_problems/problem_setup.jl")
include("train_problems/shipment_planning.jl")
include("train_problems/capacity_expansion.jl")
include("train_problems/network_flow_allocation.jl")

function dr_ldr_calc(ldr, sample_list)
    X = value.(ldr.primal_model[:X])
    C = ldr.ext[:_LDR_ABC].C
    sum = 0
    for sample in sample_list
        ξ = [1.0]
        append!(ξ, sample)
        sum += (C * ξ)' * (X * ξ)
    end
    return sum/length(sample_list)
end

function dr_pwldr_calc(pwldr, sample_list)
    sum = 0
    X = value.(pwldr.model[:X])
    C = pwldr.model.ext[:C]
    for sample in sample_list
        sum += PiecewiseLDR.evaluate_sample(pwldr.PWVR_list, X, C, sample)
    end
    return sum/length(sample_list)
end

function ProblemInstancePWLDR(
    problem,
    pwldr::PiecewiseLDR.PWLDR
)
    optimize!(pwldr)
    obj_value = objective_value(pwldr)
    X = pwldr.model[:X]
    first_stage_index = sort(collect(pwldr.model.ext[:first_stage_index]))
    first_stage_decision = value.(X[first_stage_index,1])
    return ProblemInstance(problem, first_stage_decision, pwldr, obj_value, 0)
end

function get_data(
    problem_setup::ProblemSetup,
    dist_list::Vector{Distributions.Distribution{Univariate, Continuous}},
    max_bp::Int,
    n_samples_train::Int,
    n_samples_test::Int,
    n_problems::Int,
    optimizer
)
    Random.seed!(1234)
    displace_function = [("black_box", PiecewiseLDR.black_box!),
                        ("local_search", PiecewiseLDR.local_search!)]
    list_bp = [i for i in 1:max_bp]

    regular_metrics = DataFrame(
        idx_p = Int[],
        metric = String[],
        value = Float64[],
        time = Float64[],
        first_stage_decision = String[]
    )

    vector_cols = [Symbol("v$i") for i in 1:15]
    vector_types = [Float64[] for _ in 1:15]
    pwldr_metadata = DataFrame(
        :idx_p => Int[],
        :idx_v => Int[],
        :variable => String[],
        (vector_cols .=> vector_types)...
    )

    pwldr_metrics = DataFrame(
        idx_p = Int[],
        idx_v = Int[],
        nb = Int[],
        displace_func = String[],
        metric = String[],
        value_uni = Float64[],
        value_opt = Float64[],
        time_uni = Float64[],
        time_opt = Float64[]
    )

    for idx_p in 1:n_problems
        problem = problem_setup.gen_metadata(dist_list, n_samples_train,
                                            n_samples_test, optimizer)

        # Standart Model
        init_time = time()
        std = problem_setup.std(problem)
        end_time = time()
        push!(regular_metrics, (idx_p = idx_p, value = std.objective_value,
                                metric ="obj_std", time = end_time - init_time,
                                first_stage_decision = string(std.first_stage_decision)),
                                promote = true)

        init_time = time()
        reoptm_std = problem_setup.second_stage(std, problem.samples_test)
        end_time = time()
        push!(regular_metrics, (idx_p = idx_p, value = reoptm_std,
                                metric = "reopt_std", time = end_time - init_time,
                                first_stage_decision = ""),
                                promote = true)

        # Determinist Model
        init_time = time()
        deterministic = problem_setup.deterministic(problem)
        end_time = time()
        push!(regular_metrics, (idx_p = idx_p, value = deterministic.objective_value,
                                metric = "deterministic", time = end_time - init_time,
                                first_stage_decision = string(deterministic.first_stage_decision)),
                                promote = true)

        init_time = time()
        reoptm_deterministic = problem_setup.second_stage(deterministic, problem.samples_test)
        end_time = time()                       
        push!(regular_metrics, (idx_p = idx_p, value = reoptm_deterministic,
                                metric = "reopt_deterministic", time = end_time - init_time,
                                first_stage_decision = ""),
                                promote = true)

        # Wait-and-see
        init_time = time()
        ws = problem_setup.ws(problem)
        end_time = time()
        push!(regular_metrics, (idx_p = idx_p, value = ws,
                                metric = "ws", time = end_time - init_time,
                                first_stage_decision = ""),
                                promote = true)

        # LDR
        init_time = time()
        ldr_model = problem_setup.ldr(problem)
        end_time = time()
        push!(regular_metrics, (idx_p = idx_p, value = ldr_model.objective_value,
                                metric = "obj_ldr", time = end_time - init_time,
                                first_stage_decision = string(ldr_model.first_stage_decision)),
                                promote = true)

        init_time = time()
        reoptm_ldr = problem_setup.second_stage(ldr_model, problem.samples_test)
        end_time = time()
        push!(regular_metrics, (idx_p = idx_p, value = reoptm_ldr,
                                metric = "reopt_ldr", time = end_time - init_time,
                                first_stage_decision = ""),
                                promote = true)

        init_time = time()
        dr_ldr = dr_ldr_calc(ldr_model.model, problem.samples_test)
        end_time = time()
        push!(regular_metrics, (idx_p = idx_p, value = dr_ldr,
                                metric = "dr_ldr", time = end_time - init_time,
                                first_stage_decision = ""),
                                promote = true)

        name = problem_setup.name
        CSV.write("data/$(name)_regular_metrics.csv", regular_metrics)
        println("Checkpoint regular metrics $name, problem: $idx_p")

        #PWLDR
        pwldr = PiecewiseLDR.PWLDR(ldr_model.model)
        optimize!(pwldr)
        for (idx_v, variable) in enumerate(keys(pwldr.uncertainty_to_distribution))
            V = PiecewiseLDR.vector_representation(pwldr, variable)
            v_data = (; zip(vector_cols, V)...)
            push!(pwldr_metadata,
                    (idx_p = idx_p,
                    idx_v = idx_v,
                    variable = pwldr.PWVR_list[idx_v].distribution,
                    v_data...) , promote = true
                )
            CSV.write("data/$(name)_pwldr_metadata.csv", pwldr_metadata)
            println("Checkpoint pwldr metadata $name, problem: $idx_p, variable: $idx_v")

            for nb in list_bp
                PiecewiseLDR.set_breakpoint!(pwldr, variable, nb)
                init_time = time()
                optimize!(pwldr)
                end_time = time()
                time_uni = end_time - init_time
                obj_uni = objective_value(pwldr)
                dr_uni = dr_pwldr_calc(pwldr, problem.samples_test)
                pwldr_inst = ProblemInstancePWLDR(problem, pwldr)
                reopt_uni = problem_setup.second_stage(pwldr_inst, problem.samples_test)

                for (func_name, func) in displace_function
                    init_time = time()
                    func(pwldr)
                    end_time = time()
                    optimize!(pwldr)
                    obj_opt = objective_value(pwldr)
                    dr_opt = dr_pwldr_calc(pwldr, problem.samples_test)
                    pwldr_inst = ProblemInstancePWLDR(problem, pwldr)
                    reopt_opt = problem_setup.second_stage(pwldr_inst, problem.samples_test)

                    push!(pwldr_metrics,
                        (idx_p = idx_p, idx_v = idx_v, nb = nb,
                        displace_func = func_name, metric = "obj_pwldr",
                        value_uni = obj_uni, value_opt = obj_opt,
                        time_opt = end_time - init_time, time_uni = time_uni
                        ), promote = true)
                    push!(pwldr_metrics,
                        (idx_p = idx_p, idx_v = idx_v, nb = nb,
                        displace_func = func_name, metric = "dr_pwldr",
                        value_uni = dr_uni, value_opt = dr_opt,
                        time_opt = end_time - init_time, time_uni = time_uni
                        ), promote = true)
                    push!(pwldr_metrics,
                        (idx_p = idx_p, idx_v = idx_v, nb = nb,
                        displace_func = func_name, metric = "reopt_pwldr",
                        value_uni = reopt_uni, value_opt = reopt_opt,
                        time_opt = end_time - init_time, time_uni = time_uni
                        ), promote = true)

                    CSV.write("data/$(name)_pwldr_metrics.csv", pwldr_metrics)
                    println("Checkpoint pwldr $name, problem: $idx_p, variable: $idx_v, nb: $nb, func: $func_name")
            
                end
            end
            PiecewiseLDR.set_breakpoint!(pwldr, variable, 0)
        end
    end
end

dist_list = [
    # 1. Uniform, é o baseline
    Uniform(10, 90),


    # 2. Normal Centrada, mas com ALTA variância (mais risco nos extremos)
    truncated(Normal(50, 15), 10, 90),

    # 3. Bimodal (Dois picos): testa cenários "ou é baixo ou é alto"
    MixtureModel([
        truncated(Normal(30, 8), 10, 90),  # Componente Baixa
        truncated(Normal(70, 8), 10, 90)   # Componente Alta
    ]),

    # 4. Forma de "U" (Extremos): o oposto da Normal, valores são raramente no centro
    # (Criada truncando uma Normal com variância muito alta)
    truncated(Normal(50, 40), 10, 90)
]

get_data(ShipmentPlanningSetup, dist_list, 10, 200, 2000, 30, HiGHS.Optimizer)
get_data(CapacityExpansionSetup, dist_list, 10, 200, 2000, 30, HiGHS.Optimizer)
get_data(NetworkFlowAllocationSetup, dist_list, 10, 200, 2000, 30, HiGHS.Optimizer)