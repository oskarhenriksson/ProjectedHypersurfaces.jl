# This is case 0 of Example 5.2 in the overleaf file
# Reference: https://arxiv.org/pdf/1903.06126

using Random, LinearAlgebra, ProjectedHypersurfaceRegions
mkpath("./results/3RPRv2")

Random.seed!(12345)

time_start_round1 = time()

# Incidence variety of the discriminant
a2 = 14 // 10;
a3 = 7 // 10;
b3 = 1;
A3 = 9 // 10;
B3 = 6 // 10;
c3 = 1;

@var p[1:2] φ[1:2] c[1:2] A2

f = [φ[1]^2 + φ[2]^2 - 1,
    p[1]^2 + p[2]^2 - 2 * (a3 * p[1] + b3 * p[2]) * φ[1] + 2 * (b3 * p[1] - a3 * p[2]) * φ[2] + a3^2 + b3^2 - c[1],
    p[1]^2 + p[2]^2 - 2 * A2 * p[1] + 2 * ((a2 - a3) * p[1] - b3 * p[2] + A2 * a3 - A2 * a2) * φ[1] + 2 * (b3 * p[1] + (a2 - a3) * p[2] - A2 * b3) * φ[2]
    + (a2 - a3)^2 + b3^2 + A2^2 - c[2],
    p[1]^2 + p[2]^2 - 2 * (A3 * p[1] + B3 * p[2]) + A3^2 + B3^2 - c3
]

Jac = differentiate(f, [p; φ])
F = System([f; det(Jac)], variables=[p; φ; c; A2]) # use System(prod(c)*[f;det(Jac)]) to impose c>0

# Form projected hypersurface
projection_variables = [c; A2]
h = ProjectedHypersurface(F, projection_variables)

# Degree of the discriminant
d = degree(h)
println("Degree of discriminant: $d")

# Set up routing function
center = 5 * rand(length(projection_variables))
write_parameters("./results/3RPRv2/center.txt", center)
r = RoutingFunction(h; c=center);

# Routing points
# pts = read_solutions("./results/3RPRv2/routing_points.txt") |> real
routing_result = critical_points(r)
pts = routing_points(routing_result)
res = trace_result(routing_result)
mon_res = monodromy_result(routing_result)

# Connected components 
partition_result = partition_of_critical_points(r, routing_result)
G = partitions(partition_result)
idx = morse_indices(partition_result)
failures = failed_info(partition_result)

time_end_round1 = time()
println("Computation time for round 1: $(time_end_round1 - time_start_round1) seconds")

function analyze_and_save_result()
    write_parameters("./results/3RPRv2/monodromy_parameters.txt", parameters(mon_res))
    write_solutions("./results/3RPRv2/monodromy_result.txt", solutions(mon_res))
    write_solutions("./results/3RPRv2/result.txt", solutions(res))
    write_solutions("./results/3RPRv2/routing_points.txt", pts)
    write("./results/3RPRv2/connected_components.txt", string(G))

    println("Connected components: $(G)")
    println("Indicies: $(idx)")
    println("Failed info: $(failures)")
    println()

    generate_plot(r, routing_result, partition_result;
        root_counting_system=System(f, variables=vcat(p, φ), parameters=projection_variables)
    )
end

analyze_and_save_result()

# Try another round of monodromy (only if you think the first attempt missed solutions)
println("Running second round of monodromy...")
time_start_round2 = time()
old_number_of_monodromy_solutions = length(solutions(mon_res))

options = MonodromyOptions(
    parameter_sampler=p -> 100 .* randn(ComplexF64, length(p)), # larger loops
    max_loops_no_progress=10 # stopping criterion
)
routing_result = critical_points(r, solutions(mon_res), parameters(mon_res), options=options)
pts = routing_points(routing_result)
res = trace_result(routing_result)
mon_res = monodromy_result(routing_result)

partition_result = partition_of_critical_points(r, routing_result)
G = partitions(partition_result)
idx = morse_indices(partition_result)
failures = failed_info(partition_result)

time_end_round2 = time()
println("Additional computation time for round 2: $(time_end_round2 - time_start_round2) seconds")
println("Total computation time for round 1 and 2: $((time_end_round1 - time_start_round1) + (time_end_round2 - time_start_round2)) seconds")


if length(solutions(mon_res)) > old_number_of_monodromy_solutions
    println("Found new solutions with additional monodromy round!")
    analyze_and_save_result()
else
    println("No new solutions found in the additional monodromy round.")
end
