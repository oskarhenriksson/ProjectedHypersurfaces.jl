# Algebraic formulation of the Kuramoto model with three oscillators
# See, e.g., https://doi.org/10.1063/1.4919696

using Random, Plots, LinearAlgebra, ImplicitPlots, ProjectedHypersurfaces
mkpath("./results/kuramoto");

Random.seed!(12345)

time_start_round1 = time()

### Set up incidence variety of discriminant
@var s[1:2] c[1:2] w[1:2]
freq1 = (s[1] * c[2] - c[1] * s[2]) + (s[1] * 1 - c[1] * 0) - 3 * w[1]
freq2 = (s[2] * c[1] - c[2] * s[1]) + (s[2] * 1 - c[2] * 0) - 3 * w[2]
norm1 = s[1]^2 + c[1]^2 - 1
norm2 = s[2]^2 + c[2]^2 - 1
steady_state = [freq1, freq2, norm1, norm2]
Jac = differentiate.(steady_state, [s; c]')
detJac = expand(det(Jac) / 4)
F = System([steady_state; detJac], variables=[s; c; w])

# Set up projected hypersurface
h = ProjectedHypersurface(F, w)

# Degree of the discriminant
d = degree(h)
println("Degree of discriminant: $d")

# Routing gradient
C = [0.4723952824712583, 0.4334474432194565]
write_parameters("./results/kuramoto/center.txt", C)
r = RoutingFunction(h; c=C)

# Critical points
# pts = read_solutions("./results/kuramoto/routing_points.txt") |> real
routing_result = critical_points(r)
pts = routing_points(routing_result)
res = result(routing_result)
mon_res = monodromy_result(routing_result)

# Connected components
partition_result = partition_of_critical_points(r, routing_result)
G = partitions(partition_result)
idx = morse_indices(partition_result)
failures = failed_info(partition_result)

# Record computation time
time_end_round1 = time()
println("Computation time for round 1: $(time_end_round1 - time_start_round1) seconds")

M_x = maximum(p -> abs(p[1]), pts) * 1.05
M_y = maximum(p -> abs(p[2]), pts) * 1.05
h_symbolic(x, y) = 314928 * x^8 * y^4 + 1259712 * x^7 * y^5 + 1889568 * x^6 * y^6 + 1259712 * x^5 * y^7 +
          314928 * x^4 * y^8 + 139968 * x^10 + 699840 * x^9 * y + 1277208 * x^8 * y^2 +
          909792 * x^7 * y^3 - 279936 * x^6 * y^4 - 1084752 * x^5 * y^5 - 279936 * x^4 * y^6 +
          909792 * x^3 * y^7 + 1277208 * x^2 * y^8 + 699840 * x * y^9 +
          139968 * y^10 - 96957 * x^8 - 387828 * x^7 * y - 226962 * x^6 * y^2 + 676512 * x^5 * y^3 +
          1128249 * x^4 * y^4 + 676512 * x^3 * y^5 - 226962 * x^2 * y^6 - 387828 * x * y^7 - 96957 * y^8 +
          22680 * x^6 + 68040 * x^5 * y - 20844 * x^4 * y^2 - 155088 * x^3 * y^3 - 20844 * x^2 * y^4 +
          68040 * x * y^5 + 22680 * y^6 - 2298 * x^4 - 4596 * x^3 * y - 6894 * x^2 * y^2 - 4596 * x * y^3 - 2298 * y^4 +
          96 * x^2 + 96 * x * y + 96 * y^2 - 1;
root_counting_system = System(steady_state, variables=[s; c], parameters=w);

function analyze_and_save_result()

    write_parameters("./results/kuramoto/monodromy_parameters.txt", parameters(mon_res))
    write_solutions("./results/kuramoto/monodromy_result.txt", solutions(mon_res))
    write_solutions("./results/kuramoto/result.txt", solutions(res))
    write_solutions("./results/kuramoto/routing_points.txt", pts)
    write("./results/kuramoto/connected_components.txt", string(G))

    println("Connected components: $(G)")
    println("Indicies: $(idx)")
    println("Failed info: $(failures)")
    println()

    generate_plot(r, routing_result, partition_result;
        h=h_symbolic,
        xlims=(-M_x, M_x),
        ylims=(-M_y, M_y),
    )

    plot!(; xlims=(-M_x, M_x), ylims=(-M_y, M_y), legend=false, dpi=400)
    savefig("./figures/kuramoto.pdf")
    savefig("./figures/kuramoto.svg")
    savefig("./figures/kuramoto.png")

    generate_plot(r, routing_result, partition_result;
        h=h_symbolic,
        root_counting_system=root_counting_system,
        markersize=6,
        annotation_textsize=5,
        contour_stepsize = 0.005,
        xlims=(-M_x, M_x),
        ylims=(-M_y, M_y),
    )

    plot!(; xlims=(-0.8, 0.8), ylims=(-0.8, 0.8), legend=false, dpi=400)
    savefig("./figures/kuramoto_zoomed_in.pdf")
    savefig("./figures/kuramoto_zoomed_in.svg")
    savefig("./figures/kuramoto_zoomed_in.png")

end
analyze_and_save_result()

# Try another round of monodromy (only if you think the first attempt missed solutions)
println("Running second round of monodromy...")
time_start_round2 = time()
old_number_of_monodromy_solutions = length(solutions(mon_res))
options = MonodromyOptions(
    parameter_sampler=p -> 100 .* randn(ComplexF64, length(p)), # smaller loops
    max_loops_no_progress=15 # change the stopping criterion
)
routing_result = critical_points(r, solutions(mon_res), parameters(mon_res), options=options)
pts = routing_points(routing_result)
res = result(routing_result)
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
