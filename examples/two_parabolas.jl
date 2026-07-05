
using Random, ProjectedHypersurfaces
mkpath("./results/two_parabolas");

Random.seed!(1234)

########

# Incidence variety of discriminants
@var a b x
F1 = System([x^2 + a * x + b; 2*x + a], variables=[a, b, x])
h1 = ProjectedHypersurface(F1, [a; b])

@var z
F2 = System([z^2 + a * z - b; 2*z + a], variables=[a, z, b])
h2 = ProjectedHypersurface(F2, [a; b])

# Routing gradient
c = [7, 3]
r = RoutingFunction([h1, h2]; c=c)

# Critical points
# pts = read_solutions("./results/two_parabolas/routing_points.txt") |> real
routing_result = critical_points(r)
pts = routing_points(routing_result)
res = trace_result(routing_result)
mon_res = monodromy_result(routing_result)

write_parameters("./results/two_parabolas/monodromy_parameters.txt", parameters(mon_res))
write_solutions("./results/two_parabolas/monodromy_result.txt", solutions(mon_res))
write_solutions("./results/two_parabolas/result.txt", solutions(res))
write_solutions("./results/two_parabolas/routing_points.txt", pts)

# Connecting 
partition_result = partition_of_critical_points(r, routing_result)
G = partitions(partition_result)
idx = morse_indices(partition_result)
failures = failed_info(partition_result)
println("Connected components: $(G)")
println("Indicies: $(idx)")
println("Failed info: $(failures)")
println()

write("./results/two_parabolas/connected_components.txt", string(G))

M_x = maximum(p -> abs(p[1]), pts) + 6
M_y = maximum(p -> abs(p[2]), pts) + 6
generate_plot(r, routing_result, partition_result;
    h = (x,y) -> (4*y-x^2)*(-4*y-x^2),
    markersize=7,
    arrowstyle=:simple,
    flow_linewidth=3,
    discriminant_linewidth=4,
    legend=:bottomright,
    contour_stepsize=0.1,
    xlims = (-M_x, M_x),
    ylims = (-M_y, M_y)
)

savefig("./figures/two_parabolas.pdf")
savefig("./figures/two_parabolas.svg")
savefig("./figures/two_parabolas.png")
