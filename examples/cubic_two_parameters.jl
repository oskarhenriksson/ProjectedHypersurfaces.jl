using Random, Plots, ProjectedHypersurfaces
mkpath("./results/cubic_two_parameters");

Random.seed!(0x8b868320)

########

# The discriminant of x^3 + a * x^2 + b*x + 1 is
# 4*a^3 - a^2*b^2 - 18*a*b + 4*b^3 + 27

# Incidence variety of discriminant
@var a b x
F = System([x^3 + a * x^2 + b * x + 1; 3 * x^2 + 2 * a * x + b], variables=[a, b, x])

# Set up projected hypersurface
h = ProjectedHypersurface(F, [a; b])

# Degree of discriminant
d = degree(h)
println("Degree of discriminant: $d")

# Routing gradient
c = [13.758979284873828, -0.09884333335596635]
r = RoutingFunction(h; c=c)

# Critical points
# pts = read_solutions("./results/cubic_two_parameters/routing_points.txt") |> real
routing_result = critical_points(r)
pts = routing_points(routing_result)
res = trace_result(routing_result)
mon_res = monodromy_result(routing_result)

write_parameters("./results/cubic_two_parameters/monodromy_parameters.txt", parameters(mon_res))
write_solutions("./results/cubic_two_parameters/monodromy_result.txt", solutions(mon_res))
write_solutions("./results/cubic_two_parameters/result.txt", solutions(res))
write_solutions("./results/cubic_two_parameters/routing_points.txt", pts)

# Connecting 
partition_result = partition_of_critical_points(r, routing_result)
G = partitions(partition_result)
idx = morse_indices(partition_result)
failures = failed_info(partition_result)
println("Connected components: $(G)")
println("Indicies: $(idx)")
println("Failed info: $(failures)")
println()

write("./results/cubic_two_parameters/connected_components.txt", string(G))

M_x = maximum(p -> abs(p[1]), pts) + 6
M_y = maximum(p -> abs(p[2]), pts) + 6
generate_plot(r, routing_result, partition_result;
    h = (x,y) -> 4 * x^3 - x^2 * y^2 - 18 * x * y + 4 * y^3 + 27,
    markersize=6,
    annotation_textsize=5,
    arrowstyle=:simple,
    flow_linewidth=3,
    discriminant_linewidth=0,
    legend=false,
    #root_counting_system=System([x^3 + a * x^2 + b * x + 1], variables=[x], parameters=[a; b]),
    contour_stepsize=0.1,
    xlims=(-M_x, M_x),
    ylims=(-M_y, M_y),
)

savefig("./figures/cubic.pdf")
savefig("./figures/cubic_no_flow.svg")
savefig("./figures/cubic.png")
