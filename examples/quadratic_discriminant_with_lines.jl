using Random, Plots, ProjectedHypersurfaces
mkpath("./results/quadratic_discriminant_with_lines");

Random.seed!(12345)

# Set up the system
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])
h = ProjectedHypersurface(F, [a, b]);

# Pick a center for the routing function
c = [10, 5]

# Set up the routing function and gradient
r = RoutingFunction(h; c=c, g=[a, b]);
e = denominator_exponent(r)
∇r = RoutingGradient(r)

# Find the complex critical points 
routing_result = critical_points(r)
pts = routing_points(routing_result)
res = result(routing_result)
mon_res = monodromy_result(routing_result)

write_parameters("./results/quadratic_discriminant_with_lines/monodromy_parameters.txt", parameters(mon_res))
write_solutions("./results/quadratic_discriminant_with_lines/monodromy_result.txt", solutions(mon_res))
write_solutions("./results/quadratic_discriminant_with_lines/result.txt", solutions(res))
write_solutions("./results/quadratic_discriminant_with_lines/routing_points.txt", pts)

# Connect the critical points
partition_result = partition_of_critical_points(r, routing_result)
G = regions(partition_result)
idx = morse_indices(partition_result)
failures = failed_info(partition_result)
println("Connected components: $(G)")
println("Indicies: $(idx)")
println("Failed info: $(failures)")
println()

write("./results/quadratic_discriminant_with_lines/connected_components.txt", string(G))

##### Plotting 
M_x = maximum(p -> abs(p[1]), pts) + 10
M_y = maximum(p -> abs(p[2]), pts) + 10
generate_plot(r, routing_result, partition_result;
    h=(a, b) -> (a^2 - 4 * b)*a*b,
    markersize=7,
    arrowstyle=:simple,
    flow_linewidth=3,
    discriminant_linewidth=4,
    root_counting_system=System([x^2 + a * x + b], variables=[x], parameters=[a; b]),
    legend=:bottomright,
    contour_stepsize=0.1,
    xlims=(-M_x, M_x),
    ylims=(-M_y, M_y),
)

savefig("./figures/quadratic_discriminant_with_lines.png")
savefig("./figures/quadratic_discriminant_with_lines.svg")
savefig("./figures/quadratic_discriminant_with_lines.pdf")
