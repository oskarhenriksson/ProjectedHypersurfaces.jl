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
pts, res, mon_res = critical_points(r)

write_parameters("./results/quadratic_discriminant_with_lines/monodromy_parameters.txt", parameters(mon_res))
write_solutions("./results/quadratic_discriminant_with_lines/monodromy_result.txt", solutions(mon_res))
write_solutions("./results/quadratic_discriminant_with_lines/result.txt", solutions(res))
write_solutions("./results/quadratic_discriminant_with_lines/routing_points.txt", pts)

# Connect the critical points
G, idx, failed_info = partition_of_critical_points(r, pts)
println("Connected components: $(G)")
println("Indicies: $(idx)")
println("Failed info: $(failed_info)")
println()

write("./results/quadratic_discriminant_with_lines/connected_components.txt", string(G))

##### Plotting 
M_x = maximum(p -> abs(p[1]), pts) + 10
M_y = maximum(p -> abs(p[2]), pts) + 10
generate_plot(r, pts, G, idx;
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