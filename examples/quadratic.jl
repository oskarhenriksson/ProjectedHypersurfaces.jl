using Random, Plots, ProjectedHypersurfaces

mkpath("./results/quadratic");

Random.seed!(12345)

# Incidence variety of the discriminant
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])

# Set up projected hypersurface
h = ProjectedHypersurface(F, [a, b])

# Degree of the discriminant
d = degree(h)
println("Degree of discriminant: $d")

# Pick a center for the routing function
c = [13, 2]

# Set up the routing function gradient
r = RoutingFunction(h; c=c)

# Find the complex critical points 
# pts = read_solutions("./results/quadratic/routing_points.txt") |> real
pts, res, mon_res = critical_points(r)

write_parameters("./results/quadratic/monodromy_parameters.txt", parameters(mon_res))
write_solutions("./results/quadratic/monodromy_result.txt", solutions(mon_res))
write_solutions("./results/quadratic/result.txt", solutions(res))
write_solutions("./results/quadratic/routing_points.txt", pts)

# Connect the critical points
G, idx, failed_info = partition_of_critical_points(r, pts)
println("Connected components: $(G)")
println("Indicies: $(idx)")
println("Failed info: $(failed_info)")
println()

write("./results/quadratic/connected_components.txt", string(G))

# Analyze root counts and plot result
M_x = maximum(p -> abs(p[1]), pts) + 4
M_y = maximum(p -> abs(p[2]), pts) + 3
generate_plot(r, pts, G, idx;
    h=(a, b) -> (a^2 - 4 * b),
    markersize=7,
    annotation_textsize=6,
    arrowstyle=:simple,
    flow_linewidth=3,
    discriminant_linewidth=4,
    #root_counting_system=System([x^2 + a * x + b], variables=[x], parameters=[a; b]),
    legend=:bottomright,
    contour_stepsize=0.1,
    xlims=(-M_x, M_x),
    ylims=(-M_y, M_y),
)

savefig("./figures/quadratic.png")
savefig("./figures/quadratic.svg")
savefig("./figures/quadratic.pdf")