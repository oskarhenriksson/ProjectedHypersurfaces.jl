
using Random, Plots, ProjectedHypersurfaces
mkpath("./results/two_discriminants");

Random.seed!(1234)

########

# Incidence variety of discriminants
@var a b x
F1 = System([x^3 + a * x^2 + b * x + 1; 3 * x^2 + 2 * a * x + b], variables=[a, b, x])
h1 = ProjectedHypersurface(F1, [a; b])

@var z
F2 = System([z^2 + a * z - b; 2*z + a], variables=[a, z, b])
h2 = ProjectedHypersurface(F2, [a; b])

# Routing gradient
c = [7, 3]
r = RoutingFunction([h1, h2]; c=c)

# Critical points
# pts = read_solutions("./results/two_discriminants/routing_points.txt") |> real
pts, res, mon_res = critical_points(r)

write_parameters("./results/two_discriminants/monodromy_parameters.txt", parameters(mon_res))
write_solutions("./results/two_discriminants/monodromy_result.txt", solutions(mon_res))
write_solutions("./results/two_discriminants/result.txt", solutions(res))
write_solutions("./results/two_discriminants/routing_points.txt", pts)

# Connecting 
G, idx, failed_info = partition_of_critical_points(r, pts)
println("Connected components: $(G)")
println("Indicies: $(idx)")
println("Failed info: $(failed_info)")
println()

write("./results/two_discriminants/connected_components.txt", string(G))

M_x = maximum(p -> abs(p[1]), pts) + 6
M_y = maximum(p -> abs(p[2]), pts) + 6
generate_plot(r, pts, G, idx;
    h = (a,b) -> (-4*b-a^2)*(4*a^3 - a^2*b^2 - 18*a*b + 4*b^3 + 27),
    markersize=7,
    arrowstyle=:simple,
    flow_linewidth=3,
    discriminant_linewidth=4,
    legend=:bottomright,
    contour_stepsize=0.1,
    xlims=(-M_x, M_x),
    ylims=(-M_y, M_y),
)

savefig("./figures/two_discriminants.pdf")
savefig("./figures/two_discriminants.svg")
savefig("./figures/two_discriminants.png")