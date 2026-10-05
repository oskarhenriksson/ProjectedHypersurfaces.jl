# system given in section 4 of https://arxiv.org/pdf/2505.00128

using Random, Plots, LinearAlgebra,  ImplicitPlots, ProjectedHypersurfaces

Random.seed!(123)

# Set up incidence variety of discriminant
@var b,c,d,f βy,βz, γy,γz x[1:3]
steady_state = [
    (b*(x[1] + x[2]) + (b + c)*x[3])*(1- (x[1] + x[2] + x[3])) - βy*x[1]*x[2] - βz*x[1]*x[3] + γy*x[2] + γz*x[3] - d*x[1],
     βy*x[1]*x[2] + βy*x[2]*x[3]- (d + f + γy)*x[2],
    βz*x[1]*x[3] - βy*x[2]*x[3] - (d + γz)*x[3]
]
Jac = differentiate(steady_state, x)
detJac = det(Jac)
F = System([steady_state; detJac], variables = [b; c; d; f; βy; βz; γy;γz; x])

# Set up the projected hypersurface
h_original = ProjectedHypersurface(F, [b; c; d; f; βy; βz; γy;γz;])

# Degree of the original discriminant
d_original = degree(h_original)

# Fix a center for the routing functions
C = [0.02, 0.2, 0.05,0.5,0.4,0.04,0.98,0.07]
r = RoutingFunction(h_original; c=C)

routing_result = critical_points(r; start_grid_width=0)
pts = routing_points(routing_result)
res = result(routing_result)
mon_res = monodromy_result(routing_result)

# Connect the critical points
roadmap = gradient_roadmap(r, routing_result)
G = partition(roadmap)
idx = morse_indices(roadmap)
failures = failed_info(roadmap)

#pl_original_smaller = generate_plot(
#    r_original, pts_original,
#    G_original,
#    idx_original;
#    contour_stepsize = 5e-4, contour_levels = 60, markersize = 6, legend = false,
#    xlims = (0, 0.09), ylims = (0, 0.5),
#    root_count_fn = allee_root_count_fn,
#    annotate_root_counts = true, annotation_textsize = 4,
#)
#savefig("figures/tripartite.png")
#savefig("figures/tripartite.pdf")
#savefig("figures/tripartite.svg")

