using ProjectedHypersurfaces, LinearAlgebra

@var x[1:3] a[1:2] b[1:2] t
f = sum(x.^2)-1
∇f = differentiate(f, x)

L = [a; 1] .* t + [b; 0]


# Incidence variety of the discriminant
F = System([
    f;
    x-L
    ∇f⋅[a; 1]
])

# Set up projected hypersurface
h = ProjectedHypersurface(F, [a; b])

# Set up the routing function gradient
r = RoutingFunction(h)

# Find the complex critical points
pts, res, mon_res = critical_points(r; start_grid_width=0)

# Connect the critical points
G, idx, failed_info = partition_of_critical_points(r, pts)