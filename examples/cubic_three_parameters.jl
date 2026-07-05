using Random, Plots, ProjectedHypersurfaces

Random.seed!(0x8b868320)

# The discriminant of x^3 + a * x^2 + b*x + γ is
# 4*a^3*γ - a^2*b^2 - 18*a*b*γ + 4*b^3 + 27*γ^2

# Incidence variety of discriminant
@var a b γ x
f = x^3 + a * x^2 + b * x + γ
F = System([f; differentiate(f, x)], variables=[a, b, γ, x])

# Set up projected hypersurface
projection_variables = [a; b; γ]
k = length(projection_variables)
h = ProjectedHypersurface(F, projection_variables)

# Degree of discriminant
d = degree(h)
println("Degree of discriminant: $d")

# Routing gradient
c = 10 .* randn(k)
r = RoutingFunction(h; c=c)

# Critical points
routing_result = critical_points(r)
pts = routing_points(routing_result)
res = result(routing_result)
mon_res = monodromy_result(routing_result)

# Connecting critical points
partition_result = partition_of_critical_points(r, routing_result)
G = partitions(partition_result)
idx = morse_indices(partition_result)
failures = failed_info(partition_result)
println("Connected components: $(G)")
println("Indicies: $(idx)")
println("Failed info: $(failures)")
println()

generate_plot(r, routing_result, partition_result;
    root_counting_system=System([x^3 + a * x^2 + b * x + γ], variables=[x], parameters=[a; b; γ])
)

savefig("./figures/testcubic.pdf")
