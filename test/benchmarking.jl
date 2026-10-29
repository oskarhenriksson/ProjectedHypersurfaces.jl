
using Plots, ImplicitPlots, Random, LinearAlgebra, ProjectedHypersurfaces, BenchmarkTools

# Setup the projected hypersurface

a2 = 14 // 10;

a3 = 7 // 10;

b3 = 1;

A2 = 16 // 10;

A3 = 9 // 10;

B3 = 6 // 10;

c3 = 1;

@var p[1:2] φ[1:2] c[1:2]

f = [φ[1]^2 + φ[2]^2 - 1,

p[1]^2 + p[2]^2 - 2 * (a3 * p[1] + b3 * p[2]) * φ[1] + 2 * (b3 * p[1] - a3 * p[2]) * φ[2] + a3^2 + b3^2 - c[1],

p[1]^2 + p[2]^2 - 2 * A2 * p[1] + 2 * ((a2 - a3) * p[1] - b3 * p[2] + A2 * a3 - A2 * a2) * φ[1] + 2 * (b3 * p[1] + (a2 - a3) * p[2] - A2 * b3) * φ[2]

+ (a2 - a3)^2 + b3^2 + A2^2 - c[2],

p[1]^2 + p[2]^2 - 2 * (A3 * p[1] + B3 * p[2]) + A3^2 + B3^2 - c3

]

Jac = differentiate(f, [p; φ])

F = System([f; det(Jac)], variables=[p; φ; c]) # use System(prod(c)*[f;det(Jac)]) to impose c>0

h = ProjectedHypersurface(F, c)

# Benchmark individual evaluations

u = rand(ComplexF64, 2);

U = rand(ComplexF64, 2, 2);

@benchmark ProjectedHypersurfaces.gradient_and_hessian!(

$u, $U, $h, x

) setup=(

x = rand(ComplexF64, 2)

)

@benchmark ProjectedHypersurfaces.gradient!(

$u, $h, x

) setup=(

x = rand(ComplexF64, 2)

)

# Benchmark tracking of witness points

PWS = h.PWS;

tZ = deepcopy(PWS.tZ);

@benchmark ProjectedHypersurfaces.track!(

$tZ, $PWS, z

) setup=(

z = rand(ComplexF64, 2);

)

# Benchmark repeated evaluation along a path. The gradient flow and the
# monodromy path trackers ask for many evaluations at points that are close to
# each other, so reusing a fibre should be better.

function repeated_gradient!(u, h, x0, direction, stepsize, steps)
    x = copy(x0)
    for _ = 1:steps
        ProjectedHypersurfaces.gradient!(u, h, x)
        @. x += stepsize * direction
    end
    u
end

function repeated_gradient_and_hessian!(u, U, h, x0, direction, stepsize, steps)
    x = copy(x0)
    for _ = 1:steps
        ProjectedHypersurfaces.gradient_and_hessian!(u, U, h, x)
        @. x += stepsize * direction
    end
    u, U
end

direction = rand(ComplexF64, 2);

direction ./= norm(direction);

@benchmark repeated_gradient!(

$u, $h, x0, $direction, 1e-2, 20

) setup=(

x0 = rand(ComplexF64, 2)

)

@benchmark repeated_gradient_and_hessian!(

$u, $U, $h, x0, $direction, 1e-2, 20

) setup=(

x0 = rand(ComplexF64, 2)

)
