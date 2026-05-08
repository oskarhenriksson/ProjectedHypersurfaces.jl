using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Test, Random, ProjectedHypersurfaceRegions, LinearAlgebra, Logging



@testset "Quadratic discriminant" begin
    
    @var a b x
    F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])
    h = ProjectedHypersurface(F, [a, b]);

    c = [13, 2]

    r = RoutingFunction(h; c=c);
    e = denominator_exponent(r)
    @test e == 2

    ∇r = RoutingGradient(r);

    disc = a^2 - 4 * b
    q = 1 + sum(([a;b] - c).^2)
    r_test = differentiate(log(disc / q^e), [a;b])
    H_test = differentiate(r_test, [a;b])

    k = 2
    p = randn(ComplexF64, k)
    u1 = evaluate(r_test, [a;b] => p)
    U1 = evaluate(H_test, [a;b] => p)
    u2, u22, U2 = randn(ComplexF64, k), randn(ComplexF64, k), randn(ComplexF64, k, k);
    ProjectedHypersurfaceRegions.evaluate_and_jacobian!(u2, U2, ∇r, p);
    ProjectedHypersurfaceRegions.evaluate!(u22, ∇r, p);

    @test norm(u1 - u2) < 1e-12
    @test norm(u1 - u22) < 1e-12
    @test norm(U1 - U2) < 1e-12
end

@testset "Cubic discriminant" begin
    
    @var a b x
    F = System([x^3 + a * x^2 + b * x + 1; 3 * x^2 + 2 * a * x + b], variables=[a, b, x])
    h = ProjectedHypersurface(F, [a, b]);

    c = [10, 5]
    r = RoutingFunction(h; c=c)
    ∇r = RoutingGradient(r)

    @test degree(h) == 4
    @test denominator_exponent(r) == 3

    # Symbolic routing function
    h_symbolic = 4*a^3 - a^2*b^2 - 18*a*b + 4*b^3 + 27
    r_symbolic = h_symbolic/((a - c[1])^2 + (b - c[2])^2 + 1)^3
    ∇r_symbolic = System(differentiate(log(r_symbolic), [a, b]), variables=[a, b]) |> fixed

    # Test evaluation and Jacobian
    p0 = randn(ComplexF64, 2)
    u = zeros(ComplexF64, 2)
    U = zeros(ComplexF64, 2, 2)
    u2 = zeros(ComplexF64, 2)
    U2 = zeros(ComplexF64, 2, 2)
    ProjectedHypersurfaceRegions.evaluate_and_jacobian!(u, U, ∇r, p0)
    ProjectedHypersurfaceRegions.evaluate_and_jacobian!(u2, U2, ∇r_symbolic, p0)
    @test norm(u-u2) < 1e-12
    @test norm(U-U2) < 1e-12
    @test norm(∇r_symbolic(p0)-∇r(p0)) < 1e-12

    # Check realness
    p0 = randn(2)
    ProjectedHypersurfaceRegions.evaluate_and_jacobian!(u, U, ∇r, p0)
    @test norm(imag(u)) < 1e-12
    @test norm(imag(U)) < 1e-12

    # Test forming the routing point homotopies
    p1 = zeros(2)
    q1 = randn(2)
    H = ProjectedHypersurfaceRegions.RoutingPointsHomotopy(∇r, p1, q1)
    u = randn(ComplexF64, 2)
    U = randn(ComplexF64, 2, 2)
    x0 = randn(ComplexF64, 2)
    ProjectedHypersurfaceRegions.evaluate_and_jacobian!(u, U, H, x0, 1.0)
    @test norm(∇r_symbolic(x0) - u) < 1e-12
    ProjectedHypersurfaceRegions.evaluate_and_jacobian!(u, U, H, x0, 0.0)
    @test norm(∇r_symbolic(x0)-q1 - u) < 1e-12


    # Test that the expansion of start solutions works
    ∇r = RoutingGradient(r)
    MS, H, S0, rhs0, k = ProjectedHypersurfaceRegions._setup_monodromy_solver(∇r)
    S0, new_pts = ProjectedHypersurfaceRegions._expand_start_solutions(
        ∇r, H, S0, rhs0, k;
        start_grid_width = 10,
        start_grid_stepsize = 1,
    )
    @test length(S0) >= 5 # should find at least five new points (one for each local maximum)
    @test all(norm(∇r(z)) < 1e-12 for z in new_pts) # all new points are routing points
    @test all(norm(∇r(z)-rhs0) < 1e-12 for z in S0) # all points are traced to solutions of ∇r=rhs0

    # Check critical points
    options = MonodromyOptions(target_solutions_count = 2)
    pts, res0, mon_res = critical_points(r, start_grid_width=0, options=options)    
    @test all(norm.(∇r_symbolic.(solutions(res0))) .< 1e-12)

    pl = generate_plot(
        r,
        [[0.0, 0.0]],
        [[1]],
        [0];
        h = (a, b) -> a^2 - 4 * b,
        root_counting_system = System([x^2 + a * x + b], variables = [x], parameters = [a; b]),
        annotate_root_counts = true,
        contour_stepsize = 0.5,
        xlims = (-1.0, 1.0),
        ylims = (-1.0, 1.0),
    )
    @test !isnothing(pl)

end;

@testset "Quadratic discriminant with lines" begin
    
    @var a b x
    F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])
    c = [10, 5]
    h = ProjectedHypersurface(F, [a, b]);
    r = RoutingFunction(h; c=c, g=[a, b]);
    ∇r = RoutingGradient(r)

    e = denominator_exponent(r)

    @test degree(h) == 2
    @test e == 3

    # Symbolic routing function
    h_symbolic = (a^2 - 4*b)*a*b
    r_symbolic = h_symbolic/((a - c[1])^2 + (b - c[2])^2 + 1)^3
    ∇r_symbolic = System(differentiate(log(r_symbolic), [a, b]), variables=[a, b]) |> fixed

    # Test evaluation and Jacobian
    p0 = randn(ComplexF64, 2)
    u = zeros(ComplexF64, 2)
    U = zeros(ComplexF64, 2, 2)
    u2 = zeros(ComplexF64, 2)
    U2 = zeros(ComplexF64, 2, 2)
    ProjectedHypersurfaceRegions.evaluate_and_jacobian!(u, U, ∇r, p0)
    ProjectedHypersurfaceRegions.evaluate_and_jacobian!(u2, U2, ∇r_symbolic, p0)
    @test norm(u-u2) < 1e-12
    @test norm(U-U2) < 1e-12
    @test norm(∇r_symbolic(p0)-∇r(p0)) < 1e-12

    # Check critical points
    options = MonodromyOptions(target_solutions_count = 2)
    pts, res0, mon_res = critical_points(r, start_grid_width=0, options=options)   
    @test all(norm.(∇r_symbolic.(solutions(res0))) .< 1e-12)

end;


@testset "Two discriminants" begin
    
    @var a b x z
    F1 = System([x^3 + a * x^2 + b * x + 1; 3 * x^2 + 2 * a * x + b], variables=[a, b, x])
    h1 = ProjectedHypersurface(F1, [a; b])
    F2 = System([z^2 + a * z - b; 2*z + a], variables=[a, z, b])
    h2 = ProjectedHypersurface(F2, [a; b])
    c = [7, 3]
    r = RoutingFunction([h1, h2]; c=c);
    ∇r = RoutingGradient(r)

    e = denominator_exponent(r)
    @test e == 4

    # Test evaluation
    h_symbolic = (-a^2 - 4*b)*(4*a^3 - a^2*b^2 - 18*a*b + 4*b^3 + 27)
    r_symbolic = h_symbolic/((a - c[1])^2 + (b - c[2])^2 + 1)^e
    ∇r_symbolic = System(differentiate(log(r_symbolic), [a, b]), variables=[a, b]) |> fixed
    p0 = [1, 3]
    @test norm(∇r(p0) - ∇r_symbolic(p0)) < 1e-12

    # Check critical points
    options = MonodromyOptions(target_solutions_count = 2)
    pts, res0, mon_res = critical_points(r, start_grid_width=0, options=options)   
    @test all(norm.(∇r_symbolic.(solutions(res0))) .< 1e-12)

end

@testset "Kuramoto discriminant" begin
    
    @var s[1:2] c[1:2] w[1:2]
    freq1 = (s[1] * c[2] - c[1] * s[2]) + (s[1] * 1 - c[1] * 0) - 3 * w[1]
    freq2 = (s[2] * c[1] - c[2] * s[1]) + (s[2] * 1 - c[2] * 0) - 3 * w[2]
    norm1 = s[1]^2 + c[1]^2 - 1
    norm2 = s[2]^2 + c[2]^2 - 1
    steady_state = [freq1, freq2, norm1, norm2]
    Jac = differentiate.(steady_state, [s; c]')
    detJac = expand(det(Jac) / 4)

    F = System([steady_state; detJac], variables=[s; c; w])

    C = [1, 1]
    h = ProjectedHypersurface(F, [w[1], w[2]]);
    r = RoutingFunction(h; c=C);
    ∇r = RoutingGradient(r)

    # Degree of the discriminant
    @test degree(h) == 12

    h_symbolic = 314928 * w[1]^8 * w[2]^4 + 1259712 * w[1]^7 * w[2]^5 + 1889568 * w[1]^6 * w[2]^6 + 1259712 * w[1]^5 * w[2]^7 +
          314928 * w[1]^4 * w[2]^8 + 139968 * w[1]^10 + 699840 * w[1]^9 * w[2]  + 1277208 * w[1]^8 * w[2]^2 +
          909792 * w[1]^7 * w[2]^3 - 279936 * w[1]^6 * w[2]^4 - 1084752 * w[1]^5 * w[2]^5 - 279936 * w[1]^4 * w[2]^6 +
          909792 * w[1]^3 * w[2]^7 + 1277208 * w[1]^2 * w[2]^8 + 699840 * w[1]  * w[2]^9 +
          139968 * w[2]^10 - 96957 * w[1]^8 - 387828 * w[1]^7 * w[2]  - 226962 * w[1]^6 * w[2]^2 + 676512 * w[1]^5 * w[2]^3 +
          1128249 * w[1]^4 * w[2]^4 + 676512 * w[1]^3 * w[2]^5 - 226962 * w[1]^2 * w[2]^6 - 387828 * w[1]  * w[2]^7 - 96957 * w[2]^8 +
          22680 * w[1]^6 + 68040 * w[1]^5 * w[2]  - 20844 * w[1]^4 * w[2]^2 - 155088 * w[1]^3 * w[2]^3 - 20844 * w[1]^2 * w[2]^4 +
          68040 * w[1]  * w[2]^5 + 22680 * w[2]^6 - 2298 * w[1]^4 - 4596 * w[1]^3 * w[2]  - 6894 * w[1]^2 * w[2]^2 - 4596 * w[1]  * w[2]^3 - 2298 * w[2]^4 +
          96 * w[1]^2 + 96 * w[1]  * w[2]  + 96 * w[2]^2 - 1;
    
    r_symbolic = h_symbolic/((w[1] - C[1])^2 + (w[2] - C[2])^2 + 1)^7
    ∇r_symbolic = System(differentiate(log(r_symbolic), [w[1], w[2]]), variables=[w[1], w[2]]) |> fixed

    p0 = randn(ComplexF64, 2)
    @test norm(∇r_symbolic(p0)-∇r(p0)) < 1e-12

end;

@testset "Connect points" begin
    
    @var a b x
    F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])
    h = ProjectedHypersurface(F, [a, b]);

    c = [13, 2]
    r = RoutingFunction(h; c=c)
    ∇r = RoutingGradient(r)

    pts = [
        [-3.9180890683992278, -6.635887940807433], 
        [13.040296300414134, 1.993819726256856], 
        [3.2168112092392103, 8.082538361382138], 
        [-12.339018441254076, -2.1071368134982302]
    ]

    @test all(norm.(∇r.(pts)) .< 1e-12) 

    G, idx, failed_info = partition_of_critical_points(r, pts)

    @test sort(G) == [[1, 2, 4], [3]]
    @test idx == [1, 0, 0, 0]
    @test isempty(failed_info)

end;
@testset "Hypersurface evaluations for quadratic" begin

    # Set up the system
    @var a b x
    F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])
    h = ProjectedHypersurface(F, [a, b])

    # Test the degree
    @test degree(h) == 2

    # Random point
    p = rand(2)

    # Test the evaluation formula
    pt = [1, 1]
    log_abs_h = p -> log(abs(p[1]^2 - 4*p[2]))
    direction = h.PWS.L.direction
    C = log(abs(direction[1]^2))
    @test h(pt) + C - log_abs_h(pt) |> abs < 1e-6

    # Test the gradient 
    pt = [3, 2]
    ∇log_abs_h = p -> [(2 * p[1])/(p[1]^2 - 4 * p[2]), -4/(p[1]^2 - 4 * p[2])]
    @test gradient(h, pt) - ∇log_abs_h(pt) |> norm < 1e-6

    # Test the Hessian
    pt = [11, 7]
    Hess_log_abs_h = p -> [[2/(p[1]^2 - 4*p[2]) - 4*p[1]^2/(p[1]^2 - 4*p[2])^2 8*p[1]/(p[1]^2 - 4*p[2])^2]; 
    [8*p[1]/(p[1]^2 - 4*p[2])^2  -16/(p[1]^2 - 4*p[2])^2]]
    @test Hess_log_abs_h(pt) - ProjectedHypersurfaceRegions.gradient_and_hessian(h, pt)[2] |> norm < 1e-6

end
@testset "Noninjective projection" begin
    @var x y z
    F = System([z-x^2, y], variables = [x,y, z])
    h = ProjectedHypersurface(F, [y, z])

    @test degree(h) == 1 # the downstairs degree should be 1

    # h(y,z) = y (up to a constant) so gradient(h, [y, z]) = [1/y, 0]
    @test gradient(h, [2, 3]) - [1/2, 0] |> norm < 1e-6

end

@testset "Two components projecting to the same hypersurface" begin
    @var a, b, x
    F = System([a^2 - 4*b, (x - a + 1) * (x - a)], variables=[a, b, x])
    # V(F) has two irreducible components that project down to V(a^2-4b)
    h = ProjectedHypersurface(F, [a, b])
    @test degree(h) == 2
end

@testset "Empty PWS" begin
    Random.seed!(1234)
    @var a b x
    F = System([(x - a) * (x - b); 2 * x - (a + b)], variables=[a, b, x])
    @test_throws "No witness points found." PseudoWitnessSet(F, 2)
end

@testset "Multiplicity detection" begin
    @var a, b, x
    F = System([a^2 - 4*b, (x - a + 1)^2 * (x - a)], variables=[a, b, x])
    @test_logs (:warn, "Irreducible component of higher multiplicity detected in the incidence variety.") match_mode=:any PseudoWitnessSet(F,2)
end