using Test, Random, ProjectedHypersurfaces, LinearAlgebra, Logging
@testset "Quadratic discriminant" begin

    Random.seed!(12345)

    @var a b x
    F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])
    h = ProjectedHypersurface(F, [a, b])

    @test degree(h) == 2
    @test trace_test(h) < 1e-10

    c = [13, 2]

    r = RoutingFunction(h; c=c)
    e = denominator_exponent(r)
    @test e == 2

    ∇r = RoutingGradient(r)

    disc = a^2 - 4 * b
    q = 1 + sum(([a; b] - c) .^ 2)
    r_test = differentiate(log(disc / q^e), [a; b])
    H_test = differentiate(r_test, [a; b])

    k = 2
    p = randn(ComplexF64, k)
    u1 = evaluate(r_test, [a; b] => p)
    U1 = evaluate(H_test, [a; b] => p)
    u2, u22, U2 = randn(ComplexF64, k), randn(ComplexF64, k), randn(ComplexF64, k, k)
    ProjectedHypersurfaces.evaluate_and_jacobian!(u2, U2, ∇r, p)
    ProjectedHypersurfaces.evaluate!(u22, ∇r, p)

    @test norm(u1 - u2) < 1e-12
    @test norm(u1 - u22) < 1e-12
    @test norm(U1 - U2) < 1e-12
end

@testset "Cubic discriminant" begin

    Random.seed!(12345)

    @var a b x
    F = System([x^3 + a * x^2 + b * x + 1; 3 * x^2 + 2 * a * x + b], variables=[a, b, x])
    h = ProjectedHypersurface(F, [a, b])

    c = [10, 5]
    r = RoutingFunction(h; c=c)
    ∇r = RoutingGradient(r)

    @test degree(h) == 4
    @test trace_test(h) < 1e-6
    @test denominator_exponent(r) == 3

    # Symbolic routing function
    h_symbolic = 4*a^3 - a^2*b^2 - 18*a*b + 4*b^3 + 27
    r_symbolic = h_symbolic/((a - c[1])^2 + (b - c[2])^2 + 1)^3
    ∇r_symbolic = System(differentiate(log(r_symbolic), [a, b]), variables=[a, b]) |> fixed

    # Test sample
    sample = sample_points(h, 6)
    @test all(contains.(Ref(h), sample))
    @test all([abs(evaluate(h_symbolic, [a, b]=>pt)) < 1e-9 for pt in sample])

    # Test interpolation
    IR = interpolate(h)
    @test polynomial(IR) == h_symbolic

    # Test evaluation and Jacobian
    p0 = randn(ComplexF64, 2)
    u = zeros(ComplexF64, 2)
    U = zeros(ComplexF64, 2, 2)
    u2 = zeros(ComplexF64, 2)
    U2 = zeros(ComplexF64, 2, 2)
    ProjectedHypersurfaces.evaluate_and_jacobian!(u, U, ∇r, p0)
    ProjectedHypersurfaces.evaluate_and_jacobian!(u2, U2, ∇r_symbolic, p0)
    @test norm(u-u2) < 1e-12
    @test norm(U-U2) < 1e-12
    @test norm(∇r_symbolic(p0)-∇r(p0)) < 1e-12

    # test Jacobian (Hessian) across an evaluation sequence to verify correct cache usage. 
    log_gradient = System(differentiate(log(h_symbolic), [a, b]), variables=[a, b])
    log_hessian = System(vec(differentiate(log_gradient.expressions, [a, b])), variables=[a, b])
    for start in ([0.3, -0.7], ComplexF64[0.2+0.3im, -0.4+0.1im])
        reset_fiber_cache!(h)
        nearby = start + [1e-3, -2e-3]
        # Evaluate a fresh point, move to a nearby point, then repeat it.
        # Every result must agree with the defining polynomial, regardless
        # of evaluation history or previously written output buffers.
        for point in (start, nearby, nearby), offset in (nothing, ComplexF64[0.1, -0.2])
            fill!(u, ComplexF64(NaN))
            fill!(U, ComplexF64(NaN))
            ProjectedHypersurfaces.gradient_and_hessian!(u, U, h, point, offset)
            expected_gradient = log_gradient(point)
            if !isnothing(offset)
                expected_gradient -= offset
            end
            @test u ≈ expected_gradient atol = 1e-11
            @test U ≈ reshape(log_hessian(point), 2, 2) atol = 1e-10
        end
    end

    # Check realness
    p0 = randn(2)
    ProjectedHypersurfaces.evaluate_and_jacobian!(u, U, ∇r, p0)
    @test norm(imag(u)) < 1e-12
    @test norm(imag(U)) < 1e-12

    # Test forming the routing point homotopies
    p1 = zeros(2)
    q1 = randn(2)
    H = ProjectedHypersurfaces.RoutingPointsHomotopy(∇r, p1, q1)
    u = randn(ComplexF64, 2)
    U = randn(ComplexF64, 2, 2)
    x0 = randn(ComplexF64, 2)
    ProjectedHypersurfaces.evaluate_and_jacobian!(u, U, H, x0, 1.0)
    @test norm(∇r_symbolic(x0) - u) < 1e-12
    ProjectedHypersurfaces.evaluate_and_jacobian!(u, U, H, x0, 0.0)
    @test norm(∇r_symbolic(x0)-q1 - u) < 1e-12

    # The parameter-homotopy Taylor coefficient is exactly the negative of the
    # first parameter coefficient, independently of the current x series.
    p0 = randn(ComplexF64, 2)
    p1 = randn(ComplexF64, 2)
    tp = TaylorVector{2}(ComplexF64, 2)
    for i = 1:2
        tp[i] = (p0[i], p1[i])
    end
    taylor_coefficient = zeros(ComplexF64, 2)
    ProjectedHypersurfaces.taylor!(taylor_coefficient, Val(1), ∇r, x0, tp)
    @test taylor_coefficient == -p1

    # Test that the expansion of start solutions works
    ∇r = RoutingGradient(r)
    MS, H, S0, rhs0, k = ProjectedHypersurfaces._setup_monodromy_solver(∇r)
    @test length(MS.trackers) == Threads.nthreads()
    if Threads.nthreads() > 1
        worker_stats_before = ProjectedHypersurfaces._snapshot_worker_fiber_stats(MS)
        target_before = ProjectedHypersurfaces._fiber_tracking_counters(h.GC)
        copied_worker_h = ProjectedHypersurfaces._worker_hypersurfaces(MS.trackers[2])[1]
        @test copied_worker_h.GC !== h.GC
        copied_worker_h.GC.fiber_evaluations += 2
        copied_worker_h.GC.fiber_tracking_ns += UInt64(7)
        ProjectedHypersurfaces._merge_worker_fiber_stats!(∇r.r.H, MS, worker_stats_before)
        target_after = ProjectedHypersurfaces._fiber_tracking_counters(h.GC)
        @test target_after.evaluations == target_before.evaluations + 2
        @test target_after.tracking_ns == target_before.tracking_ns + UInt64(7)
    end
    S0, new_pts = ProjectedHypersurfaces._expand_start_solutions(
        ∇r, H, S0, rhs0, k;
        start_grid_width=10,
        start_grid_stepsize=1,
    )
    @test length(S0) >= 5 # should find at least five new points (one for each local maximum)
    @test all(norm(∇r(z)) < 1e-12 for z in new_pts) # all new points are routing points
    @test all(norm(∇r(z)-rhs0) < 1e-12 for z in S0) # all points are traced to solutions of ∇r=rhs0

    # Check critical points
    options = MonodromyOptions(target_solutions_count=2)
    routing_result = critical_points(r, start_grid_width=0, options=options)
    @test routing_result isa RoutingPointsResult
    @test all(norm(evaluate(∇r, z, zeros(ComplexF64, size(∇r, 1)))) < 1e-12 for z in routing_points(routing_result))
    @test all(norm.(∇r_symbolic.(complex_critical_points(routing_result))) .< 1e-12)

    # Test that trying to fix a seed via keyword (deprecated) gives an error
    @test_throws MethodError critical_points(r, start_grid_width=0, options=options, seed=0x12345678)

end;

@testset "Quadratic discriminant with lines" begin

    Random.seed!(12345)

    @var a b x
    F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])
    c = [10, 5]
    h = ProjectedHypersurface(F, [a, b])
    r = RoutingFunction(h; c=c, g=[a, b])
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
    ProjectedHypersurfaces.evaluate_and_jacobian!(u, U, ∇r, p0)
    ProjectedHypersurfaces.evaluate_and_jacobian!(u2, U2, ∇r_symbolic, p0)
    @test norm(u-u2) < 1e-12
    @test norm(U-U2) < 1e-12
    @test norm(∇r_symbolic(p0)-∇r(p0)) < 1e-12

    # Check critical points
    options = MonodromyOptions(target_solutions_count=2)
    routing_result = critical_points(r, start_grid_width=0, options=options)
    @test all(norm.(∇r_symbolic.(solutions(result(routing_result)))) .< 1e-12)

end;


@testset "Two discriminants" begin

    Random.seed!(12345)

    @var a b x z
    F1 = System([x^3 + a * x^2 + b * x + 1; 3 * x^2 + 2 * a * x + b], variables=[a, b, x])
    h1 = ProjectedHypersurface(F1, [a; b])
    F2 = System([z^2 + a * z - b; 2*z + a], variables=[a, z, b])
    h2 = ProjectedHypersurface(F2, [a; b])
    c = [7, 3]
    r = RoutingFunction([h1, h2]; c=c)
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
    options = MonodromyOptions(target_solutions_count=2)
    routing_result = critical_points(r, start_grid_width=0, options=options)
    @test all(norm.(∇r_symbolic.(solutions(result(routing_result)))) .< 1e-12)

end

@testset "Kuramoto discriminant" begin

    Random.seed!(12345)

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
    h = ProjectedHypersurface(F, [w[1], w[2]])
    r = RoutingFunction(h; c=C)
    ∇r = RoutingGradient(r)

    # Degree of the discriminant
    @test degree(h) == 12
    @test trace_test(h) < 1e-6

    h_symbolic = 314928 * w[1]^8 * w[2]^4 + 1259712 * w[1]^7 * w[2]^5 + 1889568 * w[1]^6 * w[2]^6 + 1259712 * w[1]^5 * w[2]^7 +
                            314928 * w[1]^4 * w[2]^8 + 139968 * w[1]^10 + 699840 * w[1]^9 * w[2] + 1277208 * w[1]^8 * w[2]^2 +
                            909792 * w[1]^7 * w[2]^3 - 279936 * w[1]^6 * w[2]^4 - 1084752 * w[1]^5 * w[2]^5 - 279936 * w[1]^4 * w[2]^6 +
                        909792 * w[1]^3 * w[2]^7 + 1277208 * w[1]^2 * w[2]^8 + 699840 * w[1] * w[2]^9 +
                        139968 * w[2]^10 - 96957 * w[1]^8 - 387828 * w[1]^7 * w[2] - 226962 * w[1]^6 * w[2]^2 + 676512 * w[1]^5 * w[2]^3 +
                    1128249 * w[1]^4 * w[2]^4 + 676512 * w[1]^3 * w[2]^5 - 226962 * w[1]^2 * w[2]^6 - 387828 * w[1] * w[2]^7 - 96957 * w[2]^8 +
                22680 * w[1]^6 + 68040 * w[1]^5 * w[2] - 20844 * w[1]^4 * w[2]^2 - 155088 * w[1]^3 * w[2]^3 - 20844 * w[1]^2 * w[2]^4 +
            68040 * w[1] * w[2]^5 + 22680 * w[2]^6 - 2298 * w[1]^4 - 4596 * w[1]^3 * w[2] - 6894 * w[1]^2 * w[2]^2 - 4596 * w[1] * w[2]^3 - 2298 * w[2]^4 +
        96 * w[1]^2 + 96 * w[1] * w[2] + 96 * w[2]^2 - 1

    r_symbolic = h_symbolic/((w[1] - C[1])^2 + (w[2] - C[2])^2 + 1)^7
    ∇r_symbolic = System(differentiate(log(r_symbolic), [w[1], w[2]]), variables=[w[1], w[2]]) |> fixed

    p0 = randn(ComplexF64, 2)
    @test norm(∇r_symbolic(p0)-∇r(p0)) < 1e-12

end;

@testset "Connect points" begin

    Random.seed!(12345)

    @var a b x
    F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])
    h = ProjectedHypersurface(F, [a, b])

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

    roadmap = gradient_roadmap(r, pts)
    roadmap_from_routing_result =
        gradient_roadmap(r, RoutingPointsResult(pts, nothing, nothing, r))

    @test roadmap isa GradientRoadmap
    @test sort(partition(roadmap)) == [[1, 2, 4], [3]]
    @test morse_indices(roadmap) == [1, 0, 0, 0]
    @test isempty(failed_info(roadmap))
    @test return_code(roadmap) == :success
    @test partition(roadmap_from_routing_result) == partition(roadmap)

end;

@testset "Region struct and membership" begin

    Random.seed!(12345)

    @var a b x
    F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])
    h = ProjectedHypersurface(F, [a, b])

    c = [13, 2]
    r = RoutingFunction(h; c=c)

    # Direct construction computes the Euler characteristic χ = ∑ᵢ (-1)^μᵢ
    C0 = Region([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [0, 1, 2], r, 7)
    @test C0 isa Region
    @test routing_points(C0) isa Vector{Vector{Float64}}
    @test morse_indices(C0) == [0, 1, 2]
    @test euler_characteristic(C0) == 1
    @test number(C0) == 7
    # Standalone construction numbers the routing points 1:n
    @test routing_point_indices(C0) == [1, 2, 3]
    # ...but the indices can be given explicitly
    C1 = Region([[1.0, 2.0], [3.0, 4.0]], [0, 1], r, 2; routing_point_indices=[4, 7])
    @test routing_point_indices(C1) == [4, 7]
    # There must be one Morse index per critical point
    @test_throws AssertionError Region([[1.0, 2.0]], [0, 1], r, 1)
    # ...and one routing point index per critical point
    @test_throws AssertionError Region(
        [[1.0, 2.0]], [0], r, 1; routing_point_indices=[1, 2],
    )

    # Same fixed routing points as in "Connect points" (deterministic partition)
    pts = [
        [-3.9180890683992278, -6.635887940807433],
        [13.040296300414134, 1.993819726256856],
        [3.2168112092392103, 8.082538361382138],
        [-12.339018441254076, -2.1071368134982302]
    ]

    roadmap = gradient_roadmap(r, pts)

    # The Region objects are stored on the roadmap itself
    Rs = regions(roadmap)
    @test Rs isa Vector{Region}
    @test length(Rs) == length(partition(roadmap))
    @test nregions(roadmap) == length(Rs)
    @test number.(Rs) == collect(1:length(Rs))

    # Every routing point lands in exactly one region
    @test sum(length(routing_points(C)) for C in Rs) == length(pts)

    # `partition` is derived from the regions' routing point indices
    @test partition(roadmap) == [routing_point_indices(C) for C in Rs]

    # Each region carries the data of its connected component
    idx = morse_indices(roadmap)
    for (C, component) in zip(Rs, partition(roadmap))
        @test routing_point_indices(C) == component
        @test routing_points(C) == [pts[j] for j in component]
        @test morse_indices(C) == [idx[j] for j in component]
        @test euler_characteristic(C) == sum(mu -> (-1)^mu, morse_indices(C); init=0)
    end

    # membership: each index-0 critical point flows back to its own region
    for C in Rs
        for (cp, mu) in zip(routing_points(C), morse_indices(C))
            mu == 0 || continue
            M = membership(Rs, cp)
            @test M isa Region
            @test number(M) == number(C)
        end
    end

    # membership of a non-critical point: the hypersurface here is the
    # discriminant a^2 - 4b = 0 of the quadratic x^2 + a*x + b, so the complement
    # has two regions, a^2 - 4b > 0 and a^2 - 4b < 0. The point pts[3] lies in the
    # latter, all other routing points in the former.
    @test pts[3][1]^2 - 4 * pts[3][2] < 0
    @test all(p[1]^2 - 4 * p[2] > 0 for p in pts[[1, 2, 4]])
    inside = only(C for C in Rs if pts[3] in routing_points(C))
    outside = only(C for C in Rs if pts[2] in routing_points(C))

    # (a, b) = (0, 5): a^2 - 4b = -20 < 0, so the same region as pts[3]
    M_in = membership(Rs, [0.0, 5.0])
    @test M_in isa Region
    @test number(M_in) == number(inside)
    # (a, b) = (0, -5): a^2 - 4b = 20 > 0, so the same region as pts[1], pts[2], pts[4]
    M_out = membership(Rs, [0.0, -5.0])
    @test M_out isa Region
    @test number(M_out) == number(outside)

    # membership can be called on the roadmap directly
    @test number(membership(roadmap, [0.0, 5.0])) == number(inside)
    @test number(membership(roadmap, [0.0, -5.0])) == number(outside)

    # membership on an empty vector of regions returns nothing
    @test membership(Region[], [0.0, 0.0]) === nothing

end;

@testset "Projected hypersurface sampling and membership" begin
    Random.seed!(12345)
    @var a b x
    F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])
    h = ProjectedHypersurface(F, [a, b])

    @test contains(h, [2.0, 1.0])
    @test !contains(h, [3.0, 1.0])
    @test_throws ArgumentError contains(h, [2.0])

    sample = sample_points(h, 6)
    @test all(contains.(Ref(h), sample))
end
@testset "Hypersurface evaluations for quadratic" begin

    Random.seed!(12345)

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
        [8*p[1]/(p[1]^2 - 4*p[2])^2 -16/(p[1]^2 - 4*p[2])^2]]
    @test Hess_log_abs_h(pt) - ProjectedHypersurfaces.gradient_and_hessian(h, pt)[2] |> norm < 1e-6

end
@testset "Noninjective projection" begin

    Random.seed!(12345)

    @var x y z
    F = System([z-x^2, y], variables=[x, y, z])
    h = ProjectedHypersurface(F, [y, z])

    @test degree(h) == 1 # the downstairs degree should be 1

    # h(y,z) = y (up to a constant) so gradient(h, [y, z]) = [1/y, 0]
    @test gradient(h, [2, 3]) - [1/2, 0] |> norm < 1e-6

end

@testset "Two components projecting to the same hypersurface" begin

    Random.seed!(12345)

    @var a, b, x
    F = System([a^2 - 4*b, (x - a + 1) * (x - a)], variables=[a, b, x])
    # V(F) has two irreducible components that project down to V(a^2-4b)
    h = ProjectedHypersurface(F, [a, b])
    @test degree(h) == 2

    # Check that the decomposition function only gives a single downstairs component
    components = decompose(h)
    @test length(components) == 1

end

@testset "Empty PWS" begin

    Random.seed!(12345)
    @var a b x
    F = System([(x - a) * (x - b); 2 * x - (a + b)], variables=[a, b, x])
    @test_throws "No witness points found." PseudoWitnessSet(F, 2)
end

@testset "Multiplicity detection" begin
    Random.seed!(12345)
    @var a, b, x
    F = System([a^2 - 4*b, (x - a + 1)^2 * (x - a)], variables=[a, b, x])
    @test_logs (:warn, "Irreducible component of higher multiplicity detected in the incidence variety.") match_mode=:any PseudoWitnessSet(F, 2)
end

@testset "Trace test" begin

    Random.seed!(12345)

    @var a b x
    F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])
    h = ProjectedHypersurface(F, [a, b])

    PWS = h.PWS
    @test trace_test(PWS) < 1e-10
    # Create an artificial failed PWS
    PWS_messed_up = PseudoWitnessSet(PWS.F,
        PWS.k,
        PWS.L,
        PWS.generic_witness_set,
        PWS.W[[1]],
        PWS.πW[[1]],
        PWS.tZ[[1]],
        PWS.tracker,
        PWS.track_report[[1]]
    )

    @test trace_test(PWS_messed_up) > 1e-6


end



@testset "Singular witness set" begin

    @var a b x
    F = System([x^2+a*x+b, 2*x+a], variables=[a, b, x])

    # Use a horizontal line to form the wintess set
    L = ProjectedHypersurfaces.Line(ComplexF64[2, 3], ComplexF64[1, 0])
    PWS = PseudoWitnessSet(F, 2; L=L)
    h = ProjectedHypersurface(F, [a, b]; PWS=PWS)

    p = [1, 0] # point with singular witness point

    @test h(p) |> abs < 1e-12
    @test h.PWS.tracker.state.singular
    @test !h.GC.fiber_valid

    @test all(isnan, gradient(h, p))


end