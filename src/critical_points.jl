
export critical_points

import HomotopyContinuation: MonodromyOptions, UniquePoints, EndgameTracker

"""
    critical_points(r, S0, rhs0; kwargs...)

Find critical points of the routing function using monodromy and gradient flow.
"""
function critical_points(
    r::RoutingFunction,
    S0::Union{AbstractVector{<:AbstractVector{<:Number}},Nothing} = nothing,
    rhs0::Union{AbstractVector{<:Number},Nothing} = nothing;
    verbose = true,
    start_grid_width = 5,
    start_grid_stepsize = 0.2,
    start_grid_center = nothing,
    monodromy_at_zero = false,
    options = MonodromyOptions(
        parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)),
        max_loops_no_progress = 15
    ),
    seed = rand(UInt32),
)

    ∇r = RoutingGradient(r)


    # Step 1: Setup monodromy solver
    MS, H, S0, rhs0, k = _setup_monodromy_solver(
        ∇r, S0, rhs0;
        monodromy_at_zero = monodromy_at_zero,
        options = options,
    )

    # Step 2: Expand start solutions via Newton's method and gradient flow
    S0, new_pts = _expand_start_solutions(
        ∇r, H, S0, rhs0, k;
        verbose = verbose,
        start_grid_width = start_grid_width,
        start_grid_stepsize = start_grid_stepsize,
        start_grid_center = start_grid_center,
        monodromy_at_zero = monodromy_at_zero,
    )

    # Step 3: Solve and trace to critical points
    return _solve_and_trace(
        MS, H, S0, rhs0, new_pts;
        monodromy_at_zero = monodromy_at_zero,
        start_grid_width = start_grid_width,
    )
end

"""
    _setup_monodromy_solver(∇r, S0, rhs0; monodromy_at_zero, options)

Set up the monodromy solver and initial start pair.
Returns (MS, H, S0, rhs0, k) where MS is the MonodromySolver, H is the homotopy,
S0 are the start solutions, rhs0 is the target parameters, and k is the number of variables.
"""
function _setup_monodromy_solver(
    ∇r::RoutingGradient,
    S0::Union{AbstractVector{<:AbstractVector{<:Number}},Nothing} = nothing,
    rhs0::Union{AbstractVector{<:Number},Nothing} = nothing;
    monodromy_at_zero = false,
    options = MonodromyOptions(
        parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)),
        max_loops_no_progress = 15
    ),
)
    k = size(∇r, 2) # number of variables
    p1 = zeros(k)
    q1 = randn(k)
    H = RoutingPointsHomotopy(∇r, p1, q1)

    ### Use monodromy to the system ∇r = rhs0 where we view the right-hand side are the parameters of the system
    egtracker = EndgameTracker(H)
    trackers = [egtracker]
    x₀ = zeros(ComplexF64, size(H, k))

    unique_points = UniquePoints(x₀, 1;)

    trace = zeros(ComplexF64, length(x₀) + 1, 3)
    P = Vector{ComplexF64}
    MS = HomotopyContinuation.MonodromySolver(
        trackers,
        HomotopyContinuation.MonodromyLoop{P}[],
        unique_points,
        ReentrantLock(),
        nothing,
        options,
        HomotopyContinuation.MonodromyStatistics(),
        trace,
        ReentrantLock(),
    )

    #### set up start pair
    if !monodromy_at_zero
        if isnothing(rhs0) || isnothing(S0)
            s0 = randn(ComplexF64, k)
            rhs0 = evaluate(∇r, s0)
            S0 = [s0]
        end
    else
        rhs0 = zeros(k)
        if isnothing(S0)
            S0 = Vector{ComplexF64}[]
        end
    end

    return MS, H, S0, rhs0, k
end

"""
    _expand_start_solutions(∇r, H, S0, rhs0, k; verbose, start_grid_width, start_stepsize, start_center, monodromy_at_zero)

Expand S0 by finding solutions of ∇r=0 through gradient descent and tracing to ∇r=rhs0.
Returns (S0, new_pts) where S0 is the expanded set of start solutions and new_pts are the points found via gradient flow.
"""
function _expand_start_solutions(
    ∇r::RoutingGradient,
    H::RoutingPointsHomotopy,
    S0::AbstractVector{<:AbstractVector{<:Number}},
    rhs0::AbstractVector{<:Number},
    k::Int;
    verbose = true,
    start_grid_width = 5,
    start_grid_stepsize = 0.2,
    start_grid_center = nothing,
    monodromy_at_zero = false,
)
    new_pts = Vector{ComplexF64}[]
    # Setting up grid
    if start_grid_width <= 0
        return S0, new_pts
    end

    if isnothing(start_grid_center)
        start_grid_center = zeros(k)
    end

    w = (start_grid_width / 2)
    grid = [
        (start_grid_center[i]-w):start_grid_stepsize:(start_grid_center[i]+w) for
        i = 1:k
    ]
    newton_w = 10*w
    newton_grid = [
        (start_grid_center[i]-newton_w):start_grid_stepsize:(start_grid_center[i]+newton_w) for
        i = 1:k
    ]

    # First we try to find start solutions via blindly applying Newton's method to ∇r=0.
    verbose && println("Expanding start solutions via Newton's method...")
    newton_success_count = 0
    start_pt = zeros(ComplexF64, k)
    ProgressMeter.@showprogress for start_point in Iterators.product(newton_grid...)
            start_pt .= start_point # this avoids allocations from splatting the tuple into the newton function
            # Newton's method on each initial guess
            try
                pt = newton(∇r, start_pt, rhs0; max_iters = 200) |> solution
                if norm(evaluate(∇r, pt, rhs0)) < 1e-10
                    newton_success_count += 1
                    push!(new_pts, pt)
                end
            catch e
                continue
            end

    end

    num_newton_pts = 0
    if length(new_pts) > 0
        new_pts = HC.unique_points(new_pts)
        num_newton_pts += length(new_pts)
    end

    # since the points obtained via Newton's method are already solutions to ∇r=rhs0, we can directly add them to S0 if monodromy_at_zero is false
    if !monodromy_at_zero
        S0 = HC.unique_points([S0; new_pts])
        # to avoid redundant work later, we remove the points found via Newton's method from new_pts so that we aren't tracing them using monodromy
        empty!(new_pts)
    end

    verbose && println("Successful Newton's method attempts: $(newton_success_count) out of $(length(newton_grid[1])^k) ($(round(newton_success_count / (length(newton_grid[1])^k) * 100, digits=2))%)")
    verbose && println("Found $num_newton_pts solutions to ∇r(z)=rhs0.")
    
    # Now we try gradient flow
    g(x, param, t) = real(evaluate(∇r, x))
    tspan = (0.0, 1e4)
    
    verbose && println("Expanding the set of start solutions via gradient flow...")

    gradient_success_count = 0
    start_pt = zeros(k)
    ProgressMeter.@showprogress for start_point in Iterators.product(grid...)
        try
            start_pt .= start_point # this avoids allocations from splatting the tuple into the ODEProblem
            prob = SciMLBase.ODEProblem(g, start_pt, tspan)
            sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
            convergence_point = last(sol.u)
            improved_point = newton(∇r, convergence_point) |> solution
            push!(new_pts, improved_point)
            gradient_success_count += 1                
        catch e
            continue
        end
    end
    if length(new_pts) > 0
        new_pts = HC.unique_points(new_pts)
    end
    verbose && println("Successful gradient flow attempts: $(gradient_success_count) out of $(length(grid[1])^k) ($(round(gradient_success_count / (length(grid[1])^k) * 100, digits=2))%)")
    verbose && println("Found $(length(new_pts)) routing points via gradient flow.")

    if !monodromy_at_zero
        start_parameters!(H, zeros(ComplexF64, length(rhs0)))
        target_parameters!(H, rhs0)
        S0_new_sols = HC.solve(H, new_pts) |> solutions
        number_of_old_sols = length(S0)
        S0 = HC.unique_points([S0; S0_new_sols])
        verbose && println(
            "Traced to $(length(S0)-number_of_old_sols) additional start solutions for the monodromy.",
        )
    else
        S0 = [S0; new_pts]
    end

    return S0, new_pts
end

"""
    _solve_and_trace(MS, H, S0, rhs0, new_pts; monodromy_at_zero, start_grid_width)

Perform monodromy solving and trace solutions to ∇r=0.
Returns (routing_points, result, mon_result).
"""
function _solve_and_trace(
    MS::HomotopyContinuation.MonodromySolver,
    H::RoutingPointsHomotopy,
    S0::AbstractVector{<:AbstractVector{<:Number}},
    rhs0::AbstractVector{<:Number},
    new_pts::AbstractVector{<:AbstractVector{<:Number}};
    monodromy_at_zero = false,
    start_grid_width = 5,
)
    ### Monodromy
    mon_result = monodromy_solve(MS, S0, rhs0, rand(UInt32);)

    ### Trace to ∇r=0
    if !monodromy_at_zero
        intermediate_rhs = randn(ComplexF64, length(rhs0))
        start_parameters!(H, rhs0)
        target_parameters!(H, intermediate_rhs)
        result_intermediate = HomotopyContinuation.solve(H, solutions(mon_result))
        start_parameters!(H, intermediate_rhs)
        target_parameters!(H, zeros(ComplexF64, length(rhs0)))
        result = HomotopyContinuation.solve(H, result_intermediate)
        routing_points = real_solutions(result)

        # Make sure none of the routing points found via gradient flow are lost
        if start_grid_width > 0
            routing_points = HC.unique_points([routing_points; real.(new_pts)])
        end
        return routing_points, result, mon_result
    else
        routing_points = real_solutions(results(mon_result))
        return routing_points, mon_result, mon_result
    end
end
