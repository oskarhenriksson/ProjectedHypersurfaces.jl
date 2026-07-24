
export critical_points

export RoutingPointsResult,
    PartitionResult,
    routing_points,
    complex_critical_points,
    result,
    monodromy_result

import HomotopyContinuation: MonodromyOptions, UniquePoints, EndgameTracker

struct StartSolutionExpansionResult{TS,TN}
    start_solutions::TS
    routing_points::TN
    interrupted::Bool
end

function Base.iterate(expansion::StartSolutionExpansionResult, state = 1)
    if state == 1
        return expansion.start_solutions, 2
    elseif state == 2
        return expansion.routing_points, 3
    end
    return nothing
end

Base.length(::StartSolutionExpansionResult) = 2

function _is_interrupt_exception(exception)
    if exception isa InterruptException
        return true
    elseif exception isa TaskFailedException
        return _is_interrupt_exception(exception.task.exception)
    elseif exception isa CompositeException
        return any(_is_interrupt_exception, exception.exceptions)
    end
    return false
end

function _run_interruptible(f; catch_interrupt)
    try
        f()
        return false
    catch exception
        catch_interrupt && _is_interrupt_exception(exception) || rethrow()
        return true
    end
end

function _solve_homotopy(H, starts; catch_interrupt)
    starts = collect(starts)
    result = HomotopyContinuation.solve(H, starts; catch_interrupt = catch_interrupt)
    interrupted = catch_interrupt && length(result) < length(starts)
    return result, interrupted
end

_monodromy_was_interrupted(result) =
    hasproperty(result, :returncode) && result.returncode == :interrupted

@doc raw"""
    critical_points(r, S0, rhs0; kwargs...)

Find critical points of the routing function using monodromy and gradient flow.
Returns a [`RoutingPointsResult`](@ref).

Set `expand_start_solutions = false` to disable all start-solution expansion.
The Newton-grid and gradient-flow expansion phases can be controlled independently
with `expand_start_solutions_newton` and `expand_start_solutions_gradient_flow`.
When Newton expansion is disabled, gradient-flow endpoints are accepted only when
their residual is below `1e-10`; no Newton refinement is performed.

If `catch_interrupt = true`, an interrupted phase returns the valid routing points
found so far in a result whose [`return_code`](@ref) is `:interrupted`.
"""
function critical_points(
    r::RoutingFunction,
    S0::Union{AbstractVector{<:AbstractVector{<:Number}},Nothing} = nothing,
    rhs0::Union{AbstractVector{<:Number},Nothing} = nothing;
    verbose = true,
    start_grid_width = 5,
    start_grid_stepsize = 0.2,
    start_grid_center = nothing,
    expand_start_solutions = true,
    expand_start_solutions_newton = true,
    expand_start_solutions_gradient_flow = true,
    catch_interrupt = true,
    monodromy_at_zero = false,
    options = MonodromyOptions(
        parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)),
        max_loops_no_progress = 15
    ),
)

    ∇r = RoutingGradient(r)

    # Step 1: Setup monodromy solver
    MS, H, S0, rhs0, k = _setup_monodromy_solver(
        ∇r, S0, rhs0;
        monodromy_at_zero = monodromy_at_zero,
        options = options,
    )

    # Step 2: Expand start solutions via Newton's method and gradient flow
    expansion = _expand_start_solutions(
        ∇r, H, S0, rhs0, k;
        verbose = verbose,
        start_grid_width = start_grid_width,
        start_grid_stepsize = start_grid_stepsize,
        start_grid_center = start_grid_center,
        expand_start_solutions = expand_start_solutions,
        expand_start_solutions_newton = expand_start_solutions_newton,
        expand_start_solutions_gradient_flow = expand_start_solutions_gradient_flow,
        catch_interrupt = catch_interrupt,
        monodromy_at_zero = monodromy_at_zero,
    )
    S0, new_pts = expansion
    if expansion.interrupted
        verbose &&
            @warn "Interrupted while expanding start solutions. Returning routing points found so far."
        return RoutingPointsResult(real.(new_pts), nothing, nothing, :interrupted)
    end

    # Step 3: Solve and trace to critical points
    return _solve_and_trace(
        MS, H, S0, rhs0, new_pts;
        monodromy_at_zero = monodromy_at_zero,
        expand_start_solutions = expand_start_solutions && start_grid_width > 0,
        catch_interrupt = catch_interrupt,
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
    expand_start_solutions = true,
    expand_start_solutions_newton = true,
    expand_start_solutions_gradient_flow = true,
    catch_interrupt = true,
    monodromy_at_zero = false,
)
    new_pts = Vector{ComplexF64}[]
    # Setting up grid
    if !expand_start_solutions || start_grid_width <= 0 ||
       (!expand_start_solutions_newton && !expand_start_solutions_gradient_flow)
        return StartSolutionExpansionResult(S0, new_pts, false)
    end

    if isnothing(start_grid_center)
        start_grid_center = zeros(k)
    end

    w = (start_grid_width / 2)
    interrupted = false

    # First we try to find start solutions via blindly applying Newton's method to ∇r=0.
    newton_pts = Vector{ComplexF64}[]
    newton_success_count = 0
    newton_total_count = 0
    if expand_start_solutions_newton
        newton_w = 10*w
        newton_grid = [
            (start_grid_center[i]-newton_w):start_grid_stepsize:(start_grid_center[i]+newton_w) for
            i = 1:k
        ]
        newton_total_count = prod(length.(newton_grid))
        verbose && println("Expanding start solutions via Newton's method...")
        start_pt = zeros(ComplexF64, k)
        interrupted = _run_interruptible(; catch_interrupt = catch_interrupt) do
            ProgressMeter.@showprogress for start_point in Iterators.product(newton_grid...)
            start_pt .= start_point # this avoids allocations from splatting the tuple into the newton function
            # Newton's method on each initial guess
            try
                pt = newton(∇r, start_pt, rhs0; max_iters = 200) |> solution
                if norm(evaluate(∇r, pt, rhs0)) < 1e-10
                    newton_success_count += 1
                    push!(newton_pts, pt)
                end
            catch exception
                _is_interrupt_exception(exception) && rethrow()
                continue
            end
            end
        end
    end

    num_newton_pts = 0
    if !isempty(newton_pts)
        newton_pts = HC.unique_points(newton_pts)
        num_newton_pts = length(newton_pts)
    end

    # since the points obtained via Newton's method are already solutions to ∇r=rhs0, we can directly add them to S0 if monodromy_at_zero is false
    if !monodromy_at_zero
        S0 = HC.unique_points([S0; newton_pts])
    elseif !isempty(newton_pts)
        new_pts = HC.unique_points([new_pts; newton_pts])
    end

    if expand_start_solutions_newton
        verbose && println(
            "Successful Newton's method attempts: $(newton_success_count) out of $(newton_total_count) ($(round(newton_success_count / newton_total_count * 100, digits=2))%)",
        )
        verbose && println("Found $num_newton_pts solutions to ∇r(z)=rhs0.")
    end

    if interrupted
        return StartSolutionExpansionResult(S0, new_pts, true)
    end

    # Now we try gradient flow
    gradient_success_count = 0
    gradient_total_count = 0
    if expand_start_solutions_gradient_flow
        grid = [
            (start_grid_center[i]-w):start_grid_stepsize:(start_grid_center[i]+w) for
            i = 1:k
        ]
        gradient_total_count = prod(length.(grid))
        g(x, param, t) = real(evaluate(∇r, x))
        tspan = (0.0, 1e4)

        verbose && println("Expanding the set of start solutions via gradient flow...")

        start_pt = zeros(k)
        interrupted = _run_interruptible(; catch_interrupt = catch_interrupt) do
            ProgressMeter.@showprogress for start_point in Iterators.product(grid...)
                try
                    start_pt .= start_point # this avoids allocations from splatting the tuple into the ODEProblem
                    prob = SciMLBase.ODEProblem(g, start_pt, tspan)
                    sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
                    convergence_point = last(sol.u)
                    candidate_point =
                        expand_start_solutions_newton ?
                        newton(∇r, convergence_point) |> solution :
                        ComplexF64.(convergence_point)
                    if norm(evaluate(∇r, candidate_point)) < 1e-10
                        push!(new_pts, candidate_point)
                        gradient_success_count += 1
                    end
                catch exception
                    _is_interrupt_exception(exception) && rethrow()
                    continue
                end
            end
        end
        if !isempty(new_pts)
            new_pts = HC.unique_points(new_pts)
        end
        verbose && println(
            "Successful gradient flow attempts: $(gradient_success_count) out of $(gradient_total_count) ($(round(gradient_success_count / gradient_total_count * 100, digits=2))%)",
        )
        verbose && println("Found $(length(new_pts)) routing points via gradient flow.")
    end

    if interrupted
        return StartSolutionExpansionResult(S0, new_pts, true)
    end

    if !monodromy_at_zero
        if !isempty(new_pts)
            start_parameters!(H, zeros(ComplexF64, length(rhs0)))
            target_parameters!(H, rhs0)
            S0_result, interrupted =
                _solve_homotopy(H, new_pts; catch_interrupt = catch_interrupt)
            S0_new_sols = solutions(S0_result)
            number_of_old_sols = length(S0)
            S0 = HC.unique_points([S0; S0_new_sols])
            verbose && println(
                "Traced to $(length(S0)-number_of_old_sols) additional start solutions for the monodromy.",
            )
            if interrupted
                return StartSolutionExpansionResult(S0, new_pts, true)
            end
        end
    else
        S0 = [S0; new_pts]
    end

    return StartSolutionExpansionResult(S0, new_pts, false)
end

"""
    _solve_and_trace(MS, H, S0, rhs0, new_pts; monodromy_at_zero, expand_start_solutions)

Perform monodromy solving and trace solutions to ∇r=0.
Returns a [`RoutingPointsResult`](@ref).
"""
function _solve_and_trace(
    MS::HomotopyContinuation.MonodromySolver,
    H::RoutingPointsHomotopy,
    S0::AbstractVector{<:AbstractVector{<:Number}},
    rhs0::AbstractVector{<:Number},
    new_pts::AbstractVector{<:AbstractVector{<:Number}};
    monodromy_at_zero = false,
    expand_start_solutions = true,
    catch_interrupt = true,
)
    ### Monodromy
    mon_result =
        monodromy_solve(MS, S0, rhs0, rand(UInt32); catch_interrupt = catch_interrupt)
    interrupted = _monodromy_was_interrupted(mon_result)

    ### Trace to ∇r=0
    if !monodromy_at_zero
        intermediate_rhs = randn(ComplexF64, length(rhs0))
        start_parameters!(H, rhs0)
        target_parameters!(H, intermediate_rhs)
        result_intermediate, trace_interrupted = _solve_homotopy(
            H,
            solutions(mon_result);
            catch_interrupt = catch_interrupt,
        )
        if trace_interrupted
            routing_points =
                expand_start_solutions ? real.(new_pts) : Vector{Vector{Float64}}()
            return RoutingPointsResult(
                routing_points,
                nothing,
                mon_result,
                :interrupted,
            )
        end
        start_parameters!(H, intermediate_rhs)
        target_parameters!(H, zeros(ComplexF64, length(rhs0)))
        result, trace_interrupted = _solve_homotopy(
            H,
            solutions(result_intermediate);
            catch_interrupt = catch_interrupt,
        )
        routing_points = real_solutions(result)

        # Make sure none of the routing points found via gradient flow are lost
        if expand_start_solutions
            routing_points = HC.unique_points([routing_points; real.(new_pts)])
        end
        code = interrupted || trace_interrupted ? :interrupted : :success
        return RoutingPointsResult(routing_points, result, mon_result, code)
    else
        routing_points = real_solutions(results(mon_result))
        code = interrupted ? :interrupted : :success
        return RoutingPointsResult(routing_points, mon_result, mon_result, code)
    end
end



import HomotopyContinuation:
    solutions, real_solutions, nsolutions, results, nresults

@doc raw"""
    RoutingPointsResult

Result returned by [`critical_points`](@ref). Use [`routing_points`](@ref) for the
real routing points, [`result`](@ref) for the final result from tracking to to `∇r = 0`,
[`monodromy_result`](@ref) for the underlying monodromy computation, and
[`return_code`](@ref) to distinguish a complete result from an interrupted one.
"""
struct RoutingPointsResult{P,T,M}
    routing_points::P
    result::T
    monodromy_result::M
    return_code::Symbol
end

RoutingPointsResult(routing_points, result, monodromy_result) =
    RoutingPointsResult(routing_points, result, monodromy_result, :success)

@doc raw"""
    routing_points(result::RoutingPointsResult)

Return the real critical points used for routing.
"""
routing_points(R::RoutingPointsResult) = R.routing_points

@doc raw"""
    result(result::RoutingPointsResult)

Return the final HomotopyContinuation result obtained by tracing to ∇r = 0.
"""
result(R::RoutingPointsResult) = R.result

@doc raw"""
    complex_critical_points(result::RoutingPointsResult)

Return the complex critical points of the routing function (the real points are routing points).
"""
complex_critical_points(R::RoutingPointsResult) =
    isnothing(R.result) ? Vector{Vector{ComplexF64}}() : solutions(R.result)

@doc raw"""
    monodromy_result(result::RoutingPointsResult)

Return the underlying monodromy computation result.
"""
monodromy_result(R::RoutingPointsResult) = R.monodromy_result

@doc raw"""
    return_code(result::RoutingPointsResult)

Return `:success` when critical-point computation completed, or `:interrupted`
when the result contains only the valid routing points found before an interrupt.
"""
return_code(R::RoutingPointsResult) = R.return_code

function Base.show(io::IO, R::RoutingPointsResult)
    npts = length(routing_points(R))
    ncomplex = length(complex_critical_points(R))
    header =
        "RoutingPointsResult with $npts routing point(s) and $ncomplex complex critical point(s)"
    println(io, header)
    println(io, "="^length(header))
    print(io, "• return_code → :$(return_code(R))")
end
