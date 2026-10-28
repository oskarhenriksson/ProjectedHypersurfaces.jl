mutable struct GradientCache{T}
    v0::Vector{T}
    line_hypersurface_intersections::Vector{Vector{T}}
    JsuF::HC.CompiledSystem
    JPF::HC.CompiledSystem
    adjoint_gradient_system::HC.CompiledSystem
    contracted_hessian_system::HC.CompiledSystem
    S::Vector{T}
    X::Vector{T}
    Uvals::Matrix{T}
    SP::Matrix{T}
    SB::Matrix{T}
    UP::Array{T,3}
    UB::Array{T,3}
    rhs1::Matrix{T}
    adjoint_rhs::Vector{T}
    JsuF_vals::Vector{T}
    JPF_vals::Vector{T}
    adjoint_gradient_input::Vector{T}
    adjoint_gradient_vals::Vector{T}
    contracted_hessian_input::Vector{T}
    contracted_hessian_vals::Vector{T}
    JsuF_temp::Matrix{T}
    JPF_temp::Matrix{T}
    ipiv::Vector{LinearAlgebra.LAPACK.BlasInt} # allocation for pivot for lu! in place linear solving
    gradient_temp::Vector{T}
    Hess_temp::Matrix{T}
    # A path-local moving copy of the pseudo-witness fibre.  Consecutive calls made by
    # an outer path tracker are close, so continuing this fibre is much cheaper than
    # restarting at the original witness slice for every evaluation.
    fiber_point::Vector{T}
    fiber_solutions::Vector{Vector{T}}
    fiber_scratch::Vector{Vector{T}}
    warm_fiber_tracking::Bool
    fiber_valid::Bool
    fiber_regular::Bool
    fiber_evaluations::Int
    fiber_exact_hits::Int
    fiber_warm_tracks::Int
    fiber_cold_tracks::Int
    fiber_fallbacks::Int
    fiber_failures::Int
    fiber_tracking_ns::UInt64
end

function compute_systems(F, n, k, B)
    N = n - k + 1
    @unique_var uval[1:n-k] α[1:k] β[1:k] t λ[1:N] xp[1:N,1:k] xb[1:N,1:k]
    F_on_line = F([α + (1 / t) * β; uval])
    v = vcat(t, uval)
    vars = vcat(t, uval, α)

    ∇v = map(v) do vi
        HC.ModelKit.differentiate(F_on_line, vi) 
    end
    ∇α = map(α) do αi
        HC.ModelKit.differentiate(F_on_line, αi)
    end

    JsuF_exprs = map(∇v) do ∇vi
        evaluate(∇vi, β => B)
    end
    JPF_exprs = map(∇α) do ∇vi
        evaluate(∇vi, β => B)
    end

    # Store each derivative block in one compiled system to avoid many tiny system evaluations.
    JsuF = CompiledSystem(System(reduce(vcat, JsuF_exprs), variables = vars))
    JPF = CompiledSystem(System(reduce(vcat, JPF_exprs), variables = vars))

    # Compile the contractions that are actually needed.  This avoids evaluating
    # full G_xx, G_xp, G_xβ and G_pβ tensors at every fibre point.
    adjoint_gradient_exprs = map(β) do βb
        sum(1:N; init = 0) do i
            λ[i] * HC.ModelKit.differentiate(F_on_line[i], βb)
        end
    end
    adjoint_gradient_exprs = evaluate.(adjoint_gradient_exprs, Ref(β => B))
    adjoint_gradient_system = CompiledSystem(System(
        adjoint_gradient_exprs,
        variables = [vars; λ],
    ))

    contracted_hessian_exprs = [begin
        expression = 0
        for i = 1:N
            residual_i = HC.ModelKit.differentiate(
                HC.ModelKit.differentiate(F_on_line[i], α[a]), β[b],
            )
            for r = 1:N
                residual_i += HC.ModelKit.differentiate(
                    HC.ModelKit.differentiate(F_on_line[i], v[r]), β[b],
                ) * xp[r, a]
                residual_i += HC.ModelKit.differentiate(
                    HC.ModelKit.differentiate(F_on_line[i], v[r]), α[a],
                ) * xb[r, b]
                for c = 1:N
                    residual_i += HC.ModelKit.differentiate(
                        HC.ModelKit.differentiate(F_on_line[i], v[r]), v[c],
                    ) * xp[r, a] * xb[c, b]
                end
            end
            expression += λ[i] * residual_i
        end
        evaluate(expression, β => B)
    end for a = 1:k, b = 1:k]
    contracted_hessian_system = CompiledSystem(System(
        vec(contracted_hessian_exprs),
        variables = [vars; λ; vec(xp); vec(xb)],
    ))

    return JsuF, JPF, adjoint_gradient_system, contracted_hessian_system

end

function GradientCache(PWS)
    d = degree(PWS)
    k = n_projection_variables(PWS)
    F = PWS.F
    L = PWS.L
    N, n = size(F)

    @assert N == n-k+1 "Unexpected length of system"

    # The restricted tracker stores [t; w], so each tracked point has length n - k + 1.
    line_hypersurface_intersections = [zeros(ComplexF64, n - k + 1) for _ in 1:d]
  
    @unique_var t, p[1:k]

    S = zeros(ComplexF64, d)
    X = zeros(ComplexF64, k)
    Uvals = zeros(ComplexF64, n - k, d)
    SP = zeros(ComplexF64, d, k)
    SB = zeros(ComplexF64, d, k)
    UP = zeros(ComplexF64, d, n - k, k)
    UB = zeros(ComplexF64, d, n - k, k)
    JsuF, JPF, adjoint_gradient_system, contracted_hessian_system =
        compute_systems(F, n, k, L.direction)


    # 
    rhs1 = zeros(ComplexF64, N, 2*k)  
    adjoint_rhs = zeros(ComplexF64, N)
    JsuF_vals = zeros(ComplexF64, N * N)
    JPF_vals = zeros(ComplexF64, N * k)
    adjoint_gradient_input = zeros(ComplexF64, N + k + N)
    adjoint_gradient_vals = zeros(ComplexF64, k)
    contracted_hessian_input = zeros(ComplexF64, N + k + N + 2 * N * k)
    contracted_hessian_vals = zeros(ComplexF64, k * k)

    JsuF_temp = zeros(ComplexF64, N, 1+n-k)
    JPF_temp = zeros(ComplexF64, N, k)

    ipiv = Vector{LinearAlgebra.LAPACK.BlasInt}(undef, min(size(JsuF_temp,1), size(JsuF_temp,2)))

    gradient_temp = zeros(ComplexF64, k)
    Hess_temp = zeros(ComplexF64, k, k)

    fiber_point = zeros(ComplexF64, k)
    fiber_solutions = [copy(z) for z in PWS.tZ]
    fiber_scratch = [similar(z) for z in PWS.tZ]

    v0 = randn(ComplexF64, n+1)

    GradientCache{ComplexF64}(v0, 
                    line_hypersurface_intersections,
                    JsuF,
                    JPF,
                    adjoint_gradient_system,
                    contracted_hessian_system,
                    S, 
                    X, 
                    Uvals, 
                    SP, 
                    SB, 
                    UP, 
                    UB, 
                    rhs1, 
                    adjoint_rhs,
                    JsuF_vals,
                    JPF_vals,
                    adjoint_gradient_input,
                    adjoint_gradient_vals,
                    contracted_hessian_input,
                    contracted_hessian_vals,
                    JsuF_temp, 
                    JPF_temp, 
                    ipiv,
                    gradient_temp, 
                    Hess_temp,
                    fiber_point,
                    fiber_solutions,
                    fiber_scratch,
                    true,
                    false,
                    false,
                    0, 0, 0, 0, 0, 0,
                    UInt64(0)
                )
end

@inline function _same_fiber_point(a, b)
    length(a) == length(b) || return false
    @inbounds for i in eachindex(a, b)
        a[i] == b[i] || return false
    end
    true
end

#Track the pseudo-witness fibre to `p`, reusing the most recent fibre when possible.
# Updates are transactional: a failed warm track is discarded and retried from the
# original witness slice.  An incomplete fibre is never used for differentiation.
# Return false on numerical tracking failure so evaluators can signal an
# undefined value to the outer tracker without aborting the other paths.
function track!(GC::GradientCache, PWS::PseudoWitnessSet, p)
    GC.fiber_evaluations += 1

    if GC.warm_fiber_tracking && GC.fiber_valid && _same_fiber_point(GC.fiber_point, p)
        GC.fiber_exact_hits += 1
        for i in eachindex(GC.line_hypersurface_intersections)
            copyto!(GC.line_hypersurface_intersections[i], GC.fiber_solutions[i])
            PWS.track_report[i] = true
        end
        get_s_and_Uvals!(GC.Uvals, GC.S, GC, PWS)
        return true
    end

    t0 = time_ns()
    success = false
    if GC.warm_fiber_tracking && GC.fiber_valid
        GC.fiber_warm_tracks += 1
        result = _track_fiber_from!(
            GC.fiber_scratch, PWS, GC.fiber_solutions, GC.fiber_point, p,
        )
        success, regular = result.succeeded, result.regular
        if !success
            GC.fiber_fallbacks += 1
        end
    end

    if !success
        GC.fiber_cold_tracks += 1
        result = _track_fiber_from!(GC.fiber_scratch, PWS, PWS.tZ, PWS.L.point, p)
        success, regular = result.succeeded, result.regular
    end
    GC.fiber_tracking_ns += UInt64(time_ns() - t0)

    if !success
        GC.fiber_failures += 1
        GC.fiber_valid = false
        GC.fiber_regular = false
        return false
    end

    copyto!(GC.fiber_point, p)
    for i in eachindex(GC.fiber_solutions)
        copyto!(GC.fiber_solutions[i], GC.fiber_scratch[i])
        copyto!(GC.line_hypersurface_intersections[i], GC.fiber_scratch[i])
        PWS.track_report[i] = true
    end
    GC.fiber_regular = regular
    GC.fiber_valid = GC.warm_fiber_tracking && regular
    get_s_and_Uvals!(GC.Uvals, GC.S, GC, PWS)
    true
end

function set_warm_fiber_tracking!(GC::GradientCache, enabled::Bool)
    GC.warm_fiber_tracking = enabled
    GC.fiber_valid = false
    GC
end

function reset_fiber_cache!(GC::GradientCache)
    GC.fiber_valid = false
    GC.fiber_evaluations = 0
    GC.fiber_exact_hits = 0
    GC.fiber_warm_tracks = 0
    GC.fiber_cold_tracks = 0
    GC.fiber_fallbacks = 0
    GC.fiber_failures = 0
    GC.fiber_tracking_ns = UInt64(0)
    GC
end

fiber_tracking_stats(GC::GradientCache) = (
    enabled = GC.warm_fiber_tracking,
    evaluations = GC.fiber_evaluations,
    exact_hits = GC.fiber_exact_hits,
    warm_tracks = GC.fiber_warm_tracks,
    cold_tracks = GC.fiber_cold_tracks,
    fallbacks = GC.fiber_fallbacks,
    failures = GC.fiber_failures,
    tracking_seconds = Float64(GC.fiber_tracking_ns) / 1e9,
)

