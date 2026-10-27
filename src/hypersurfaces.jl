export ProjectedHypersurface, 
evaluate,
gradient,
hessian,
degree,
trace_test,
sample_points,
decompose,
fiber_tracking_stats,
reset_fiber_cache!,
set_warm_fiber_tracking!

@doc raw"""
    ProjectedHypersurface{TC} <: HC.AbstractSystem


A hypersurface $\mathcal{H}$ in $\mathbb{C}^k$ that arises through projection of $(k-1)$-dimensional variety in a higher-dimensional ambient space.

The hypersurface $\mathcal{H}$ is represented by a pseudowitness set.
"""
struct ProjectedHypersurface{TC} <: HC.AbstractSystem
    PWS::PseudoWitnessSet
    projection_vars::Vector{HC.Variable}
    GC::TC
end
function ProjectedHypersurface(
    F,
    projection_vars;
    PWS::Union{Nothing, PseudoWitnessSet} = nothing,
    start_system_for_PWS::Symbol = :polyhedral,
    compile::Union{Bool,Symbol} = :mixed
)

    all_vars = ModelKit.variables(F)
    x_vars = setdiff(all_vars, projection_vars)
    F_ordered = System(F.expressions, variables = [projection_vars; x_vars])
    k = length(projection_vars)
    if isnothing(PWS)
        PWS = PseudoWitnessSet(F_ordered, k; start_system = start_system_for_PWS, compile = compile)
    end
    GC = GradientCache(PWS)

    ProjectedHypersurface{typeof(GC)}(PWS, projection_vars, GC)
end

degree(h::ProjectedHypersurface) = degree(h.PWS)


@doc raw"""
    trace_test(h::ProjectedHypersurface)

Performs a trace test to vertify completeness of the underlying pseduo-witness set and therefore correctness of the degree.
A value close to zero (e.g., in the order to 1e-16) indicates that the pseudo-witness set is likely complete.

See [`trace_test(::PseudoWitnessSet)`](@ref) for details.

"""
trace_test(h::ProjectedHypersurface) = trace_test(h.PWS)

@doc raw"""
    sample_points(h::ProjectedHypersurface, N::Int)

Generate a sample of `N` points from a projected hypersurface `h`. See also [`sample_points(::PseudoWitnessSet,::Int)`](@ref)
"""
sample_points(h::ProjectedHypersurface, N::Int) = sample_points(h.PWS, N::Int)

Base.show(io::IO, h::ProjectedHypersurface) = println(io, "Projected hypersurface of degree $(degree(h)) in ambient dimension $(nvariables(h))")
    
ModelKit.variables(h::ProjectedHypersurface{TC}) where {TC} = h.projection_vars
ModelKit.nvariables(h::ProjectedHypersurface{TC}) where {TC} = length(h.projection_vars)

@doc raw"""
    decompose(h::ProjectedHypersurface)

Decomposes the projected hypersurface `h` into its irreducible components. 
Returns a vector of [`ProjectedHypersurface`](@ref) objects, one for each component.

See [`decompose(::PseudoWitnessSet)`](@ref) for details.

"""
function decompose(h::ProjectedHypersurface)
    PWS_components = decompose(h.PWS)
    map(PWS_components) do PWS_comp
        ProjectedHypersurface(PWS_comp.F, h.projection_vars; PWS = PWS_comp)
    end
end


function Base.contains(
    h::ProjectedHypersurface,
    p::AbstractVector;
    atol = sqrt(eps(Float64)),
    residual_atol = atol,
)
    length(p) == nvariables(h) || throw(ArgumentError("Expected $(nvariables(h)) coordinates."))

    PWS, GC = h.PWS, h.GC
    track!(GC, PWS, p) || error("Cannot determine membership: pseudo-witness fibre tracking failed.")

    direction_norm = norm(PWS.L.direction)
    for (track_succeeded, sol) in zip(PWS.track_report, GC.line_hypersurface_intersections)
        if !track_succeeded || !all(isfinite, sol)
            continue
        end

        t = sol[1]
        w = sol[2:end]
        if abs(t) * direction_norm <= atol && norm(PWS.F([p; w])) <= residual_atol
            return true
        end
    end

    false
end

function evaluate(h::ProjectedHypersurface{TC}, x, p = nothing) where {TC}
    PWS, GC = h.PWS, h.GC

    S = GC.S

    track!(GC, PWS, x) || return NaN

    u = 0.0

    #@inbounds @simd 
    for si in S 
        u += -log(abs(si))
    end

    u
end

function gradient!(u, h::ProjectedHypersurface{TC}, x, p = nothing) where {TC}
    
    PWS, GC = h.PWS, h.GC

    # Use cached symbolic objects and arrays
    JsuF = GC.JsuF
    adjoint_gradient_system = GC.adjoint_gradient_system

    v0 = GC.v0
    S = GC.S
    Uvals = GC.Uvals
    adjoint_rhs = GC.adjoint_rhs
    JsuF_vals = GC.JsuF_vals
    adjoint_gradient_input = GC.adjoint_gradient_input
    adjoint_gradient_vals = GC.adjoint_gradient_vals

    N, n = size(PWS.F)
    k = n_projection_variables(PWS)

    u .= zero(eltype(u))

    # `track!` restores or computes both the tracked intersections and the cached S/Uvals data.
    if !track!(GC, PWS, x)
        fill!(u, NaN)
        return nothing
    end

    # Adjoint implicit differentiation.  If J = G_(s,u) and Jᵀλ = e₁, then
    # -∂s/∂β = λᵀG_β, which is precisely one fibre contribution to
    # ∇log|h|.  This needs one solve rather than 2k forward sensitivity solves.
    for i = 1:length(S)

        if !PWS.track_report[i] # skip if i-th track failed
            continue
        end

        _fill_v0!(v0, S, Uvals, x, i)

        # Evaluate the fused derivative blocks and unpack them into the working arrays.
        JsuF_temp = GC.JsuF_temp
        _evaluate_fused_columns!(JsuF_temp, JsuF_vals, JsuF, v0, N, N)

        fill!(adjoint_rhs, zero(ComplexF64))
        adjoint_rhs[1] = one(ComplexF64)
        _, ipiv, info = LinearAlgebra.LAPACK.getrf!(JsuF_temp, GC.ipiv)
        if info != 0
            fill!(u, NaN)
            return nothing
        end
        LinearAlgebra.LAPACK.getrs!('T', JsuF_temp, ipiv, adjoint_rhs)

        _fill_adjoint_gradient_input!(adjoint_gradient_input, v0, adjoint_rhs)
        evaluate!(adjoint_gradient_vals, adjoint_gradient_system, adjoint_gradient_input)
        u .+= adjoint_gradient_vals
    end


    if !isnothing(p)
        @inbounds for ii = 1:length(u)
            u[ii] -= p[ii]
        end
    end


    nothing
end
function gradient(h::ProjectedHypersurface{TC}, x, p = nothing) where {TC}
    k = nvariables(h)
    u = zeros(ComplexF64, k)
    gradient!(u, h, x, p)
    u
end

function gradient_and_hessian!(u, U, h::ProjectedHypersurface{TC}, x, p = nothing) where {TC}

    PWS, GC = h.PWS, h.GC

    # Use cached symbolic objects and arrays
    JsuF = GC.JsuF
    JPF = GC.JPF
    contracted_hessian_system = GC.contracted_hessian_system

    # Preallocated temporaries and cached LU data keep the Hessian path allocation-free.
    JsuF_lu = GC.JsuF_lu
    JsuF_ipiv = GC.JsuF_ipiv
    v0 = GC.v0
    S = GC.S
    Uvals = GC.Uvals
    SP = GC.SP
    SB = GC.SB
    UP = GC.UP
    UB = GC.UB
    rhs1, adjoint_rhs = GC.rhs1, GC.adjoint_rhs
    JsuF_vals, JPF_vals, JBF_vals = GC.JsuF_vals, GC.JPF_vals, GC.JBF_vals
    contracted_hessian_input = GC.contracted_hessian_input
    contracted_hessian_vals = GC.contracted_hessian_vals

    k = n_projection_variables(PWS)
    N, n = size(PWS.F)

    u .= zero(eltype(u))
    U .= zero(eltype(U))

    # `track!` restores or computes both the tracked intersections and the cached S/Uvals data.
    if !track!(GC, PWS, x)
        fill!(u, NaN)
        fill!(U, NaN)
        return nothing
    end

    #Obtain gradients of S and U with respect to p and β
    for i = 1:length(S)

        if !PWS.track_report[i] # skip if i-th track failed
            continue
        end

        _fill_v0!(v0, S, Uvals, x, i)

        # Evaluate the fused first-derivative blocks and unpack them into working storage.
        JsuF_temp = GC.JsuF_temp
        _evaluate_fused_columns!(JsuF_temp, JsuF_vals, JsuF, v0, N, N)

        JPF_temp = GC.JPF_temp
        _evaluate_fused_columns!(JPF_temp, JPF_vals, JPF, v0, N, k)

        # G_b = G_p / s for G(s,u;p,b) = F(p + b/s,u). Cache the
        # derived block, but solve only for z_p; z_b = z_p / s.
        JBF_temp = GC.JBF_temp
        @inbounds for col = 1:k, row = 1:N
            value = JPF_temp[row, col] / S[i]
            JBF_temp[col, row] = value
            JBF_vals[(row - 1) * k + col] = value
            rhs1[row, col] = -JPF_temp[row, col]
        end
        # In-place linear solving with pre-allocated pivot vector
        _, ipiv, info = LinearAlgebra.LAPACK.getrf!(JsuF_temp, GC.ipiv)
        if info != 0
            fill!(u, NaN)
            fill!(U, NaN)
            return nothing
        end
        @inbounds for row = 1:N, col = 1:N
            JsuF_lu[i, row, col] = JsuF_temp[row, col]
        end
        @inbounds for jj = 1:N
            JsuF_ipiv[i, jj] = ipiv[jj]
        end
        LinearAlgebra.LAPACK.getrs!('N', JsuF_temp, ipiv, view(rhs1, :, 1:k))
        @inbounds for col = 1:k, row = 1:N
            rhs1[row, k + col] = rhs1[row, col] / S[i]
        end

        _copy_rhs1_blocks!(SP, SB, UP, UB, rhs1, i)
        @inbounds @simd for jj = 1:k
            u[jj] -= SB[i, jj]
        end

        # Finish this fibre while its LU factors and input are still in the
        # workspaces. The factors and sensitivities above remain cached.
        fill!(adjoint_rhs, zero(ComplexF64))
        adjoint_rhs[1] = one(ComplexF64)
        LinearAlgebra.LAPACK.getrs!('T', JsuF_temp, ipiv, adjoint_rhs)

        _fill_contracted_hessian_input!(
            contracted_hessian_input, v0, adjoint_rhs, SP, UP, SB, UB, i,
        )
        evaluate!(contracted_hessian_vals, contracted_hessian_system, contracted_hessian_input)
        @inbounds for b = 1:k, a = 1:k
            U[a, b] += contracted_hessian_vals[(b - 1) * k + a]
        end
    end

    if !isnothing(p)
        @inbounds for ii = 1:length(u)
            u[ii] -= p[ii]
        end
    end

    nothing
end


function gradient_and_hessian(h::ProjectedHypersurface{TC}, x, p = nothing) where {TC}

    k = nvariables(h)
    u = zeros(ComplexF64, k)
    U = zeros(ComplexF64, k, k)
    gradient_and_hessian!(u, U, h, x, p)
    u, U
end

hessian(h::ProjectedHypersurface{TC}, x, p = nothing) where {TC} = gradient_and_hessian(h, x, p)[2]

"""
    fiber_tracking_stats(h::ProjectedHypersurface)

Return cumulative fibre-tracking diagnostics. Work performed by parallel
monodromy workers is aggregated into the original hypersurface after each solve.
"""
fiber_tracking_stats(h::ProjectedHypersurface) = fiber_tracking_stats(h.GC)
reset_fiber_cache!(h::ProjectedHypersurface) = reset_fiber_cache!(h.GC)
set_warm_fiber_tracking!(h::ProjectedHypersurface, enabled::Bool) =
    set_warm_fiber_tracking!(h.GC, enabled)



# Helpers for the fused derivative systems in GradientCache. They unpack one flat evaluation
# buffer into the matrix and tensor layouts used by the local linear algebra.
@inline function _fill_v0!(v0, S, Uvals, x, idx)
    v0[1] = S[idx]
    @inbounds for ii = 1:size(Uvals, 1)
        v0[1 + ii] = Uvals[ii, idx]
    end
    @inbounds for ii = 1:length(x)
        v0[1 + size(Uvals, 1) + ii] = x[ii]
    end
    v0
end

@inline function _fill_adjoint_gradient_input!(dest, v0, λ)
    offset = 0
    @inbounds for i in eachindex(v0)
        dest[offset + i] = v0[i]
    end
    offset += length(v0)
    @inbounds for i in eachindex(λ)
        dest[offset + i] = λ[i]
    end
    dest
end

@inline function _fill_contracted_hessian_input!(dest, v0, λ, SP, UP, SB, UB, idx)
    offset = 0
    @inbounds for i in eachindex(v0)
        dest[offset + i] = v0[i]
    end
    offset += length(v0)
    @inbounds for i in eachindex(λ)
        dest[offset + i] = λ[i]
    end
    offset += length(λ)

    # vec(x_p), column-major, with x=(s,u).
    k = size(SP, 2)
    N = size(UP, 2) + 1
    @inbounds for a = 1:k
        dest[offset + (a - 1) * N + 1] = SP[idx, a]
        for row = 2:N
            dest[offset + (a - 1) * N + row] = UP[idx, row - 1, a]
        end
    end
    offset += N * k

    # vec(x_β), column-major.
    @inbounds for b = 1:k
        dest[offset + (b - 1) * N + 1] = SB[idx, b]
        for row = 2:N
            dest[offset + (b - 1) * N + row] = UB[idx, row - 1, b]
        end
    end
    dest
end

@inline function _unpack_fused_columns!(dest, vals, nrows, ncols)
    for col = 1:ncols
        offset = (col - 1) * nrows
        @inbounds for row = 1:nrows
            dest[row, col] = vals[offset + row]
        end
    end
    dest
end

@inline function _evaluate_fused_columns!(dest, vals, F, x, nrows, ncols)
    evaluate!(vals, F, x)
    _unpack_fused_columns!(dest, vals, nrows, ncols)
end

@inline function _fill_rhs1!(rhs1, JPF_temp, JBF_temp)
    for col = 1:size(JPF_temp, 2)
        @inbounds for row = 1:size(rhs1, 1)
            rhs1[row, col] = JPF_temp[row, col]
        end
    end
    for idx = 1:size(JBF_temp, 1)
        col = size(JPF_temp, 2) + idx
        @inbounds for row = 1:size(rhs1, 1)
            rhs1[row, col] = JBF_temp[idx, row]
        end
    end
    rhs1
end

@inline function _copy_rhs1_blocks!(SP, SB, UP, UB, rhs1, idx)
    @inbounds @simd for jj = 1:size(SP, 2)
        SP[idx, jj] = rhs1[1, jj]
        SB[idx, jj] = rhs1[1, size(SP, 2) + jj]
    end
    @inbounds for ii = 1:size(rhs1, 1) - 1
        for jj = 1:size(SP, 2)
            UP[idx, ii, jj] = rhs1[1 + ii, jj]
            UB[idx, ii, jj] = rhs1[1 + ii, size(SP, 2) + jj]
        end
    end
    nothing
end
