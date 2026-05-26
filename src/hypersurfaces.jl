export ProjectedHypersurface, evaluate, gradient, hessian, degree

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

trace_test(h::ProjectedHypersurface) = trace_test(h.PWS)

Base.show(io::IO, h::ProjectedHypersurface) = println(io, "Projected hypersurface of degree $(degree(h)) in ambient dimension $(nvariables(h))")
    
ModelKit.variables(h::ProjectedHypersurface{TC}) where {TC} = h.projection_vars
ModelKit.nvariables(h::ProjectedHypersurface{TC}) where {TC} = length(h.projection_vars)

function Base.contains(
    h::ProjectedHypersurface,
    p::AbstractVector;
    atol = sqrt(eps(Float64)),
    residual_atol = atol,
)
    length(p) == nvariables(h) || throw(ArgumentError("Expected $(nvariables(h)) coordinates."))

    PWS, GC = h.PWS, h.GC
    track!(GC, PWS, p)

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

    track!(GC, PWS, x)

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
    JPF = GC.JPF
    JBF = GC.JBF

    v0 = GC.v0
    S = GC.S
    Uvals = GC.Uvals
    SB = GC.SB
    rhs1, rhs2, rhs3 = GC.rhs1, GC.rhs2, GC.rhs3
    JsuF_vals, JPF_vals, JBF_vals = GC.JsuF_vals, GC.JPF_vals, GC.JBF_vals

    N, n = size(PWS.F)
    k = n_projection_variables(PWS)

    u .= zero(eltype(u))

    # `track!` restores or computes both the tracked intersections and the cached S/Uvals data.
    track!(GC, PWS, x)

    #Obtain gradients of S and U with respect to p and β
    for i = 1:length(S)

        if !PWS.track_report[i] # skip if i-th track failed
            continue
        end

        _fill_v0!(v0, S, Uvals, x, i)

        # Evaluate the fused derivative blocks and unpack them into the working arrays.
        JsuF_temp = GC.JsuF_temp
        _evaluate_fused_columns!(JsuF_temp, JsuF_vals, JsuF, v0, N, N)

        JPF_temp = GC.JPF_temp
        _evaluate_fused_columns!(JPF_temp, JPF_vals, JPF, v0, N, k)

        JBF_temp = GC.JBF_temp
        _evaluate_fused_columns!(JBF_temp, JBF_vals, JBF, v0, k, N)

        _fill_rhs1!(rhs1, JPF_temp, JBF_temp)

        rhs1 .*= -1
        # In-place linear solving with pre-allocated pivot vector
        _, ipiv, info = LinearAlgebra.LAPACK.getrf!(JsuF_temp, GC.ipiv)
        if info == 0 # this indicates successful factorization
            LinearAlgebra.LAPACK.getrs!('N', JsuF_temp, ipiv, rhs1)
        else
            fill!(rhs1, zero(ComplexF64))
        end

        # copy rhs1 row segment into SB row without creating slices
        @inbounds @simd for jj = 1:k
            SB[i, jj] = rhs1[1, k + jj]
        end
        @inbounds @simd for jj = 1:k
            u[jj] -= SB[i, jj]
        end
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
    JBF = GC.JBF
    HF = GC.HF
    JxB = GC.JxB
    JxP = GC.JxP
    JPB = GC.JPB

    # Preallocated temporaries and cached LU data keep the Hessian path allocation-free.
    JsuF_lu = GC.JsuF_lu
    JsuF_ipiv = GC.JsuF_ipiv
    JsuF_lu_success = GC.JsuF_lu_success
    temp_Hi = GC.temp_Hi
    temp_Jxpi = GC.temp_Jxpi
    temp_Jxbi = GC.temp_Jxbi
    temp_Jpbi = GC.temp_Jpbi

    v0 = GC.v0
    S = GC.S
    Uvals = GC.Uvals
    SP = GC.SP
    SB = GC.SB
    UP = GC.UP
    UB = GC.UB
    A = GC.A
    rhs1, rhs2, rhs3 = GC.rhs1, GC.rhs2, GC.rhs3
    JsuF_vals, JPF_vals, JBF_vals = GC.JsuF_vals, GC.JPF_vals, GC.JBF_vals
    HF_vals, JxB_vals, JxP_vals, JPB_vals = GC.HF_vals, GC.JxB_vals, GC.JxP_vals, GC.JPB_vals

    M, M1, M2, M3 = GC.M, GC.M1, GC.M2, GC.M3

    k = n_projection_variables(PWS)
    N, n = size(PWS.F)

    u .= zero(eltype(u))
    U .= zero(eltype(U))

    # `track!` restores or computes both the tracked intersections and the cached S/Uvals data.
    track!(GC, PWS, x)

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

        JBF_temp = GC.JBF_temp
        _evaluate_fused_columns!(JBF_temp, JBF_vals, JBF, v0, k, N)

        _fill_rhs1!(rhs1, JPF_temp, JBF_temp)

        rhs1 .*= -1
        # In-place linear solving with pre-allocated pivot vector
        _, ipiv, info = LinearAlgebra.LAPACK.getrf!(JsuF_temp, GC.ipiv)
        JsuF_lu_success[i] = (info == 0)
        @inbounds for row = 1:N, col = 1:N
            JsuF_lu[i, row, col] = JsuF_temp[row, col]
        end
        @inbounds for jj = 1:N
            JsuF_ipiv[i, jj] = ipiv[jj]
        end
        if info == 0  # this indicates successful factorization
            LinearAlgebra.LAPACK.getrs!('N', JsuF_temp, ipiv, rhs1)
        else
            fill!(rhs1, zero(ComplexF64))
        end

        _copy_rhs1_blocks!(SP, SB, UP, UB, rhs1, i)
        @inbounds @simd for jj = 1:k
            u[jj] -= SB[i, jj]
        end

    end

    if !isnothing(p)
        @inbounds for ii = 1:length(u)
            u[ii] -= p[ii]
        end
    end

    # Compute the second-derivative contributions using the fused tensor systems.
    for j = 1:length(S)

        !PWS.track_report[j] && continue # skip if j-th track failed

        _fill_v0!(v0, S, Uvals, x, j)

        HF_temp = GC.HF_temp
        HF_nrows, HF_ncols = N, N
        _evaluate_fused_tensor!(HF_temp, HF_vals, HF, v0, N, HF_nrows, HF_ncols)

        JxB_temp = GC.JxB_temp
        JxB_nrows, JxB_ncols = N, k
        _evaluate_fused_tensor!(JxB_temp, JxB_vals, JxB, v0, N, JxB_nrows, JxB_ncols)

        JxP_temp = GC.JxP_temp
        JxP_nrows, JxP_ncols = N, k
        _evaluate_fused_tensor!(JxP_temp, JxP_vals, JxP, v0, N, JxP_nrows, JxP_ncols)

        JPB_temp = GC.JPB_temp
        JPB_nrows, JPB_ncols = k, k
        _evaluate_fused_tensor!(JPB_temp, JPB_vals, JPB, v0, N, JPB_nrows, JPB_ncols)

        _fill_M1_M2!(M1, M2, SP, SB, UP, UB, j)

        for i = 1:N

            # copy slices into temporaries (avoids allocating SubArray objects)
            @inbounds for r = 1:HF_nrows, c = 1:HF_ncols
                temp_Hi[r, c] = HF_temp[i, r, c]
            end
            @inbounds for r = 1:JxP_nrows, c = 1:JxP_ncols
                temp_Jxpi[r, c] = JxP_temp[i, r, c]
            end
            @inbounds for r = 1:JxB_nrows, c = 1:JxB_ncols
                temp_Jxbi[r, c] = JxB_temp[i, r, c]
            end
            @inbounds for r = 1:JPB_nrows, c = 1:JPB_ncols
                temp_Jpbi[r, c] = JPB_temp[i, r, c]
            end

            # now step by step in-place matrix multiplications. 
            for a = 1:k, b = 1:k
                A[j, i, a, b] = temp_Jpbi[b, a] # note the transpose here
            end
            mul!(M, transpose(temp_Jxpi), M2)
            for a = 1:k, b = 1:k
                A[j, i, a, b] += M[b, a] # note the transpose here
            end
            mul!(M, M1, temp_Jxbi)
            for a = 1:k, b = 1:k
                A[j, i, a, b] += M[b, a] # note the transpose here
            end
            mul!(M3, M1, temp_Hi)
            mul!(M, M3, M2)
            for a = 1:k, b = 1:k
                A[j, i, a, b] += M[b, a] # note the transpose here
            end

        end
    end


    # Reuse the LU factors of JsuF computed above when solving the final Hessian systems.
    fill!(M, zero(ComplexF64)) # here M will get assigned the Hessian of log r
    for j = 1:length(S)
        
        !PWS.track_report[j] && continue # skip if j-th track failed
        !JsuF_lu_success[j] && continue


        Jtu = GC.Jtu_temp
        @inbounds for row = 1:N, col = 1:N
            Jtu[row, col] = JsuF_lu[j, row, col]
        end
        @inbounds for jj = 1:N
            GC.ipiv[jj] = JsuF_ipiv[j, jj]
        end
        for a = 1:k, b = 1:k
            for i = 1:N
                rhs2[i] = A[j, i, a, b]
            end
            LinearAlgebra.LAPACK.getrs!('N', Jtu, GC.ipiv, rhs2)
            M[a, b] += rhs2[1]
        end
    end

    for a = 1:k, b = 1:k
        U[a, b] += M[a, b]
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

@inline function _unpack_fused_columns!(dest, vals, nrows, ncols)
    for col = 1:ncols
        offset = (col - 1) * nrows
        @inbounds for row = 1:nrows
            dest[row, col] = vals[offset + row]
        end
    end
    dest
end

@inline function _unpack_fused_tensor!(dest, vals, nout, nrows, ncols)
    for col = 1:ncols
        for row = 1:nrows
            offset = ((col - 1) * nrows + (row - 1)) * nout
            @inbounds for out = 1:nout
                dest[out, row, col] = vals[offset + out]
            end
        end
    end
    dest
end

@inline function _evaluate_fused_columns!(dest, vals, F, x, nrows, ncols)
    evaluate!(vals, F, x)
    _unpack_fused_columns!(dest, vals, nrows, ncols)
end

@inline function _evaluate_fused_tensor!(dest, vals, F, x, nout, nrows, ncols)
    evaluate!(vals, F, x)
    _unpack_fused_tensor!(dest, vals, nout, nrows, ncols)
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

@inline function _fill_M1_M2!(M1, M2, SP, SB, UP, UB, idx)
    k = size(SP, 2)
    N = size(M2, 1)
    for a = 1:k
        M1[a, 1] = SP[idx, a]
    end
    for a = 1:k
        for b = 2:N
            M1[a, b] = UP[idx, b - 1, a]
        end
    end
    for b = 1:k
        M2[1, b] = SB[idx, b]
    end
    for b = 1:k
        for a = 2:N
            M2[a, b] = UB[idx, a - 1, b]
        end
    end
    nothing
end
