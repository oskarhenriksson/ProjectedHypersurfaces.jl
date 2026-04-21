export RestrictionToLineSystem

# `AbstractSystem` for the substitution x = [p + t * direction; w].
# The base point p is treated as a parameter, while the line direction is fixed.
"""
    RestrictionToLineSystem(F::AbstractSystem, direction, k)

Construct the system obtained from `F(x)` under the substitution
`x = [p + t * direction; w]`, where the first `k` coordinates of `x` are replaced
by a point on the affine line with base point `p` and fixed direction `direction`.

The new variables are `[t; w]` and the new parameters are the coordinates of `p`.
"""
struct RestrictionToLineSystem{S<:AbstractSystem} <: AbstractSystem
    system::S
    direction::Vector{ComplexF64}
    k::Int
    t::Variable
    parameters::Vector{Variable}

    v::Vector{ComplexF64}
    v_high::Vector{HC.ComplexDF64}
    u::Vector{ComplexF64}
    ū::Vector{HC.ComplexDF64}
    J::Matrix{ComplexF64}

    tv⁴::TaylorVector{5,ComplexF64}
    tv³::TaylorVector{4,ComplexF64}
    tv²::TaylorVector{3,ComplexF64}
    tv¹::TaylorVector{2,ComplexF64}
end

function RestrictionToLineSystem(
    F::AbstractSystem,
    direction::AbstractVector{T},
    k::Int,
) where {T<:Number}
    isempty(parameters(F)) || throw(
        ArgumentError(
            "RestrictionToLineSystem currently expects a system without parameters.",
        ),
    )

    N = size(F, 2)
    0 <= k <= N || throw(ArgumentError("Expected 0 <= k <= number of variables of F."))
    length(direction) == k || throw(
        ArgumentError(
            "Dimension mismatch between the number of projection variables and the length of the direction vector.",
        ),
    )

    @unique_var t, p[1:k]

    n = size(F, 1)
    direction̂ = ComplexF64.(direction)

    RestrictionToLineSystem(
        F,
        direction̂,
        k,
        t,
        p,
        zeros(ComplexF64, N),
        zeros(HC.ComplexDF64, N),
        zeros(ComplexF64, n),
        zeros(HC.ComplexDF64, n),
        zeros(ComplexF64, size(F)),
        TaylorVector{5}(ComplexF64, N),
        TaylorVector{4}(ComplexF64, N),
        TaylorVector{3}(ComplexF64, N),
        TaylorVector{2}(ComplexF64, N),
    )
end

RestrictionToLineSystem(
    F::System,
    direction::AbstractVector{T},
    k::Int;
    compile::Union{Bool,Symbol} = HC.COMPILE_DEFAULT[],
) where {T<:Number} = RestrictionToLineSystem(fixed(F; compile = compile), direction, k)

function Base.show(io::IO, mime::MIME"text/plain", F::RestrictionToLineSystem)
    println(io, typeof(F), ":")
    println(io, "direction:")
    show(io, mime, F.direction)
    println(io, "\n\nF:")
    show(io, mime, F.system)
end

Base.size(F::RestrictionToLineSystem) = (size(F.system, 1), size(F.system, 2) - F.k + 1)
ModelKit.variables(F::RestrictionToLineSystem) = [F.t; variables(F.system)[(F.k+1):end]]
ModelKit.parameters(F::RestrictionToLineSystem) = F.parameters
ModelKit.variable_groups(F::RestrictionToLineSystem) = nothing

function (F::RestrictionToLineSystem)(x, p)
    length(x) == size(F, 2) || throw(ArgumentError("Expected $(size(F, 2)) variables."))
    length(p) == F.k || throw(ArgumentError("Expected $(F.k) parameters."))
    t = x[1]
    w = x[2:end]
    v = p .+ t .* F.direction
    F.system([v; w])
end

# Cache the ambient point [p + t * direction; w] before delegating to the compiled system.
@inline function _set_restricted_vector!(
    v::AbstractVector,
    F::RestrictionToLineSystem,
    x::AbstractVector,
    p::AbstractVector,
)
    length(x) == size(F, 2) || throw(ArgumentError("Expected $(size(F, 2)) variables."))
    length(p) == F.k || throw(ArgumentError("Expected $(F.k) parameters."))
    t = x[1]
    @inbounds for i = 1:F.k
        v[i] = p[i] + t * F.direction[i]
    end
    @inbounds for i = (F.k + 1):length(v)
        v[i] = x[i - F.k + 1]
    end
    v
end

function ModelKit.evaluate!(u, F::RestrictionToLineSystem, x::AbstractVector{HC.ComplexDF64}, p)
    _set_restricted_vector!(F.v_high, F, x, p)
    evaluate!(u, F.system, F.v_high)
    u
end

function ModelKit.evaluate!(u, F::RestrictionToLineSystem, x, p)
    _set_restricted_vector!(F.v, F, x, p)
    evaluate!(u, F.system, F.v)
    u
end

function ModelKit.evaluate_and_jacobian!(u, U, F::RestrictionToLineSystem, x, p)
    _set_restricted_vector!(F.v, F, x, p)
    evaluate_and_jacobian!(u, F.J, F.system, F.v)

    # Apply the chain rule for the affine restriction map [t; w] -> [p + t * direction; w].
    n = size(F.system, 1)
    wdim = size(F.system, 2) - F.k
    @inbounds for i = 1:n
        Ui1 = zero(eltype(U))
        for j = 1:F.k
            Ui1 += F.J[i, j] * F.direction[j]
        end
        U[i, 1] = Ui1
    end
    @inbounds for j = 1:wdim, i = 1:n
        U[i, j + 1] = F.J[i, F.k + j]
    end

    nothing
end

function ModelKit.taylor!(
    u::AbstractVector,
    ::Val{1},
    F::RestrictionToLineSystem,
    x::AbstractVector,
    p::TaylorVector,
)
    length(x) == size(F, 2) || throw(ArgumentError("Expected $(size(F, 2)) variables."))
    length(p) == F.k || throw(ArgumentError("Expected $(F.k) parameters."))

    t = x[1]
    p0, p1 = vectors(p)
    @inbounds for j = 1:F.k
        F.v[j] = p0[j] + t * F.direction[j]
    end
    @inbounds for j = (F.k + 1):length(F.v)
        F.v[j] = x[j - F.k + 1]
    end

    evaluate_and_jacobian!(F.u, F.J, F.system, F.v)

    @inbounds for i = 1:size(F.system, 1)
        ui = zero(eltype(u))
        for j = 1:F.k
            ui += F.J[i, j] * p1[j]
        end
        u[i] = ui
    end

    u
end

_taylor_tv(F::RestrictionToLineSystem, ::Val{1}) = F.tv¹
_taylor_tv(F::RestrictionToLineSystem, ::Val{2}) = F.tv²
_taylor_tv(F::RestrictionToLineSystem, ::Val{3}) = F.tv³
_taylor_tv(F::RestrictionToLineSystem, ::Val{4}) = F.tv⁴

@inline function _parameter_coeff(p, i, j)
    j <= length(p[i]) ? p[i][j - 1] : zero(eltype(p[i]))
end
@inline function _parameter_coeff(p::TaylorVector, i, j)
    pi = p[i]
    j <= length(pi) ? pi[j - 1] : zero(eltype(pi))
end
@inline _parameter_coeff(p::AbstractVector, i, j) = j == 1 ? p[i] : zero(eltype(p))

function _set_restricted_taylor!(
    tv_out::TaylorVector{M,ComplexF64},
    F::RestrictionToLineSystem,
    tx::TaylorVector,
    p,
) where {M}
    length(tx) == size(F, 2) || throw(ArgumentError("Expected $(size(F, 2)) variables."))
    length(p) == F.k || throw(ArgumentError("Expected $(F.k) parameters."))
    z = vectors(tv_out)
    x = vectors(tx)

    # Assemble Taylor coefficients of the restricted ambient point coefficient-wise.
    for j = 1:M
        zj = z[j]
        @inbounds for i = 1:F.k
            zj[i] = _parameter_coeff(p, i, j)
        end
        if j <= length(x)
            xj = x[j]
            tj = xj[1]
            @inbounds for i = 1:F.k
                zj[i] += tj * F.direction[i]
            end
            @inbounds for i = (F.k + 1):length(zj)
                zj[i] = xj[i - F.k + 1]
            end
        else
            @inbounds for i = (F.k + 1):length(zj)
                zj[i] = zero(eltype(zj))
            end
        end
    end

    tv_out
end

function ModelKit.taylor!(
    u::AbstractVector,
    v::Val{N},
    F::RestrictionToLineSystem,
    tx,
    p,
) where {N}
    tv = _taylor_tv(F, Val(N))
    _set_restricted_taylor!(tv, F, tx, p)
    taylor!(u, v, F.system, tv)
end

function ModelKit.taylor!(
    u::TaylorVector,
    v::Val{N},
    F::RestrictionToLineSystem,
    tx,
    p,
) where {N}
    tv = _taylor_tv(F, Val(N))
    _set_restricted_taylor!(tv, F, tx, p)
    taylor!(u, v, F.system, tv)
end
