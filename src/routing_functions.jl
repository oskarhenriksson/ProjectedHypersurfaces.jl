export RoutingFunction, evaluate, evaluate_and_jacobian, gradient, hessian, denominator_exponent, RoutingGradient

struct RoutingFunction{TQ,TP,TC} <: HC.AbstractSystem
    H::Vector{ProjectedHypersurface{TC}} 
    projection_vars::Vector{HC.Variable}
    e::Int
    c::Vector
    G::Union{MixedSystem, Nothing}
    ∇logprodg::Union{TP, Nothing}
    q::Expression
    ∇logqe::Union{TQ, Nothing}
end
function RoutingFunction(
    H::Vector{<:ProjectedHypersurface};
    e::Union{Int,Nothing} = nothing,
    c::Union{Vector,Nothing} = nothing,
    g::Union{Vector{Expression},Vector{Variable},Nothing} = nothing,
)

    projection_vars = H[1].projection_vars

    for h in H[2:end]
        @assert h.projection_vars == projection_vars "All hypersurfaces must have the same projection variables"
    end

    k = length(projection_vars)

    if isnothing(g) || length(g) == 0
        ∇logprodg = nothing
        g_degree = 0
        G = nothing
    else
        g = Expression.(g)
        @assert ModelKit.variables(g) ⊆ projection_vars "Variables in g must match projection_vars"
        G = System(g, variables=projection_vars) |> fixed
        ∇logprodg = System(sum([differentiate(log(gi), projection_vars) for gi in g]), variables=projection_vars) |> fixed
        g_degree = sum(HC.degree.(g))
    end

    deg = sum([degree(h) for h in H])
    if isnothing(e)
        e = div(deg + g_degree, 2) + 1
    end

    if isnothing(c)
        c = randn(k)
    end

    q = 1 + sum((projection_vars - c) .* (projection_vars - c))
    ∇logqe = System(differentiate(-e * log(q), projection_vars), variables = projection_vars) |> fixed

    RoutingFunction{typeof(∇logqe), typeof(∇logprodg), typeof(H[1].GC)}(H, projection_vars, e, c, G, ∇logprodg, q, ∇logqe)
end
RoutingFunction(h::ProjectedHypersurface; kwargs...) = RoutingFunction([h]; kwargs...)

denominator_exponent(r::RoutingFunction) = r.e
ModelKit.variables(r::RoutingFunction) = r.projection_vars
ModelKit.nvariables(r::RoutingFunction) = length(r.projection_vars)

function Base.show(io::IO, r::RoutingFunction)
    header = "Routing function for projected hypersurface"
    println(io, header) 
    println(io, "="^(length(header)))
    println(io, " Variables: ", join(r.projection_vars, ", "))
    if length(r.H) == 1
        println(io, " Numerator: ", r.H[1])
    else
        println(io, " Summands in numerator: projected hypersurfaces of degrees [", join(degree.(r.H), ", "), "] in ambient dimension ", nvariables(r.H[1]))
    end
    if !isnothing(r.G)
        println(io, " Additional summands in numerator: ", join(r.G.compiled.system.expressions, ", "))
    end
    println(io, " Denominator: ", (r.q)^r.e)
end


function ModelKit.evaluate(r::RoutingFunction, x, p = nothing)
    H = r.H
    e, c = r.e, r.c
    G = r.G

    if !isnothing(G)
        u = sum(log(abs(gi)) for gi in G(x)) - e * log(1 + sum((x - c) .* (x - c)))
    else
        u = - e * log(1 + sum((x - c) .* (x - c)))
    end

    for h in H
        u += h(x)
    end

    u
end

(r::RoutingFunction{TQ,TP,TC})(x) where {TQ,TP,TC} = evaluate(r, x)


function gradient!(u, r::RoutingFunction{TQ,TP,TC}, x, p = nothing) where {TQ,TP,TC}
    
    H, ∇logqe, ∇logprodg = r.H, r.∇logqe, r.∇logprodg

    gradient_temp = H[1].GC.gradient_temp

    # Denominator
    evaluate!(u, ∇logqe, x)

    # Known numerator
    if !isnothing(∇logprodg)
        evaluate!(gradient_temp, ∇logprodg, x)
        u .+= gradient_temp
    end

    # Projected hypersurface
    for h in H
        gradient!(gradient_temp, h, x)
        u .+= gradient_temp
    end

    if !isnothing(p)
        @inbounds for ii = 1:length(u)
            u[ii] -= p[ii]
        end
    end

    nothing
end

function gradient(r::RoutingFunction{TQ,TP,TC}, x, p = nothing) where {TQ,TP,TC}
    k = nvariables(r)
    u = zeros(ComplexF64, k)
    gradient!(u, r, x, p)
    u
end


function gradient_and_hessian!(u, U, r::RoutingFunction{TQ,TP,TC}, x, p = nothing) where {TQ,TP,TC}

    H, ∇logqe, ∇logprodg = r.H, r.∇logqe, r.∇logprodg
    GC = H[1].GC
    gradient_temp = GC.gradient_temp
    Hess_temp = GC.Hess_temp


    # Denominator
    evaluate_and_jacobian!(u, U, ∇logqe, x)

    # Known numberator
    if !isnothing(∇logprodg)
        evaluate_and_jacobian!(gradient_temp, Hess_temp, ∇logprodg, x)
        u .+= gradient_temp
        U .+= Hess_temp
    end

    # Projected hypersurface
    for h in H
        gradient_and_hessian!(gradient_temp, Hess_temp, h, x)
        u .+= gradient_temp
        U .+= Hess_temp
    end

    if !isnothing(p)
        @inbounds for ii = 1:length(u)
            u[ii] -= p[ii]
        end
    end

    nothing
end
function gradient_and_hessian(r::RoutingFunction{TQ,TP,TC}, x, p = nothing) where {TQ,TP,TC}
    k = nvariables(r)
    u = zeros(ComplexF64, k)
    U = zeros(ComplexF64, k, k)
    gradient_and_hessian!(u, U, r, x, p)
    u, U
end
hessian(r::RoutingFunction{TQ,TP,TC}, x, p = nothing) where {TQ,TP,TC} =
    gradient_and_hessian(r, x, p)[2]

#########

struct RoutingGradient <: HC.AbstractSystem
    r::RoutingFunction
end

Base.show(io::IO, ∇r::RoutingGradient) = print(io, "Routing gradient")

import Base.size
function Base.size(∇r::RoutingGradient)
    k = nvariables(∇r.r)
    (k, k)
end
ModelKit.variables(∇r::RoutingGradient) = ∇r.r.projection_vars
denominator_exponent(∇r::RoutingGradient) = ∇r.r.e

evaluate(∇r::RoutingGradient, x, p = nothing) = gradient(∇r.r, x, p)
evaluate!(u, ∇r::RoutingGradient, x, p = nothing) = gradient!(u, ∇r.r, x, p)
evaluate_and_jacobian(∇r::RoutingGradient, x, p = nothing) = gradient_and_hessian(∇r.r, x, p)
evaluate_and_jacobian!(u, U, ∇r::RoutingGradient, x, p = nothing) = gradient_and_hessian!(u, U, ∇r.r, x, p)

function taylor!(u, ::Val, F::RoutingGradient, x, p)
    fill!(u, zero(ComplexF64))
end
