export membership

### Adapted from the membership test in
### https://github.com/JuliaAlgebra/HypersurfaceRegions.jl/blob/main/src/membership.jl

@doc raw"""
    membership(regions::Vector{Region}, p; reltol = 1e-6, abstol = 1e-9)

Determine which [`Region`](@ref) a point `p` in the parameter space belongs to.

The point `p` is flowed by gradient ascent of the routing function until it
reaches a critical point; the region whose critical points contain that limit is
returned. All regions in `regions` are assumed to come from the same partition,
i.e. to share a common routing function (the one stored on the first region).

Returns `nothing` if `regions` is empty or if the gradient flow does not converge
to one of the known critical points.

Options:
* `reltol = 1e-6`, `abstol = 1e-9`: parameters for the accuracy of the ODE solver.
"""
function membership(
    regions::AbstractVector{Region},
    p::AbstractVector{<:Real};
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-9,
)
    isempty(regions) && return nothing

    r = first(regions).r
    ∇r = RoutingGradient(r)
    ode_log! = set_up_ode(∇r)

    # flatten the critical points of all regions into a single list
    all_critical_points = reduce(vcat, critical_points(C) for C in regions)

    critical_point_index, _ =
        limit_critical_point(ode_log!, Float64.(p), all_critical_points, reltol, abstol)

    if critical_point_index == -1
        return nothing
    end

    end_critical_point = all_critical_points[critical_point_index]
    for C in regions
        if end_critical_point in critical_points(C)
            return C
        end
    end
    return nothing
end
