export RoutingPointsResult,
    PartitionResult,
    routing_points,
    result,
    monodromy_result,
    return_code,
    partitions,
    morse_indices,
    failed_info,
    npartitions

import HomotopyContinuation:
    solutions, real_solutions, nsolutions, results, nresults

"""
    RoutingPointsResult

Result returned by [`critical_points`](@ref). Use [`routing_points`](@ref) for the
real routing points, [`result`](@ref) for the final trace result,
[`monodromy_result`](@ref) for the underlying monodromy computation, and
[`solutions(result)`](../homotopy_continuation.md#solutions) for the routing points as solutions to `∇r = 0`; use `monodromy_result(result)` for the monodromy start pair.
"""
struct RoutingPointsResult{P,T,M}
    routing_points::P
    result::T
    monodromy_result::M
end

"""
    routing_points(result::RoutingPointsResult)

Return the real critical points used for routing.
"""
routing_points(R::RoutingPointsResult) = R.routing_points

"""
    result(result::RoutingPointsResult)

Return the final HomotopyContinuation result obtained by tracing to ∇r = 0.
"""
result(R::RoutingPointsResult) = R.result

"""
    monodromy_result(result::RoutingPointsResult)

Return the underlying monodromy computation result.
"""
monodromy_result(R::RoutingPointsResult) = R.monodromy_result

"""
    PartitionResult

Result returned by [`partition_of_critical_points`](@ref). Use
[`partitions`](@ref) for connected components, [`morse_indices`](@ref) for the
critical point indices, and [`failed_info`](@ref) for failed connection attempts.
"""
struct PartitionResult{P,I,F}
    partitions::P
    morse_indices::I # TODO: this is a strange output to expose to users, since it is indexing yet another list you must reference. Considering changing this to a Dict or something.
    failed_info::F
    return_code::Symbol
end

"""
    partitions(result::PartitionResult)

Return the connected components as vectors of routing point indices.
"""
partitions(R::PartitionResult) = R.partitions

"""
    morse_indices(result::PartitionResult)

Return the Morse index computed for each routing point, or `nothing` if the
partition failed before indices were available.
"""
morse_indices(R::PartitionResult) = R.morse_indices

"""
    failed_info(result::PartitionResult)

Return information collected from failed connection attempts.
"""
failed_info(R::PartitionResult) = R.failed_info

"""
    return_code(result::PartitionResult)

Return a symbolic status code for a partition result.
"""
return_code(R::PartitionResult) = R.return_code

"""
    npartitions(result::PartitionResult)

Return the number of connected components in the partition result.
"""
npartitions(R::PartitionResult) = length(partitions(R))

function _plural(noun::AbstractString, n::Integer)
    n == 1 ? noun : string(noun, "s")
end

function Base.show(io::IO, R::RoutingPointsResult)
    npts = length(routing_points(R))
    println(io, "RoutingPointsResult with $npts $(_plural("routing point", npts))")
end

function Base.show(io::IO, R::PartitionResult)
    npars = npartitions(R)
    nfailures = length(failed_info(R))
    header = "PartitionResult with $npars $(_plural("partition", npars))"
    println(io, header)
    println(io, "="^length(header))
    println(io, "• return_code → :$(return_code(R))")
    println(io, "• $(nfailures) failed $(_plural("connection", nfailures))")
    print(io, "• morse_indices → ", isnothing(morse_indices(R)) ? "not computed" : length(morse_indices(R)))
end
