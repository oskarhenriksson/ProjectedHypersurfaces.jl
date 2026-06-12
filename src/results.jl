export RoutingPointsResult,
    PartitionResult,
    routing_points,
    trace_result,
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
real routing points, [`trace_result`](@ref) for the final trace result, and
[`monodromy_result`](@ref) for the underlying monodromy computation.
The pair `solutions(result), parameters(result)` represents the routing points as
solutions to `∇r = 0`; use `monodromy_result(result)` for the monodromy start pair.
"""
struct RoutingPointsResult{P,T,M,Q}
    routing_points::P
    trace_result::T
    monodromy_result::M
    target_parameters::Q
end
function RoutingPointsResult(routing_points, trace_result, monodromy_result)
    if isempty(routing_points)
        nparameters = isnothing(monodromy_result) ? 0 : length(parameters(monodromy_result))
    else
        nparameters = length(first(routing_points))
    end
    RoutingPointsResult(
        routing_points,
        trace_result,
        monodromy_result,
        zeros(ComplexF64, nparameters),
    )
end

"""
    routing_points(result::RoutingPointsResult)

Return the real critical points used for routing.
"""
routing_points(R::RoutingPointsResult) = R.routing_points

"""
    trace_result(result::RoutingPointsResult)

Return the final HomotopyContinuation result obtained by tracing to ∇r = 0.
"""
trace_result(R::RoutingPointsResult) = R.trace_result

"""
    monodromy_result(result::RoutingPointsResult)

Return the underlying monodromy computation result.
"""
monodromy_result(R::RoutingPointsResult) = R.monodromy_result

solutions(R::RoutingPointsResult) = routing_points(R)
real_solutions(R::RoutingPointsResult; kwargs...) = routing_points(R)
nsolutions(R::RoutingPointsResult) = length(routing_points(R))
results(R::RoutingPointsResult; kwargs...) = results(trace_result(R); kwargs...)
nresults(R::RoutingPointsResult; kwargs...) = nresults(trace_result(R); kwargs...)
parameters(R::RoutingPointsResult) = R.target_parameters

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
    npts = nsolutions(R)
    header = "RoutingPointsResult with $npts $(_plural("routing point", npts))"
    println(io, header)
    println(io, "="^length(header))
    if isnothing(trace_result(R))
        println(io, "• raw trace results → unknown")
    else
        ntrace = nresults(R)
        println(io, "• $ntrace raw trace $(_plural("result", ntrace))")
    end
    if isnothing(monodromy_result(R))
        print(io, "• monodromy solutions → unknown")
    else
        nmon = nsolutions(monodromy_result(R))
        print(io, "• $nmon monodromy $(_plural("solution", nmon))")
    end
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
