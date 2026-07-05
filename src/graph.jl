export partition_of_critical_points

export return_code,
    partitions,
    morse_indices,
    failed_info,
    nregions

### This is adapted from https://github.com/JuliaAlgebra/HypersurfaceRegions.jl/blob/main/src/partition.jl


# return index and unstable eigenvectors of the hessian 
function index_unstable_eigenvector!(u, U, ∇r, a)
    evaluate_and_jacobian!(u, U, ∇r, a)
    if is_almost_singular(U)
        flag = true
        println("The Hessian is almost singular for", a)
        return nothing, nothing, flag
    else
        flag = false
    end
    eigvals, eigvecs = LinearAlgebra.eigen!(real(U))
    eigvals = eigvals
    index = sum(eigvals .> 0)
    unstable_eigenvectors = eigvecs[:, eigvals.>0]
    return index, unstable_eigenvectors, flag
end

function is_almost_singular(matrix::Matrix; threshold = 1e10)
    condition_number = LinearAlgebra.cond(matrix)
    return condition_number > threshold
end

## calculate the index and unstable eigenvectors of the critical points
function _index_list(∇r, crit_pts)
    index_list::Vector{Int} = []
    unstable_vector_list::Vector{Matrix{Float64}} = []
    flag_prime = false
    k = size(∇r, 1)
    u = randn(ComplexF64, k)
    U = zeros(ComplexF64, k, k)

    for a in crit_pts
        index, unstable_eigenvectors, flag = index_unstable_eigenvector!(u, U, ∇r, a)
        if flag == true
            flag_prime = true
            return nothing, nothing, flag_prime
        end

        push!(index_list, index)
        push!(unstable_vector_list, unstable_eigenvectors)
    end

    return index_list, unstable_vector_list, flag_prime
end


# input a list, output a list of indices that have the same value in the list
function partition_indices(lst)
    partitions = Dict()
    for (index, element) in enumerate(lst)
        if haskey(partitions, element)
            push!(partitions[element], index)
        else
            partitions[element] = [index]
        end
    end
    return partitions
end


@doc raw"""
    partition_of_critical_points(
    r::RoutingFunction,
    crit_pts::AbstractVector{<:AbstractVector{<:Real}},
    epsilon::Float64 = 1e-6,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-9,
)

Partition a collection `crit_pts` of critical points of a routing function `r` into connected components via gradient flow.
The function returns a [`PartitionResult`](@ref) containing the connected components, Morse indices, and any failed connection attempts.  
"""
function partition_of_critical_points(
    r::RoutingFunction,
    crit_pts::AbstractVector{<:AbstractVector{<:Real}},
    epsilon::Float64 = 1e-6,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-9,
)

    ∇r = RoutingGradient(r)

    index_list, unstable_eigenvector_list, flag_prime = _index_list(∇r, crit_pts)
    if flag_prime == true
        @warn "The Hessian is almost singular for some critical points"
        return PartitionResult(Vector{Vector{Int}}(), nothing, [], :singular_hessian)
    end


    ode_log! = set_up_ode(∇r)

    graph = LightGraphs.SimpleGraph(length(crit_pts))
    connectivity_status = zeros(Int, length(crit_pts))
    critical_points_indices = collect(1:length(crit_pts))

    # count the number of index 0 critical points
    count_index_0 = sum([index == 0 for index in index_list])

    if count_index_0 == 1
        # we do not need to do any path tracking in this case
        return PartitionResult([critical_points_indices], index_list, [], :success)
    end

    failed_info_list = []
    ProgressMeter.@showprogress for i = 1:length(index_list)
        if connectivity_status[i] == 0 && index_list[i] == 1
            # need to do path tracking in two directions
            critical_point_index = critical_points_indices[i]
            unstable_eigenvector = unstable_eigenvector_list[critical_point_index]
            pair_pos, failed_info_pos = limit_critical_point_from_critical_point(
                ode_log!,
                crit_pts,
                i,
                index_list,
                unstable_eigenvector,
                epsilon,
                reltol,
                abstol,
            )
            if isempty(failed_info_pos)
                LightGraphs.add_edge!(graph, pair_pos[1], pair_pos[2])
            else
                push!(
                    failed_info_list,
                    [critical_point_index, epsilon, unstable_eigenvector, failed_info_pos],
                )
            end

            pair_neg, failed_info_neg = limit_critical_point_from_critical_point(
                ode_log!,
                crit_pts,
                i,
                index_list,
                unstable_eigenvector,
                -epsilon,
                reltol,
                abstol,
            )

            if isempty(failed_info_neg)
                LightGraphs.add_edge!(graph, pair_neg[1], pair_neg[2])
            else
                push!(
                    failed_info_list,
                    [critical_point_index, -epsilon, unstable_eigenvector, failed_info_neg],
                )
            end
            connectivity_status[i] = 1
        end
    end

    ProgressMeter.@showprogress for i = 1:length(index_list)
        if connectivity_status[i] == 0 && index_list[i] > 1
            # track paths and stop whenever one path converges
            critical_point_index = critical_points_indices[i]
            sub_failed_info_list = []
            for v in eachcol(unstable_eigenvector_list[critical_point_index])
                pair, failed_info = limit_critical_point_from_critical_point(
                    ode_log!,
                    crit_pts,
                    i,
                    index_list,
                    v,
                    epsilon,
                    reltol,
                    abstol,
                )
                if isempty(failed_info)
                    LightGraphs.add_edge!(graph, pair[1], pair[2])
                    connectivity_status[i] = 1
                    sub_failed_info_list = []
                    break
                else
                    push!(
                        sub_failed_info_list,
                        [critical_point_index, epsilon, v, failed_info],
                    )
                end

                pair, failed_info = limit_critical_point_from_critical_point(
                    ode_log!,
                    crit_pts,
                    i,
                    index_list,
                    v,
                    -epsilon,
                    reltol,
                    abstol,
                )
                if isempty(failed_info)
                    LightGraphs.add_edge!(graph, pair[1], pair[2])
                    connectivity_status[i] = 1
                    sub_failed_info_list = []
                    break
                else
                    push!(
                        sub_failed_info_list,
                        [critical_point_index, epsilon, v, failed_info],
                    )
                end
            end

            if !isempty(sub_failed_info_list)
                push!(failed_info_list, sub_failed_info_list)
            end
        end
    end


    partition = LightGraphs.connected_components(graph)
    partition_critical_point_indices = Vector{Int}[]
    for par in partition
        push!(partition_critical_point_indices, @view(critical_points_indices[par]))
    end
    code = isempty(failed_info_list) ? :success : :partial_success
    return PartitionResult(partition_critical_point_indices, index_list, failed_info_list, code)
end

partition_of_critical_points(
    r::RoutingFunction,
    result::RoutingPointsResult,
    args...;
    kwargs...,
) = partition_of_critical_points(r, routing_points(result), args...; kwargs...)


@doc raw"""
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
    regions(result::PartitionResult)

Return the connected components as vectors of routing point indices.
"""
regions(R::PartitionResult) = R.partitions

@doc raw"""
    morse_indices(result::PartitionResult)

Return the Morse index computed for each routing point, or `nothing` if the
partition failed before indices were available.
"""
morse_indices(R::PartitionResult) = R.morse_indices

@doc raw"""
    failed_info(result::PartitionResult)

Return information collected from failed connection attempts.
"""
failed_info(R::PartitionResult) = R.failed_info

@doc raw"""
    return_code(result::PartitionResult)

Return a symbolic status code for a partition result.
"""
return_code(R::PartitionResult) = R.return_code

@doc raw"""
    nregions(result::PartitionResult)

Return the number of connected components in the partition result.
"""
nregions(R::PartitionResult) = length(regions(R))

function Base.show(io::IO, R::PartitionResult)
    npars = nregions(R)
    nfailures = length(failed_info(R))
    header = "PartitionResult with $npars connected components"
    println(io, header)
    println(io, "="^length(header))
    println(io, "• return_code → :$(return_code(R))")
    println(io, "• $(nfailures) failed path(s)")
    print(io, "• morse_indices → ", isnothing(morse_indices(R)) ? "not computed" : length(morse_indices(R)))
end
