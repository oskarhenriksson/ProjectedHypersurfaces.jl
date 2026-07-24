export PseudoWitnessSet, degree, total_dim, system, trace_test, sample_points, decompose
struct Line{T<:Number}
    point::Vector{T}
    direction::Vector{T}
    linear_subspace::LinearSubspace
end

function Line(point::Vector{T}, direction::Vector{T}) where T<:Number
    A = transpose(direction) |> nullspace |> transpose
    b = A * point
    L = LinearSubspace(A, b)
    Line(point, direction, L)
end
struct PseudoWitnessSet{TF<:System,T<:Number,TT}
    F::TF
    k::Int
    L::Line{T}
    generic_witness_set::WitnessSet # WitnessSet for the upstairs variety
    W::Vector{Vector{ComplexF64}} # Unprojected witness points
    πW::Vector{Vector{ComplexF64}} # Projections of the upstairs witness points (without duplicates)
    tZ::Vector{Vector{ComplexF64}} # Restricted coordinates of the witness points used for tracking (one per fiber)
    tracker::TT
    track_report::Vector{Bool}
end
degree(PWS::PseudoWitnessSet) = length(PWS.πW)
total_dim(PWS::PseudoWitnessSet) = size(PWS.F, 2)
n_projection_variables(PWS::PseudoWitnessSet) = PWS.k
system(PWS::PseudoWitnessSet) = PWS.F
Base.show(io::IO, PWS::PseudoWitnessSet) = 
    print(io, "Pseudo-witness set of a hypersurface of degree $(degree(PWS)) in ambient dimension $(PWS.k)")

@doc raw"""
    PseudoWitnessSet(F::System, k::Int; linear_subspace_codim::Int, L::LinearSubspace)

Generates a pseudo witness set for the image of the variety $V(F)\subseteq\mathbb{C}^n$
for a system $F\in\mathbb{C}[x_1,\ldots,x_n]^r$ under the projection $\pi\colon\mathbb{C}^k\times\mathbb{C}^{n-k}\to\mathbb{C}^k$.
     

Optional inputs:

- `linear_subspace_codim`: The codimension of the linear space used for the witness set. Defaults to `n - length(F)`.
- `L`: The linear space used for the witness set. Should be the preimage under $\pi$ of a linear subspace in $\mathbb{C}^k$.
- `filter_condition`: A function that takes a point in $\mathbb{C}^{n}$ and returns a boolean. If provided, only points 
in the pseudo-witness set that satisfy this condition will be included. Can be used to filter out irrelevant irreducible components.

"""
function PseudoWitnessSet(
    F::System,
    k::Int;
    L::Union{Line, Nothing} = nothing,
    start_system::Symbol = :total_degree,
    compile::Union{Bool,Symbol} = :mixed,
    filter_condition::Union{Function,Nothing} = nothing,
    generic_witness_set::Union{WitnessSet,Nothing} = nothing
)
    
    n = size(F, 2)

    if isnothing(generic_witness_set)
        
        L_generic = rand_subspace(size(F, 2); codim = k - 1)
        generic_witness_result = HC.solve(F, target_subspace = L_generic, start_system = start_system)

        # Check for singular solutions
        if nsingular(generic_witness_result) > 0
            @warn "Irreducible component of higher multiplicity detected in the incidence variety."
        end

        # Generic witness points (filter out the relevant if a filter condition is provided)
        W_generic = solutions(generic_witness_result)
        if !isnothing(filter_condition)
            W_generic = filter(filter_condition, W_generic)
        end

        # Generic witness set (will be used for irreducible decomposition)
        generic_witness_set = WitnessSet(CompiledSystem(F), L_generic, W_generic)
    else
        L_generic = generic_witness_set.L
        W_generic = generic_witness_set.R
        if isa(W_generic, Vector{PathResult})
            W_generic = solutions(W_generic)
        end
    end

    # Track the generic witness points to the pseudo-witness points
    # The target linear space should be a lifting of a random line downstairs (TODO: check that the provided linear space has this structure)
    if isnothing(L)
        L = Line(randn(ComplexF64, k), randn(ComplexF64, k))
    end
    lifted_linear_subspace = LinearSubspace(hcat(L.linear_subspace.extrinsic.A, zeros(k-1, n-k)), L.linear_subspace.extrinsic.b)
    E = HC.solve(F, W_generic, start_subspace = L_generic, target_subspace = lifted_linear_subspace)
    
    # Repopulate the solution set via monodromy (safety feature if solutions were lost)
    M = monodromy_solve(F, solutions(E), lifted_linear_subspace)

    # Unprojected pseudo-witness points
    W = solutions(M)

    # Raise exception if we didn't find any witness points
    if length(W) == 0
        error("No witness points found.")
    end

    # Compute the t coordinates (so that w = [L.point + t * L.direction; w[k+1:end]] for each w in W)
    idx = argmax(abs.(L.direction))
    tW = map(W) do w
        p = w[1:k]
        t = (p - L.point)[idx] / L.direction[idx]
        [t; w]
    end

    # Restrict the ambient system to F([p + t * L.direction; w]) with p as the parameter.
    F_L = RestrictionToLineSystem(F, L.direction, k; compile = compile)

    # Form πW (the projections of the upstairs witness points without dublicates)
    # For each unique downstairs point, keep one fiber representative in in tZ
    unique_points = UniquePoints(first(tW)[2:k+1], 1)
    πW = Vector{Vector{ComplexF64}}()
    tZ = Vector{Vector{ComplexF64}}()
    for (i, tw) in enumerate(tW)
        _, new_point = add!(unique_points, tw[2:k+1], i)
        if new_point
            push!(πW, tw[2:k+1])
            push!(tZ, [tw[1]; tw[k+2:end]])
        end
    end

    # Set up tracker 
    tracker = Tracker(ParameterHomotopy(F_L, L.point, L.point))
    track_report = zeros(Bool, length(tZ)) # for keeping track of which paths are successful

    # Return the pseudo-witness set
    PseudoWitnessSet{typeof(F),ComplexF64,typeof(EndgameTracker(tracker))}(
        F,
        k,
        L,
        generic_witness_set,
        W,
        πW,
        tZ,
        EndgameTracker(tracker),
        track_report,
    )
end


@doc raw"""
    decompose(PWS::PseudoWitnessSet)

Performs a numerical irreducible decomposition of the upstairs variety of a pseudo-witness set `PWS` and returns a vector of pseudo-witness sets (one for each irreducible component).

If several upstairs components have the same downstairs projection, only one of these components will be used. 

"""
function decompose(PWS::PseudoWitnessSet)

    # Numerical irreducible decomposition of the upstairs variety
    components = HC.decompose(PWS.generic_witness_set)
    
    isempty(PWS.πW) && error("Pseudo-witness set is empty.")

    # Form a PWS from each component
    # Use the same linear space for all of them
    # Keep track of which components contribute new points
    new_pwss = PseudoWitnessSet[]
    L = PWS.L
    covered = UniquePoints(first(PWS.πW), 1)
    idx = 1
    for component in components
        new_pws = PseudoWitnessSet(
            PWS.F,
            PWS.k;
            L=L,
            generic_witness_set=component
        )

        isempty(new_pws.πW) && continue

        new_points = 0

        for point in new_pws.πW
            _, is_new = add!(covered, point, idx)

            if is_new
                new_points += 1
                idx += 1
            end
        end

        if new_points > 0
            push!(new_pwss, new_pws)
        end
    end

    return new_pwss
end


@doc raw"""
    track!(u::Vector{Vector{ComplexF64}}, PWS::PseudoWitnessSet, p::AbstractVector)

Given a pseudo-witness set `PWS` and a point `p` in the downstairs space, move the line downstains so that
it passes through `p` and track the pseudo-witness points.

The resulting points are stored in `u` and the success of each track is recorded in `PWS.track_report`.

"""
function track!(u::Vector{Vector{ComplexF64}}, PWS::PseudoWitnessSet, p::AbstractVector)
    tracker = PWS.tracker
    target_parameters!(tracker, p)
    # PWS.tZ contains the coordinates (t,Z) for the points where the line
    # (PWS.L.direction*t+PWS.L.point; Z) intersects V(F)
    for (l, w) in enumerate(PWS.tZ)
            HC.track!(tracker, w, 1)
            copyto!(u[l], tracker.tracker.state.x)
            PWS.track_report[l] = all(isfinite, u[l]) # note if the track was successful or not
    end

    nothing
end

function get_s_and_Uvals!(Uvals, S, GC, PWS)
    k = n_projection_variables(PWS)
    n = total_dim(PWS)

    
    for (j, sol) in enumerate(GC.line_hypersurface_intersections)
        !PWS.track_report[j] && continue # skip if j-th track failed
        @assert all(!isnan, sol) "NaN entries in intersection points: $sol"

        S[j] = 1 / sol[1] # We need S[j] = s = 1 / t, where t = sol[1]

         for idx in 1:n-k
            Uvals[idx, j] = sol[idx+1] 
        end
    end

    nothing
end

function track_projected_point(PWS::PseudoWitnessSet,p)
    tZ = deepcopy(PWS.tZ)
    track!(tZ,PWS,p)
    b = PWS.L.direction
    map(tZ) do tz
        t=first(tz)
        b*t+p
    end
end


@doc raw"""
    trace_test(PWS::PseudoWitnessSet)

Performs a trace test for completness of a pseudo-witness set; see [^LRS18] for details.

Returns a trace, which theoretically should be zero if and only if the pseudo-witness set
is complete (has all the witness points, meaning that we have correctly computed the degree of the variety).

Since we are working with floating point arithmetic, it will likely not be exactly zero.
A very low trace (e.g. on the order of 1e-16) is a strong heutistic indication that the pseudo-witness set is complete,
but does not constitute a proof.

[^LRS18] Leykin, Anton, Jose Israel Rodriguez, and Frank Sottile. "Trace test." Arnold Mathematical Journal 4.1 (2018): 113-125.

"""
function trace_test(PWS::PseudoWitnessSet)

    L = PWS.L
    p = L.point
    b = L.direction
    πW = PWS.πW

    s₀ = sum(πW)
    v = randn(ComplexF64,PWS.k)
    #translate our linear space by v
    p₋₁ = p-v
    p₁ = p+v
    πW₋₁ = track_projected_point(PWS,p₋₁)
    @assert all(PWS.track_report) "Failed paths detected"

    πW₁ = track_projected_point(PWS,p₁)
    @assert all(PWS.track_report) "Failed paths detected"

    s₋₁ = sum(πW₋₁)
    s₁ = sum(πW₁)

    M = [s₋₁ s₀ s₁; 1 1 1]
    singvals = LinearAlgebra.svdvals(M)
    trace = singvals[3] / singvals[1]

    trace
end



@doc raw"""
    sample_points(PWS::PseudoWitnessSet, N::Int)

Generate a sample of `N` points from a hypersurface represented by the pseudo-witness set `PWS`.
"""
function sample_points(PWS::PseudoWitnessSet, N::Int)

    # Decide how many linear spaces we need
    d = degree(PWS)
    number_of_linear_spaces = div(N, d, RoundUp)

    # Move the pseudo witness set line to generate new sample points
    πW₀ = PWS.πW
    p₀ = PWS.L.point
    sample = copy(πW₀)

    max_attempts = max(1, 10 * number_of_linear_spaces)
    attempts = 0
    while length(sample) < N && attempts < max_attempts
        v = randn(ComplexF64,PWS.k)
        p = p₀ + v
        πW_new = ProjectedHypersurfaces.track_projected_point(PWS, p)
        πW_new_succeeded = πW_new[PWS.track_report]
        append!(sample, πW_new_succeeded)
        attempts += 1
    end

    if length(sample) < N
        throw(ArgumentError("Failed to sample $N points: only $(length(sample)) tracks succeeded."))
    end
    
    # Return the desired number of sample points
    sample[1:N]

end