export PseudoWitnessSet, degree, total_dim, system
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
- `L`: The linear space used for the witness set. Should be the preimage under $\pi$ of a linear subspace in $\mathbb{C}^k$

"""
function PseudoWitnessSet(
    F::System,
    k::Int;
    L::Union{Line, Nothing} = nothing,
    start_system::Symbol = :total_degree,
    compile::Union{Bool,Symbol} = :mixed,
    filter_condition::Union{Function,Nothing} = nothing,
)
 
    if isnothing(L)
        L = Line(randn(ComplexF64, k), randn(ComplexF64, k))
    end
    
    # Restrict the ambient system to F([p + t * L.direction; w]) with p as the parameter.
    F_L = RestrictionToLineSystem(F, L.direction, k; compile = compile)

    # Intersect with random linear subspace
    # we want to use G = [F.expressions; p + t .* L.direction - v[1:k]] instead of F_L to be able to use polyhedral/total_degree
    v = variables(F)
    p = F_L.parameters
    t = F_L.t
    G = System([F.expressions; p + t .* L.direction - v[1:k]], variables = [t; v], parameters = p)

    # Trace the nonsingular solutions
    E = HC.solve(G; start_system = start_system,
                    target_parameters = L.point)

     # Check for singular solutions
    if nsingular(E) > 0
        @warn "Irreducible component of higher multiplicity detected in the incidence variety."
    end

    # Repopulate the solution set via monodromy (safetey feature if solutions were lost)
    # TODO: Make a trace test here?
    M = monodromy_solve(G, solutions(E), L.point)
    n = length(v)

    tW = solutions(M)

    if !isnothing(filter_condition)
        tW = filter(tw -> filter_condition(tw[2:end]), tW)
    end

    # Raise exception if we didn't find any witness points
    if length(tW) == 0
        error("No witness points found.")
    end

    W = [tw[2:end] for tw in tW]

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
        W,
        πW,
        tZ,
        EndgameTracker(tracker),
        track_report,
    )
end

function track!(u::Vector{Vector{ComplexF64}}, PWS::PseudoWitnessSet, p::AbstractVector)
    tracker = PWS.tracker
    target_parameters!(tracker, p)
    # Update one tracker instance in place for each target parameter.
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