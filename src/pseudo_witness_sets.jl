export PseudoWitnessSet, degree, total_dim, system, witness_points
struct Line 
    p::Vector
    b::Vector 
end

struct PseudoWitnessSet
    F::System
    k::Int
    L::Line
    Wt::Vector
    πW::Vector
    tracker::EndgameTracker
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
    start_system::Symbol = :polyhedral,
    compile::Union{Bool,Symbol} = :mixed 
)
 
    if isnothing(L)
        L = Line(randn(ComplexF64, k), randn(ComplexF64, k))
    end
    
    # Intersect with random linear subspace
    @unique_var t, p[1:k]
    v = variables(F)
    F_L = System([F.expressions; p + t .* L.b - v[1:k]], variables = [v; t], parameters = p)

    # Trace the nonsingular solutions 
    E = HC.solve(F_L; start_system = start_system,
                    target_parameters = L.p)

     # Check for singular solutions
    if nsingular(E) > 0
        @warn "Irreducible component of higher multiplicity detected in the incidence variety."
    end



    # Repopulate the solution set via monodromy (safetey feature if solutions were lost)
    M = monodromy_solve(F_L, solutions(E), L.p)
    Wt = solutions(M)
    # πW = unique_points([w[1:k] for w in Wt])

    if length(Wt)==0
        @error "No witness points found."
    end
    
    unique_points = UniquePoints(first(Wt)[1:k], 1)
    πW = Vector{Vector{ComplexF64}}()
    fiber_representatives = Vector{Vector{ComplexF64}}()
    for (i, vᵢ) in enumerate(Wt)
        _, new_point = add!(unique_points, vᵢ[1:k], i)
        if new_point
            push!(πW, vᵢ[1:k])
            push!(fiber_representatives, vᵢ)
        end
    end
    Wt = fiber_representatives

    # Set up tracker 
    tracker = Tracker(ParameterHomotopy(fixed(F_L; compile = compile), L.p, L.p))
    track_report = zeros(Bool, length(solutions(M))) # for keeping track of which paths are successful

    PseudoWitnessSet(F, k, L, Wt, πW, EndgameTracker(tracker), track_report)
end

witness_points(PWS::PseudoWitnessSet) = PWS.πW

function track!(u::Vector, PWS::PseudoWitnessSet, p)
    tracker = PWS.tracker
    target_parameters!(tracker, p)
    for (l, w) in enumerate(PWS.Wt)
            HC.track!(tracker, w, 1)
            u[l] .= tracker.tracker.state.x
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

        S[j] = 1 / sol[end] # We need S[j] = s = 1 / t, where t = sol[end]

         for idx in 1:n-k
            Uvals[idx, j] = sol[idx+k] 
        end
    end

    nothing
end



