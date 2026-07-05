export interpolate, InterpolationResult, polynomial

"""
    InterpolationResult

An object that contains the result of interpolating a projected hypersurface to obtain a polynomial representation of the discriminant.

"""
struct InterpolationResult
    polynomial::Expression
    coefficients::Vector{Rational{Int}}
    exponent_vectors::Vector{Tuple{Vararg{Int}}}
    singular_values::Vector{Float64}
    σ_min::Float64
    σ_gap::Float64
    residual::Float64
    h::ProjectedHypersurface
end


"""
    polynomial(result::InterpolationResult)

Returns the polynomial of an [`InterpolationResult`](@ref) object. 
"""
polynomial(result::InterpolationResult) = result.polynomial

function Base.show(io::IO, result::InterpolationResult) 
    header = "Interpolation result for projected hypersurface"
    println(io, header) 
    println(io, "="^(length(header)))
    println(io, " Smallest singular value: ", round(result.σ_min, sigdigits=5))
    println(io, " Ratio of next-smallest to smallest singular value: ", round(result.σ_gap, sigdigits=5))
    println(io, " Residual: ", round(result.residual, sigdigits=5))
    println(io, "-"^(length(header)))
    println(io, " Variables: ", join(result.h.projection_vars, ", "))
    println(io, " Polynomial: ", result.polynomial)
end


"""
    interpolate(
    h::ProjectedHypersurface;
    tol::Float64=1e-8,
    oversampling_factor=1.5
)

Interpolate the projected hypersurface `h` to obtain a polynomial representation of the discriminant. 
The interpolation is performed by sampling points on the hypersurface via [`sample_points`](@ref) 
and finding an element of the nullspace of the Vandermonde matrix via singular value decomposition. 
The resulting polynomial is normalized to have integer coefficients with the smallest possible common denominator.

The output is an [`InterpolationResult`](@ref) object containing the polynomial. It also includes some numbers that quantify
the quality of the interpolation:

- `σ_min`: The smallest singular value of the Vandermonde matrix (should be small if the interpolation is successful).
- `σ_gap`: The ratio of the next-smallest singular value to the smallest singular value (should be large if the interpolation is successful).
- `residual`: The residual of the interpolation, computed as the norm of the Vandermonde matrix times the coefficients divided by the norm of the coefficients (should be small if the interpolation is successful).

Input:
- `h`: A [`ProjectedHypersurface`](@ref) object representing the hypersurface to be interpolated.

Keyword arguments:
- `tol`: The tolerance for rationalizing the coefficients.
- `oversampling_factor`: The factor by which to oversample the points.

"""
function interpolate(
    h::ProjectedHypersurface;
    tol::Float64=1e-8,
    oversampling_factor=1.5
)
    variables = h.projection_vars
    d = degree(h)
    k = length(variables)

    number_of_terms = binomial(d+k,k)
    sample_size = Int(ceil(number_of_terms * oversampling_factor))

    sample = sample_points(h, sample_size)

    # Vector of exponent vectors of degree at most d
    exponent_vectors = [α for α in Iterators.product([0:d for i=1:k]...) if sum(α) ≤ d]

    # Vandermonde matrix
    A = [prod(sample[i].^α) for i in 1:sample_size, α in exponent_vectors]
    A_real = vcat(real(A), imag(A))

    # Renormalize the rows
    for i = 1:size(A_real,1)
        A_real[i,:] = A_real[i,:]/norm(A_real[i,:])
    end

    res = svd(A_real)
    coefficients = res.V[:, end]
    singular_values = res.S
    σ_min = res.S[end]
    σ_gap = res.S[end-1]/res.S[end]

    # Rescale coefficients and rationalize
    coefficients = coefficients ./ maximum(abs, coefficients)
    coefficients = rationalize.(coefficients, tol=tol)
    D = lcm(denominator.(coefficients)...)
    coefficients = Int.(D .* coefficients)
    coefficients = coefficients .÷ gcd.(coefficients...)

    # Normalize the sign of first nonzero coefficient
    if coefficients[findfirst(!iszero, coefficients)] < 0 
        coefficients .= -coefficients
    end

    # Form the polynomial
    polynomial = sum(c*prod(variables.^α) for (c, α) in zip(coefficients, exponent_vectors))

    # Compute the residual
    residual = norm(A_real * float.(coefficients)) / norm(float.(coefficients))

    return InterpolationResult(
        polynomial,
        coefficients,
        exponent_vectors,
        singular_values,
        σ_min,
        σ_gap,
        residual,
        h
    )
end