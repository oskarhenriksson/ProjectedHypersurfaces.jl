mutable struct GradientCache{T}
    v0::Vector{T}
    line_hypersurface_intersections::Vector{Vector{T}}
    JsuF::Vector{HC.CompiledSystem}
    JPF::Vector{HC.CompiledSystem}
    JBF::Vector{HC.CompiledSystem}
    HF::Matrix{HC.CompiledSystem}
    JxB::Matrix{HC.CompiledSystem}
    JxP::Matrix{HC.CompiledSystem}
    JPB::Matrix{HC.CompiledSystem}
    S::Vector{T}
    X::Vector{T}
    Uvals::Matrix{T}
    SP::Matrix{T}
    SB::Matrix{T}
    UP::Array{T,3}
    UB::Array{T,3}
    A::Array{T,4}
    rhs1::Matrix{T}
    rhs2::Vector{T}
    rhs3::Vector{T}
    JsuF_temp::Matrix{T}
    JPF_temp::Matrix{T}
    JBF_temp::Matrix{T}
    Jtu_temp::Matrix{T} # Temporary storage for evaluating JsuF
    HF_temp::Array{T, 3} # Temporary storage for evaluating HF
    JxB_temp::Array{T, 3} # Temporary storage for evaluating JxB
    JxP_temp::Array{T, 3} # Temporary storage for evaluating JxP
    JPB_temp::Array{T, 3} # Temporary storage for evaluating JPB
    temp_Hi::Matrix{T}
    temp_Jxpi::Matrix{T}
    temp_Jxbi::Matrix{T}
    temp_Jpbi::Matrix{T}
    ipiv::Vector{LinearAlgebra.LAPACK.BlasInt} # allocation for pivot for lu! in place linear solving
    M::Matrix{T}
    M1::Matrix{T}
    M2::Matrix{T}
    M3::Matrix{T}
    gradient_temp::Vector{T}
    Hess_temp::Matrix{T}
end
function compute_systems(F, n, k, B)
    @unique_var uval[1:n-k] α[1:k] β[1:k] t
    F_on_line = F([α + (1 / t) * β; uval])
    v = vcat(t, uval)
    vars = vcat(t, uval, α)

    ∇v = map(v) do vi
        HC.ModelKit.differentiate(F_on_line, vi) 
    end
    ∇α = map(α) do αi
        HC.ModelKit.differentiate(F_on_line, αi)
    end

    JsuF = map(∇v) do ∇vi
        g = evaluate(∇vi, β => B)
        System(g, variables = vars)
    end
    JPF = map(∇α) do ∇vi
        g = evaluate(∇vi, β => B)
        System(g, variables = vars)
    end
    JBF = map(F_on_line) do f
        g = evaluate(HC.ModelKit.differentiate(f, β), β => B)
        System(g, variables = vars)
    end

    function J(x) 
        map(Iterators.product(∇v, x)) do (∇vi, xj)
            hess_ij = evaluate(HC.ModelKit.differentiate(∇vi, xj), β => B)
            System(hess_ij, variables = vars) 
        end
    end

    HF = J(v)
    JxB = J(β)
    JxP = J(α)
    JPB = map(Iterators.product(∇α, β)) do (∇αi, βj)
            hess_ij = evaluate(HC.ModelKit.differentiate(∇αi, βj), β => B)
            System(hess_ij, variables = vars) 
        end
    return CompiledSystem.(JsuF), CompiledSystem.(JPF), CompiledSystem.(JBF), CompiledSystem.(HF), CompiledSystem.(JxB), CompiledSystem.(JxP), CompiledSystem.(JPB)

end

function GradientCache(PWS)
    d = degree(PWS)
    k = n_projection_variables(PWS)
    F = PWS.F
    L = PWS.L
    N, n = size(F)

    @assert N == n-k+1 "Unexpected length of system"

    line_hypersurface_intersections = [zeros(ComplexF64, n - k + 1) for _ in 1:d]
  
    @unique_var t, p[1:k]

    S = zeros(ComplexF64, d)
    X = zeros(ComplexF64, k)
    Uvals = zeros(ComplexF64, n - k, d)
    SP = zeros(ComplexF64, d, k)
    SB = zeros(ComplexF64, d, k)
    UP = zeros(ComplexF64, d, n - k, k)
    UB = zeros(ComplexF64, d, n - k, k)
    A = zeros(ComplexF64, d, N, k, k) 

    JsuF, JPF, JBF, HF, JxB, JxP, JPB = compute_systems(F, n, k, L.b)


    # 
    rhs1 = zeros(ComplexF64, N, 2*k)  
    rhs2 = zeros(ComplexF64, N)  
    rhs3 = zeros(ComplexF64, k)  

    JsuF_temp = zeros(ComplexF64, N, 1+n-k)
    JPF_temp = zeros(ComplexF64, N, k)
    JBF_temp = zeros(ComplexF64, k, N)
    Jtu_temp = zeros(ComplexF64, N, 1+n-k) # TODO: Maybe can reuse Jsu_temp....
    HF_temp = zeros(ComplexF64, N, size(HF)...)
    JxB_temp = zeros(ComplexF64, N, size(JxB)...) # size(JxB)
    JxP_temp = zeros(ComplexF64, N, size(JxP)...)
    JPB_temp = zeros(ComplexF64, N, size(JPB)...)
    temp_Hi = zeros(ComplexF64, size(HF)...)
    temp_Jxpi = zeros(ComplexF64, size(JxB)...)
    temp_Jxbi = zeros(ComplexF64, size(JxP)...)
    temp_Jpbi = zeros(ComplexF64, size(JPB)...)

    ipiv = Vector{LinearAlgebra.LAPACK.BlasInt}(undef, min(size(JsuF_temp,1), size(JsuF_temp,2)))

    M = zeros(ComplexF64, k, k)
    M1 = zeros(ComplexF64, k, n-k+1)
    M2 = zeros(ComplexF64, n-k+1, k)
    M3 = zeros(ComplexF64, k, n-k+1)

    gradient_temp = zeros(ComplexF64, k)
    Hess_temp = zeros(ComplexF64, k, k)

    v0 = randn(ComplexF64, n+1)

    GradientCache{ComplexF64}(v0, 
                    line_hypersurface_intersections,
                    JsuF,
                    JPF,
                    JBF,
                    HF,
                    JxB,
                    JxP,
                    JPB,
                    S, 
                    X, 
                    Uvals, 
                    SP, 
                    SB, 
                    UP, 
                    UB, 
                    A, 
                    rhs1, 
                    rhs2, 
                    rhs3,
                    JsuF_temp, 
                    JPF_temp, 
                    JBF_temp, 
                    Jtu_temp, 
                    HF_temp, 
                    JxB_temp, 
                    JxP_temp, 
                    JPB_temp, 
                    temp_Hi,
                    temp_Jxpi,
                    temp_Jxbi,
                    temp_Jpbi,
                    ipiv,
                    M, 
                    M1, 
                    M2, 
                    M3, 
                    gradient_temp, 
                    Hess_temp
                )
end

track!(GC::GradientCache, PWS::PseudoWitnessSet, p) = track!(GC.line_hypersurface_intersections, PWS, p) 
