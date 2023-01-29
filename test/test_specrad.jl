using Test
using LinearAlgebra
using QuantumPropagators.SpectralRange: specrange
using QuantumGradientGenerators: GradGenerator
using QuantumControlBase.TestUtils
using QuantumPropagators.Controls: evaluate


@testset "Gradgen specrad" begin

    N = 10  # size of Hilbert space
    ρ = 1.0  # spectral radius
    Ĥ₀ = random_hermitian_matrix(N, ρ)
    Ĥ₁ = random_hermitian_matrix(N, ρ)
    Ĥ₂ = random_hermitian_matrix(N, ρ)
    Zero = zeros(ComplexF64, N, N)
    ϵ₁ = t -> 1.0
    ϵ₂ = t -> 1.0
    Ĥ_of_t = (Ĥ₀, (Ĥ₁, ϵ₁), (Ĥ₂, ϵ₂))
    vals_dict = IdDict(ϵ₁ => 1.0, ϵ₂ => 1.0)
    G̃_of_t = GradGenerator(Ĥ_of_t)
    Ĥ = evaluate(Ĥ_of_t; vals_dict)
    G̃ = evaluate(G̃_of_t; vals_dict)

    G_expected = [
         Ĥ    Zero  Ĥ₁
        Zero   Ĥ    Ĥ₂
        Zero  Zero  Ĥ
    ]

    G = Array(G̃)

    @test norm(G_expected - G) < 1e-14

    @test maximum(imag.(eigvals(G))) < 1e-15
    @test maximum(imag.(eigvals(Ĥ))) < 1e-15

    H_E_min = minimum(real.(eigvals(Ĥ)))
    H_E_max = maximum(real.(eigvals(Ĥ)))

    G_E_min = minimum(real.(eigvals(G)))
    G_E_max = maximum(real.(eigvals(G)))
    @test abs(H_E_min - G_E_min) < 1e-12
    @test abs(H_E_max - G_E_max) < 1e-12

    G_range_diag = collect(specrange(G̃, method=:diag))
    @test eltype(G_range_diag) ≡ Float64
    @test norm(G_range_diag - [G_E_min, G_E_max]) < 1e-12

    H_range_diag = collect(specrange(Ĥ, method=:diag))
    @test norm(H_range_diag - G_range_diag) < 1e-12

    G_range_arnoldi = collect(specrange(G̃, method=:arnoldi, m_max=100))
    @test eltype(G_range_arnoldi) ≡ Float64
    @test norm(G_range_arnoldi - [H_E_min, H_E_max]) < 1e-2
    # `specrange(Ĥ, method=:arnoldi)` isn't very exact, so we don't
    # compare against that.

end


@testset "Gradgen-sparse specrad" begin
    N = 100  # size of Hilbert space
    ρ = 1.0  # spectral radius
    sparsity = 0.1
    Ĥ₀ = random_hermitian_sparse_matrix(N, ρ, sparsity)
    Ĥ₁ = random_hermitian_sparse_matrix(N, ρ, sparsity)
    Ĥ₂ = random_hermitian_sparse_matrix(N, ρ, sparsity)
    Zero = zeros(ComplexF64, N, N)
    ϵ₁ = t -> 1.0
    ϵ₂ = t -> 1.0
    Ĥ_of_t = (Ĥ₀, (Ĥ₁, ϵ₁), (Ĥ₂, ϵ₂))
    vals_dict = IdDict(ϵ₁ => 1.0, ϵ₂ => 1.0)
    G̃_of_t = GradGenerator(Ĥ_of_t)
    Ĥ = evaluate(Ĥ_of_t; vals_dict)
    G̃ = evaluate(G̃_of_t; vals_dict)

    G_expected = Array([
         Ĥ    Zero  Ĥ₁
        Zero   Ĥ    Ĥ₂
        Zero  Zero  Ĥ
    ])

    G = Array(G̃)
    @test G isa Matrix{ComplexF64}

    @test norm(G_expected - G) < 1e-12

    @test maximum(imag.(eigvals(G))) < 1e-15
    @test maximum(imag.(eigvals(Array(Ĥ)))) < 1e-15

    H_E_min = minimum(real.(eigvals(Array(Ĥ))))
    H_E_max = maximum(real.(eigvals(Array(Ĥ))))

    G_E_min = minimum(real.(eigvals(G)))
    G_E_max = maximum(real.(eigvals(G)))
    @test abs(H_E_min - G_E_min) < 1e-12
    @test abs(H_E_max - G_E_max) < 1e-12

    G_range_diag = collect(specrange(G̃, method=:diag))
    @test eltype(G_range_diag) ≡ Float64
    @test norm(G_range_diag - [G_E_min, G_E_max]) < 1e-12

    G_range_arnoldi = collect(specrange(G̃, method=:arnoldi, m_max=100))
    @test eltype(G_range_arnoldi) ≡ Float64
    @test norm(G_range_arnoldi - [G_E_min, G_E_max]) < 0.2

end
