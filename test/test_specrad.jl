# SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test
using LinearAlgebra
using StableRNGs: StableRNG
using QuantumPropagators.SpectralRange: specrange
using QuantumGradientGenerators: GradGenerator
using QuantumControlTestUtils.RandomObjects: random_matrix
using QuantumPropagators.Controls: evaluate


@testset "Gradgen specrad" begin

    N = 10  # size of Hilbert space
    ρ = 1.0  # spectral radius
    Ĥ₀ = random_matrix(N; hermitian = true, spectral_radius = ρ)
    Ĥ₁ = random_matrix(N; hermitian = true, spectral_radius = ρ)
    Ĥ₂ = random_matrix(N; hermitian = true, spectral_radius = ρ)
    Zero = zeros(ComplexF64, N, N)
    ϵ₁ = t -> 1.0
    ϵ₂ = t -> 1.0
    Ĥ_of_t = (Ĥ₀, (Ĥ₁, ϵ₁), (Ĥ₂, ϵ₂))
    vals_dict = IdDict(ϵ₁ => 1.0, ϵ₂ => 1.0)
    G_of_t = GradGenerator(Ĥ_of_t)
    Ĥ = evaluate(Ĥ_of_t; vals_dict)
    G = evaluate(G_of_t; vals_dict)

    G_expected = [
         Ĥ    Zero  Ĥ₁
        Zero   Ĥ    Ĥ₂
        Zero  Zero  Ĥ
    ]

    G_full = Array(G)

    @test norm(G_expected - G_full) < 1e-14

    @test maximum(imag.(eigvals(G_full))) < 1e-15
    @test maximum(imag.(eigvals(Ĥ))) < 1e-15

    H_E_min = minimum(real.(eigvals(Ĥ)))
    H_E_max = maximum(real.(eigvals(Ĥ)))

    G_E_min = minimum(real.(eigvals(G_full)))
    G_E_max = maximum(real.(eigvals(G_full)))
    @test abs(H_E_min - G_E_min) < 1e-12
    @test abs(H_E_max - G_E_max) < 1e-12

    G_range_diag = collect(specrange(G, method = :diag))
    @test eltype(G_range_diag) ≡ Float64
    @test norm(G_range_diag - [G_E_min, G_E_max]) < 1e-12

    H_range_diag = collect(specrange(Ĥ, method = :diag))
    @test norm(H_range_diag - G_range_diag) < 1e-12

    # The spectral range of G is computed from Ĥ
    G_range_arnoldi =
        collect(specrange(G, method = :arnoldi, m_max = 100, rng = StableRNG(1)))
    @test eltype(G_range_arnoldi) ≡ Float64
    H_range_arnoldi =
        collect(specrange(Ĥ, method = :arnoldi, m_max = 100, rng = StableRNG(1)))
    @test G_range_arnoldi == H_range_arnoldi
    # The Arnoldi estimate is enlarged to enclose the exact spectral range
    @test G_range_arnoldi[1] <= H_E_min
    @test G_range_arnoldi[2] >= H_E_max

end


@testset "Gradgen-sparse specrad" begin
    N = 100  # size of Hilbert space
    ρ = 1.0  # spectral radius
    density = 0.1
    Ĥ₀ = random_matrix(N; hermitian = true, spectral_radius = ρ, density)
    Ĥ₁ = random_matrix(N; hermitian = true, spectral_radius = ρ, density)
    Ĥ₂ = random_matrix(N; hermitian = true, spectral_radius = ρ, density)
    Zero = zeros(ComplexF64, N, N)
    ϵ₁ = t -> 1.0
    ϵ₂ = t -> 1.0
    Ĥ_of_t = (Ĥ₀, (Ĥ₁, ϵ₁), (Ĥ₂, ϵ₂))
    vals_dict = IdDict(ϵ₁ => 1.0, ϵ₂ => 1.0)
    G_of_t = GradGenerator(Ĥ_of_t)
    Ĥ = evaluate(Ĥ_of_t; vals_dict)
    G = evaluate(G_of_t; vals_dict)

    G_expected = Array([
         Ĥ    Zero  Ĥ₁
        Zero   Ĥ    Ĥ₂
        Zero  Zero  Ĥ
    ])

    G_full = Array(G)
    @test G_full isa Matrix{ComplexF64}

    @test norm(G_expected - G_full) < 1e-12

    @test maximum(imag.(eigvals(G_full))) < 1e-15
    @test maximum(imag.(eigvals(Array(Ĥ)))) < 1e-15

    H_E_min = minimum(real.(eigvals(Array(Ĥ))))
    H_E_max = maximum(real.(eigvals(Array(Ĥ))))

    G_E_min = minimum(real.(eigvals(G_full)))
    G_E_max = maximum(real.(eigvals(G_full)))
    @test abs(H_E_min - G_E_min) < 1e-12
    @test abs(H_E_max - G_E_max) < 1e-12

    G_range_diag = collect(specrange(G, method = :diag))
    @test eltype(G_range_diag) ≡ Float64
    @test norm(G_range_diag - [G_E_min, G_E_max]) < 1e-12

    G_range_arnoldi =
        collect(specrange(G, method = :arnoldi, m_max = 100, rng = StableRNG(1)))
    @test eltype(G_range_arnoldi) ≡ Float64
    @test G_range_arnoldi[1] <= G_E_min
    @test G_range_arnoldi[2] >= G_E_max
    @test norm(G_range_arnoldi - [G_E_min, G_E_max]) < 0.5

end
