# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

# Properties of the gradient generator, as proven in the "Properties of the
# gradient generator" section of the Background page in the documentation

using Test
using LinearAlgebra
using StableRNGs: StableRNG
using QuantumPropagators.SpectralRange: specrange
using QuantumGradientGenerators: GradgenOperator, GradVector
using QuantumControlTestUtils.RandomObjects: random_matrix, random_state_vector


@testset "Spectrum, trace, determinant" begin
    rng = StableRNG(2026)
    N = 8
    Ĥ = random_matrix(N; hermitian = true, rng)
    μ̂₁ = random_matrix(N; hermitian = true, rng)
    μ̂₂ = random_matrix(N; hermitian = true, rng)
    G = GradgenOperator(Ĥ, [μ̂₁, μ̂₂])
    G_full = Array(G)

    evals = eigvals(G)
    @test evals isa Vector{Float64}  # real, for a Hermitian Ĥ
    @test length(evals) == 3N
    @test evals ≈ repeat(eigvals(Hermitian(Ĥ)); inner = 3)
    evals_full = sort(eigvals(G_full); by = real)
    @test maximum(abs.(imag.(evals_full))) < 1e-12
    @test norm(real.(evals_full) - evals) < 1e-10

    @test tr(G) ≈ tr(G_full)
    @test tr(G) ≈ 3 * tr(Ĥ)
    @test det(G) ≈ det(G_full)
    @test det(G) ≈ det(Ĥ)^3

    Ĥ_nh = random_matrix(N; rng)  # non-Hermitian
    G_nh = GradgenOperator(Ĥ_nh, [μ̂₁])
    @test eigvals(G_nh) isa Vector{ComplexF64}
    @test tr(G_nh) ≈ tr(Array(G_nh))
    @test det(G_nh) ≈ det(Array(G_nh))
end


@testset "Spectral range" begin
    rng = StableRNG(2027)
    N = 40
    Ĥ = random_matrix(N; hermitian = true, rng)
    μ̂ = random_matrix(N; hermitian = true, rng)
    G = GradgenOperator(Ĥ, [μ̂])

    E_min, E_max = extrema(eigvals(Hermitian(Ĥ)))
    @test all(specrange(G, :diag) .≈ (E_min, E_max))
    @test specrange(G, :diag) == specrange(Ĥ, :diag)
    @test specrange(G; rng = StableRNG(1)) == specrange(Ĥ; rng = StableRNG(1))  # :auto
    G_small = GradgenOperator(Ĥ[1:20, 1:20], [μ̂[1:20, 1:20]])  # size 40 > 32
    @test specrange(G_small) == specrange(Ĥ[1:20, 1:20], :diag)  # :auto → :diag
    @test specrange(G; E_min = -1, E_max = 1) == (-1.0, 1.0)

    # :arnoldi uses only the state of a `GradVector` as the starting vector
    Ψ = GradVector(random_state_vector(N; rng), 1)
    Ψ.grad_states[1] .= random_state_vector(N; rng)
    @test specrange(G, :arnoldi; state = Ψ) == specrange(Ĥ, :arnoldi; state = Ψ.state)
    @test specrange(G; method = :arnoldi, state = Ψ) ==
          specrange(Ĥ; method = :arnoldi, state = Ψ.state)
    E_min_a, E_max_a = specrange(G, :arnoldi; rng)
    @test E_min_a < E_min + 0.1 * (E_max - E_min)
    @test E_max_a > E_max - 0.1 * (E_max - E_min)
end


@testset "Jordan structure" begin
    rng = StableRNG(2028)
    N = 6
    Ĥ = random_matrix(N; hermitian = true, rng)
    μ̂ = random_matrix(N; hermitian = true, rng)
    G_full = Array(GradgenOperator(Ĥ, [μ̂]))
    F = eigen(Hermitian(Ĥ))
    for j = 1:N
        r = F.vectors[:, j]
        @test abs(dot(r, μ̂, r)) > 1e-3  # generic: Hellmann–Feynman derivative ≠ 0
        # algebraic multiplicity 2, but a one-dimensional eigenspace
        σ = svdvals(G_full - F.values[j] * I)
        @test σ[end] < 1e-12
        @test σ[end-1] > 1e-6
    end
end


@testset "Non-normality" begin
    rng = StableRNG(2029)
    N = 6
    Ĥ = random_matrix(N; hermitian = true, rng)
    μ̂ = random_matrix(N; hermitian = true, rng)
    Zero = zeros(ComplexF64, N, N)

    G_full = Array(GradgenOperator(Ĥ, [μ̂]))
    @test norm(G_full * G_full' - G_full' * G_full) > 1e-3
    @test !ishermitian(GradgenOperator(Ĥ, [μ̂]))
    G₀_full = Array(GradgenOperator(Ĥ, [Zero]))
    @test norm(G₀_full * G₀_full' - G₀_full' * G₀_full) < 1e-12

    # Rayleigh quotient outside the spectral range
    F = eigen(Hermitian(Ĥ))
    E_max = F.values[end]
    e = F.vectors[:, end]
    μ_ee = real(dot(e, μ̂, e))
    v = [sign(μ_ee) * e; e] / √2
    rq = real(dot(v, G_full, v))
    @test rq ≈ E_max + abs(μ_ee) / 2
    @test rq > E_max
end


@testset "Adjoint" begin
    rng = StableRNG(2030)
    N = 6
    Ĥ = random_matrix(N; rng)
    μ̂ = random_matrix(N; rng)
    Zero = zeros(ComplexF64, N, N)
    One = Matrix{ComplexF64}(I, N, N)
    P = [Zero One; One Zero]
    G_full = Array(GradgenOperator(Ĥ, [μ̂]))
    G_adj_full = Array(GradgenOperator(copy(Ĥ'), [copy(μ̂')]))
    @test norm(G_full' - G_adj_full) > 1e-3  # G[Ĥ]† ≠ G[Ĥ†]
    @test norm(G_full' - P * G_adj_full * P) < 1e-14
end


@testset "Gradient vector norm" begin
    rng = StableRNG(2031)
    N = 10
    Ψ = GradVector(random_state_vector(N; rng), 2)
    Ψ.grad_states[1] .= random_state_vector(N; rng)
    Ψ.grad_states[2] .= 2 * random_state_vector(N; rng)
    @test norm(Ψ) ≈ sqrt(real(dot(Ψ, Ψ)))
    @test norm(Ψ) ≈ norm(convert(Vector{ComplexF64}, Ψ))
    @test norm(Ψ) ≈ √6
end
