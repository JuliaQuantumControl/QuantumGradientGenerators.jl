# SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test
using QuantumPropagators.Generators: hamiltonian
using QuantumPropagators.Controls: get_controls
using QuantumControlTestUtils.RandomObjects: random_matrix, random_state_vector
using QuantumControl.Interfaces: check_generator
using QuantumPropagators.Interfaces:
    check_state, check_operator, supports_matrix_interface, supports_vector_interface
using QuantumGradientGenerators: GradGenerator, GradVector, GradgenOperator
using StaticArrays: SVector, SMatrix, MVector
using LinearAlgebra: norm, dot, mul!, I
using StableRNGs: StableRNG


@testset "GradVector Interface" begin

    rng = StableRNG(1179926107)
    N = 10
    Ψ₀ = random_state_vector(N; rng)
    Ψ = GradVector(Ψ₀, 2)
    # Non-zero gradient states: with zero gradient states, any norm of a
    # GradVector equals the norm of its state, so `check_state` could not
    # detect a `norm` that is inconsistent with `dot`
    for l in eachindex(Ψ.grad_states)
        Ψ.grad_states[l] .= random_state_vector(N; rng)
    end
    @test check_state(Ψ)

    @test norm(2.2 * Ψ - Ψ * 2.2) < 1e-14

end


@testset "GradVector Interface (Static)" begin

    rng = StableRNG(2188051723)
    N = 10
    Ψ₀ = SVector{N,ComplexF64}(random_state_vector(N; rng))
    Ψ = GradVector(Ψ₀, 2)
    # Non-zero gradient states: with zero gradient states, any norm of a
    # GradVector equals the norm of its state, so `check_state` could not
    # detect a `norm` that is inconsistent with `dot`
    for l in eachindex(Ψ.grad_states)
        Ψ.grad_states[l] = SVector{N,ComplexF64}(random_state_vector(N; rng))
    end
    @test check_state(Ψ)

    @test norm(2.2 * Ψ - Ψ * 2.2) < 1e-14

    Ψ2 = similar(Ψ)
    @test Ψ2 isa GradVector{2,<:MVector}

    # We've had propagators use code like
    #
    #   v0::ST = similar(Ψ::ST)
    #
    # which relies on being able to convert mutable types back to their
    # immutable version
    Ψ3 = convert(typeof(Ψ), Ψ2)
    @test typeof(Ψ3) == typeof(Ψ)

end


@testset "GradGenerator Interface" begin

    rng = StableRNG(3031820470)
    N = 10
    Ĥ₀ = random_matrix(N; hermitian = true, rng)
    Ĥ₁ = random_matrix(N; hermitian = true, rng)
    Ĥ₂ = random_matrix(N; hermitian = true, rng)
    ϵ₁(t) = 1.0
    ϵ₂(t) = 1.0
    Ĥ_of_t = hamiltonian(Ĥ₀, (Ĥ₁, ϵ₁), (Ĥ₂, ϵ₂))

    tlist = collect(range(0, 10; length = 101))

    G_of_t = GradGenerator(Ĥ_of_t)

    Ψ₀ = random_state_vector(N; rng)
    Ψ = GradVector(Ψ₀, length(get_controls(G_of_t)))

    @test check_generator(G_of_t; state = Ψ, tlist, for_gradient_optimization = false)

end


@testset "GradGenerator Interface (Static)" begin

    rng = StableRNG(1911203795)
    N = 10
    Ĥ₀ = SMatrix{N,N,ComplexF64}(random_matrix(N; hermitian = true, rng))
    Ĥ₁ = SMatrix{N,N,ComplexF64}(random_matrix(N; hermitian = true, rng))
    Ĥ₂ = SMatrix{N,N,ComplexF64}(random_matrix(N; hermitian = true, rng))
    ϵ₁(t) = 1.0
    ϵ₂(t) = 1.0
    Ĥ_of_t = hamiltonian(Ĥ₀, (Ĥ₁, ϵ₁), (Ĥ₂, ϵ₂))

    tlist = collect(range(0, 10; length = 101))

    G_of_t = GradGenerator(Ĥ_of_t)

    Ψ₀ = SVector{N,ComplexF64}(random_state_vector(N; rng))
    Ψ = GradVector(Ψ₀, length(get_controls(G_of_t)))

    @test check_generator(G_of_t; state = Ψ, tlist, for_gradient_optimization = false)

end


@testset "GradgenOperator Matrix Interface" begin

    rng = StableRNG(3317751223)
    N = 5
    L = 2
    H = Matrix{ComplexF64}(I, N, N)
    μ = [rand(rng, ComplexF64, N, N) for _ = 1:L]
    G = GradgenOperator{L,Matrix{ComplexF64},Matrix{ComplexF64}}(H, μ)
    Ψ = GradVector(rand(rng, ComplexF64, N), L)

    # supports_matrix_interface reports true for matrix-backed GradgenOperator
    @test supports_matrix_interface(typeof(G))

    # check_operator passes the full matrix interface check including for_expval
    @test check_operator(G; state = Ψ, for_expval = true)

    # getindex is consistent with the dense Array representation
    dense = Array(G)
    @test all(G[i, j] ≈ dense[i, j] for i = 1:size(G, 1), j = 1:size(G, 2))

    # length
    @test length(G) == prod(size(G))

    # iterate visits elements in column-major order, consistent with vec(Array(G))
    @test all(collect(G) .≈ vec(dense))

    # 3-arg mul! agrees with 5-arg mul!(Φ, G, Ψ, true, false)
    Φ1 = GradVector(zeros(ComplexF64, N), L)
    Φ2 = GradVector(zeros(ComplexF64, N), L)
    mul!(Φ1, G, Ψ)
    mul!(Φ2, G, Ψ, true, false)
    @test norm(Φ1 - Φ2) < 1e-14

    # isreal requires real values in all blocks
    @test !isreal(G)
    @test isreal(GradgenOperator(H, [real(μₗ) for μₗ in μ]))
    @test !isreal(GradgenOperator(1im * H, [real(μₗ) for μₗ in μ]))

    # 3-arg dot(Ψ, G, Φ) matches dot(Ψ, G * Φ)
    Φ = GradVector(rand(rng, ComplexF64, N), L)
    @test dot(Ψ, G, Φ) ≈ dot(Ψ, G * Φ)

    # similar(G) returns a dense Array of the same eltype and size (matching Operator pattern)
    G_sim = similar(G)
    @test G_sim isa Array{eltype(G)}
    @test size(G_sim) == size(G)

    # similar(G, S) returns a dense Array of type S with matching size
    @test similar(G, Float64) isa Array{Float64}
    @test size(similar(G, Float64)) == size(G)

    # similar(G, dims) returns a dense Array with given dims
    @test similar(G, (3, 4)) isa Array{eltype(G)}
    @test size(similar(G, (3, 4))) == (3, 4)

    # similar(G, S, dims) returns a dense Array of type S with given dims
    @test similar(G, Float64, (3, 4)) isa Array{Float64}
    @test size(similar(G, Float64, (3, 4))) == (3, 4)

end


@testset "GradgenOperator flat-vector mul!" begin

    rng = StableRNG(1602052280)
    N = 5
    L = 2
    H = rand(rng, ComplexF64, N, N)
    μ = [rand(rng, ComplexF64, N, N) for _ = 1:L]
    G = GradgenOperator{L,Matrix{ComplexF64},Matrix{ComplexF64}}(H, μ)
    dense = Array(G)

    # Flat-vector layout: [Ψ′₁ | Ψ′₂ | ... | Ψ′_L | Ψ_state]
    Ψ_flat = rand(rng, ComplexF64, N * (L + 1))
    Φ_ref = dense * Ψ_flat

    # 3-arg mul! agrees with the dense matrix-vector product
    Φ1 = zeros(ComplexF64, N * (L + 1))
    mul!(Φ1, G, Ψ_flat)
    @test norm(Φ1 - Φ_ref) < 1e-14

    # 5-arg mul! with α=true, β=false matches the 3-arg result
    Φ2 = zeros(ComplexF64, N * (L + 1))
    mul!(Φ2, G, Ψ_flat, true, false)
    @test norm(Φ2 - Φ_ref) < 1e-14

    # 5-arg mul! with non-trivial α, β: Φ ← α * G * Ψ + β * Φ
    α = 2.3 + 0.7im
    β = 1.5 - 0.3im
    Φ3 = rand(rng, ComplexF64, N * (L + 1))
    Φ3_init = copy(Φ3)
    mul!(Φ3, G, Ψ_flat, α, β)
    @test norm(Φ3 - (α * Φ_ref + β * Φ3_init)) < 1e-14

    # works with views (the actual ExponentialUtilities use case)
    V = rand(rng, ComplexF64, N * (L + 1), 4)
    mul!(view(V, :, 2), G, view(V, :, 1))
    @test norm(view(V, :, 2) - dense * view(V, :, 1)) < 1e-14

end


@testset "GradVector Vector Interface" begin

    rng = StableRNG(2618946253)
    N = 5
    L = 2
    Ψ₀ = rand(rng, ComplexF64, N)
    gradvec = GradVector(Ψ₀, L)
    # Non-zero gradient states: with zero gradient states, any norm of a
    # GradVector equals the norm of its state, so `check_state` could not
    # detect a `norm` that is inconsistent with `dot`
    for l in eachindex(gradvec.grad_states)
        gradvec.grad_states[l] .= rand(rng, ComplexF64, N)
    end

    # supports_vector_interface is true for Vector-backed GradVector
    @test supports_vector_interface(typeof(gradvec))

    # check_state passes full vector interface check
    @test check_state(gradvec)

    # size is 1D with total length
    @test size(gradvec) == (N * (L + 1),)
    @test size(gradvec) == (length(gradvec),)

    # getindex is consistent with convert_gradvec_to_dense layout:
    # [grad_states[1]; grad_states[2]; ...; grad_states[L]; state]
    dense = convert(Vector{ComplexF64}, gradvec)
    @test all(gradvec[k] == dense[k] for k = 1:length(gradvec))

    # iterate visits elements consistent with getindex
    @test all(collect(gradvec) .== dense)

    # eachindex covers all valid indices
    @test eachindex(gradvec) == 1:length(gradvec)
    @test all(gradvec[k] == dense[k] for k in eachindex(gradvec))

    # setindex! round-trips through getindex
    gradvec2 = GradVector(copy(Ψ₀), L)
    for k = 1:length(gradvec2)
        gradvec2[k] = gradvec[k]
    end
    @test all(gradvec2[k] == gradvec[k] for k = 1:length(gradvec))

    # similar(gradvec, S) returns a mutable Vector{S} with same length
    @test similar(gradvec, ComplexF32) isa Vector{ComplexF32}
    @test length(similar(gradvec, ComplexF32)) == length(gradvec)

    # similar(gradvec, dims) returns a plain Array with same eltype and given dims
    @test similar(gradvec, (3, 4)) isa Array{eltype(gradvec)}
    @test size(similar(gradvec, (3, 4))) == (3, 4)

end


@testset "GradVector Vector Interface (Static)" begin

    rng = StableRNG(167987434)
    N = 5
    L = 2
    Ψ₀ = SVector{N,ComplexF64}(rand(rng, ComplexF64, N))
    gradvec = GradVector(Ψ₀, L)
    # Non-zero gradient states: with zero gradient states, any norm of a
    # GradVector equals the norm of its state, so `check_state` could not
    # detect a `norm` that is inconsistent with `dot`
    for l in eachindex(gradvec.grad_states)
        gradvec.grad_states[l] = SVector{N,ComplexF64}(rand(rng, ComplexF64, N))
    end

    # SVector-backed GradVector: supports_vector_interface follows the component type
    @test supports_vector_interface(typeof(gradvec))

    # check_state passes (SVector is inplace=false, so setindex! is not checked)
    @test check_state(gradvec)

    # getindex is consistent with the dense layout
    dense = convert(Vector{ComplexF64}, gradvec)
    @test all(gradvec[k] == dense[k] for k = 1:length(gradvec))

end


@testset "GradVector without Vector Interface" begin

    rng = StableRNG(4252840018)
    N = 5
    L = 2
    # Matrix is not an AbstractVector, so supports_vector_interface returns false
    Ψ₀ = rand(rng, ComplexF64, N, N)
    gradvec = GradVector(Ψ₀, L)

    @test !supports_vector_interface(typeof(gradvec))

    # check_state still passes via the basic (non-vector) state interface
    @test check_state(gradvec)

    # Vector interface methods must throw an error when not supported
    @test_throws "does not support the vector interface" gradvec[1]
    @test_throws "does not support the vector interface" (gradvec[1] = 0.0)
    @test_throws "does not support the vector interface" size(gradvec)
    @test_throws "does not support the vector interface" length(gradvec)
    @test_throws "does not support the vector interface" iterate(gradvec)
    @test_throws "does not support the vector interface" eachindex(gradvec)

end



# A wrapper type with no supports_matrix_interface declaration (defaults to false)
struct NonMatrixOp
    data::Matrix{ComplexF64}
end

@testset "GradgenOperator without Matrix Interface" begin

    rng = StableRNG(1276996367)
    N = 5
    L = 2
    H = NonMatrixOp(rand(rng, ComplexF64, N, N))
    μ = [NonMatrixOp(rand(rng, ComplexF64, N, N)) for _ = 1:L]
    G = GradgenOperator{L,NonMatrixOp,NonMatrixOp}(H, μ)

    @test !supports_matrix_interface(typeof(G))

    # Matrix interface methods must throw an error when not supported
    @test_throws "does not support the matrix interface" G[1, 1]
    @test_throws "does not support the matrix interface" length(G)
    @test_throws "does not support the matrix interface" iterate(G)

end
