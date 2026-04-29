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
    Ψ = random_state_vector(N; rng)
    Ψ̃ = GradVector(Ψ, 2)
    @test check_state(Ψ̃)

    @test norm(2.2 * Ψ̃ - Ψ̃ * 2.2) < 1e-14

end


@testset "GradVector Interface (Static)" begin

    rng = StableRNG(2188051723)
    N = 10
    Ψ = SVector{N,ComplexF64}(random_state_vector(N; rng))
    Ψ̃ = GradVector(Ψ, 2)
    @test check_state(Ψ̃)

    @test norm(2.2 * Ψ̃ - Ψ̃ * 2.2) < 1e-14

    Ψ̃2 = similar(Ψ̃)
    @test Ψ̃2 isa GradVector{2,<:MVector}

    # We've had propagators use code like
    #
    #   v0::ST = similar(Ψ::ST)
    #
    # which relies on being able to convert mutable types back to their
    # immutable version
    Ψ̃3 = convert(typeof(Ψ̃), Ψ̃2)
    @test typeof(Ψ̃3) == typeof(Ψ̃)

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

    G̃_of_t = GradGenerator(Ĥ_of_t)

    Ψ = random_state_vector(N; rng)
    Ψ̃ = GradVector(Ψ, length(get_controls(G̃_of_t)))

    @test check_generator(G̃_of_t; state = Ψ̃, tlist, for_gradient_optimization = false)

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

    G̃_of_t = GradGenerator(Ĥ_of_t)

    Ψ = SVector{N,ComplexF64}(random_state_vector(N; rng))
    Ψ̃ = GradVector(Ψ, length(get_controls(G̃_of_t)))

    @test check_generator(G̃_of_t; state = Ψ̃, tlist, for_gradient_optimization = false)

end


@testset "GradgenOperator Matrix Interface" begin

    rng = StableRNG(3317751223)
    N = 5
    L = 2
    G = Matrix{ComplexF64}(I, N, N)
    mu = [rand(rng, ComplexF64, N, N) for _ = 1:L]
    op = GradgenOperator{L,Matrix{ComplexF64},Matrix{ComplexF64}}(G, mu)
    state = GradVector(rand(rng, ComplexF64, N), L)

    # supports_matrix_interface reports true for matrix-backed GradgenOperator
    @test supports_matrix_interface(typeof(op))

    # check_operator passes the full matrix interface check including for_expval
    @test check_operator(op; state, for_expval = true)

    # getindex is consistent with the dense Array representation
    dense = Array(op)
    @test all(op[i, j] ≈ dense[i, j] for i = 1:size(op, 1), j = 1:size(op, 2))

    # length
    @test length(op) == prod(size(op))

    # iterate visits elements in column-major order, consistent with vec(Array(op))
    @test all(collect(op) .≈ vec(dense))

    # 3-arg mul! agrees with 5-arg mul!(Phi, G, Psi, true, false)
    Psi = GradVector(rand(rng, ComplexF64, N), L)
    Phi1 = GradVector(zeros(ComplexF64, N), L)
    Phi2 = GradVector(zeros(ComplexF64, N), L)
    mul!(Phi1, op, Psi)
    mul!(Phi2, op, Psi, true, false)
    @test norm(Phi1 - Phi2) < 1e-14

    # 3-arg dot(Psi, op, Phi) matches dot(Psi, op * Phi)
    Psi2 = GradVector(rand(rng, ComplexF64, N), L)
    @test dot(state, op, Psi2) ≈ dot(state, op * Psi2)

    # similar(op) returns a dense Array of the same eltype and size (matching Operator pattern)
    op_sim = similar(op)
    @test op_sim isa Array{eltype(op)}
    @test size(op_sim) == size(op)

    # similar(op, S) returns a dense Array of type S with matching size
    @test similar(op, Float64) isa Array{Float64}
    @test size(similar(op, Float64)) == size(op)

    # similar(op, dims) returns a dense Array with given dims
    @test similar(op, (3, 4)) isa Array{eltype(op)}
    @test size(similar(op, (3, 4))) == (3, 4)

    # similar(op, S, dims) returns a dense Array of type S with given dims
    @test similar(op, Float64, (3, 4)) isa Array{Float64}
    @test size(similar(op, Float64, (3, 4))) == (3, 4)

end


@testset "GradgenOperator flat-vector mul!" begin

    rng = StableRNG(1602052280)
    N = 5
    L = 2
    G = rand(rng, ComplexF64, N, N)
    mu = [rand(rng, ComplexF64, N, N) for _ = 1:L]
    op = GradgenOperator{L,Matrix{ComplexF64},Matrix{ComplexF64}}(G, mu)
    dense = Array(op)

    # Flat-vector layout: [grad_1 | grad_2 | ... | grad_L | state]
    Ψ_flat = rand(rng, ComplexF64, N * (L + 1))
    Φ_ref = dense * Ψ_flat

    # 3-arg mul! agrees with the dense matrix-vector product
    Φ1 = zeros(ComplexF64, N * (L + 1))
    mul!(Φ1, op, Ψ_flat)
    @test norm(Φ1 - Φ_ref) < 1e-14

    # 5-arg mul! with α=true, β=false matches the 3-arg result
    Φ2 = zeros(ComplexF64, N * (L + 1))
    mul!(Φ2, op, Ψ_flat, true, false)
    @test norm(Φ2 - Φ_ref) < 1e-14

    # 5-arg mul! with non-trivial α, β: Φ ← α * G * Ψ + β * Φ
    α = 2.3 + 0.7im
    β = 1.5 - 0.3im
    Φ3 = rand(rng, ComplexF64, N * (L + 1))
    Φ3_init = copy(Φ3)
    mul!(Φ3, op, Ψ_flat, α, β)
    @test norm(Φ3 - (α * Φ_ref + β * Φ3_init)) < 1e-14

    # works with views (the actual ExponentialUtilities use case)
    V = rand(rng, ComplexF64, N * (L + 1), 4)
    mul!(view(V, :, 2), op, view(V, :, 1))
    @test norm(view(V, :, 2) - dense * view(V, :, 1)) < 1e-14

end


@testset "GradVector Vector Interface" begin

    rng = StableRNG(2618946253)
    N = 5
    L = 2
    Psi = rand(rng, ComplexF64, N)
    gradvec = GradVector(Psi, L)

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

    # setindex! round-trips through getindex
    gradvec2 = GradVector(copy(Psi), L)
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
    Psi = SVector{N,ComplexF64}(rand(rng, ComplexF64, N))
    gradvec = GradVector(Psi, L)

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
    Psi = rand(rng, ComplexF64, N, N)
    gradvec = GradVector(Psi, L)

    @test !supports_vector_interface(typeof(gradvec))

    # check_state still passes via the basic (non-vector) state interface
    @test check_state(gradvec)

    # Vector interface methods must throw an error when not supported
    @test_throws "does not support the vector interface" gradvec[1]
    @test_throws "does not support the vector interface" (gradvec[1] = 0.0)
    @test_throws "does not support the vector interface" size(gradvec)
    @test_throws "does not support the vector interface" length(gradvec)
    @test_throws "does not support the vector interface" iterate(gradvec)

end



# A wrapper type with no supports_matrix_interface declaration (defaults to false)
struct NonMatrixOp
    data::Matrix{ComplexF64}
end

@testset "GradgenOperator without Matrix Interface" begin

    rng = StableRNG(1276996367)
    N = 5
    L = 2
    G = NonMatrixOp(rand(rng, ComplexF64, N, N))
    mu = [NonMatrixOp(rand(rng, ComplexF64, N, N)) for _ = 1:L]
    op = GradgenOperator{L,NonMatrixOp,NonMatrixOp}(G, mu)

    @test !supports_matrix_interface(typeof(op))

    # Matrix interface methods must throw an error when not supported
    @test_throws "does not support the matrix interface" op[1, 1]
    @test_throws "does not support the matrix interface" length(op)
    @test_throws "does not support the matrix interface" iterate(op)

end
