using Test
using QuantumControl: hamiltonian
using QuantumControl.Controls: get_controls
using QuantumControlTestUtils.RandomObjects: random_matrix, random_state_vector
using QuantumControl.Interfaces: check_generator, check_state
using QuantumGradientGenerators: GradGenerator, GradVector
using LinearAlgebra: norm


@testset "GradVector Interface" begin

    N = 10
    Ψ = random_state_vector(N)
    Ψ̃ = GradVector(Ψ, 2)
    @test check_state(Ψ̃)

    @test norm(2.2 * Ψ̃ - Ψ̃ * 2.2) < 1e-14

end


@testset "GradGenerator Interface" begin

    N = 10
    Ĥ₀ = random_matrix(N, hermitian=true)
    Ĥ₁ = random_matrix(N, hermitian=true)
    Ĥ₂ = random_matrix(N, hermitian=true)
    ϵ₁(t) = 1.0
    ϵ₂(t) = 1.0
    Ĥ_of_t = hamiltonian(Ĥ₀, (Ĥ₁, ϵ₁), (Ĥ₂, ϵ₂))

    tlist = collect(range(0, 10; length=101))

    G̃_of_t = GradGenerator(Ĥ_of_t)

    Ψ = random_state_vector(N)
    Ψ̃ = GradVector(Ψ, length(get_controls(G̃_of_t)))

    @test check_generator(G̃_of_t; state=Ψ̃, tlist, for_gradient_optimization=false)

end
