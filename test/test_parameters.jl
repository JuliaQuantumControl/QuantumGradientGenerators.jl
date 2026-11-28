# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

# Gradient generators for the derivatives with respect to control parameters
# (`GradGenerator(H, parameters)`) and with respect to a given list of controls
# (`GradGenerator(H, controls)`), including derivatives that are `nothing`

using Test
using LinearAlgebra: norm, dot, mul!
using StableRNGs: StableRNG
using QuantumPropagators: hamiltonian, propagate, ExpProp
using QuantumPropagators.Controls:
    get_controls, get_parameters, combine_parameters, evaluate, StaticParameter
using QuantumPropagators.ParameterizedFunctions: CRABFunction, VariedFrequencyCRABFunction
using QuantumPropagators.Interfaces: supports_matrix_interface
using QuantumControl.Interfaces: check_generator
using QuantumControlTestUtils.RandomObjects: random_matrix, random_state_vector
using QuantumGradientGenerators: GradGenerator, GradgenOperator, GradVector
using RecursiveArrayTools: ArrayPartition
import OrdinaryDiffEq
using OrdinaryDiffEq: Vern9


const T = 5.0
const N = 4


# The central finite-difference derivative of `propagate(Ψ₀, H, tlist; kwargs...)`
# with respect to each element of the parameters `u` of `H`. The parameters are
# perturbed in place and restored afterwards.
function fd_derivs(Ψ₀, H, tlist, u; h = 1e-5, kwargs...)
    derivs = Vector{Vector{ComplexF64}}(undef, length(u))
    for n in eachindex(u)
        u₀ = u[n]
        u[n] = u₀ + h
        Ψ_plus = propagate(Ψ₀, H, tlist; kwargs...)
        u[n] = u₀ - h
        Ψ_minus = propagate(Ψ₀, H, tlist; kwargs...)
        u[n] = u₀
        derivs[n] = (Ψ_plus - Ψ_minus) / 2h
    end
    return derivs
end


# The maximum error of the gradient states of `Ψ` relative to the given
# derivatives, normalized to the largest derivative
function max_rel_error(Ψ, derivs)
    scale = maximum(norm, derivs)
    return maximum(norm(Ψ.grad_states[n] - derivs[n]) for n in eachindex(derivs)) / scale
end


@testset "StaticParameter" begin

    rng = StableRNG(1001)
    Ĥ₀ = random_matrix(N; hermitian = true, rng)
    Ĥ₁ = random_matrix(N; hermitian = true, rng)
    Ĥ₂ = random_matrix(N; hermitian = true, rng)
    Ψ₀ = random_state_vector(N; rng)
    tlist = collect(range(0, T; length = 101))

    # Only the first and third elements of `p` enter the generator, so the
    # derivative with respect to `p[2]` is `nothing`
    p = [0.3, 0.7, -0.2]
    ϵ(t) = 0.2 * sin(π * t / T)^2
    H = hamiltonian(Ĥ₀, (Ĥ₁, StaticParameter(p, 1)), (Ĥ₂, StaticParameter(p, 3)), (Ĥ₁, ϵ))

    u = get_parameters(H)
    @test u isa ArrayPartition
    G = GradGenerator(H, u)
    @test G.wrt === u
    @test length(G.derivs) == 3
    @test isnothing(G.derivs[2])
    @test G.derivs[1] == Ĥ₁
    @test G.derivs[3] == Ĥ₂
    @test get_controls(G) == get_controls(H)
    @test get_parameters(G) == u

    Ψ = propagate(GradVector(Ψ₀, 3), G, tlist; method = ExpProp)
    @test norm(Ψ.state - propagate(Ψ₀, H, tlist; method = ExpProp)) < 1e-12
    @test norm(Ψ.grad_states[2]) == 0.0  # `nothing` is a zero block
    dΨ_fd = fd_derivs(Ψ₀, H, tlist, u; method = ExpProp)
    @test norm(dΨ_fd[2]) == 0.0
    @test max_rel_error(Ψ, dΨ_fd) < 1e-8

    # The gradient generator remains valid when the parameters change
    p .= [-0.1, 0.4, 0.5]
    Ψ = propagate(GradVector(Ψ₀, 3), G, tlist; method = ExpProp)
    @test max_rel_error(Ψ, fd_derivs(Ψ₀, H, tlist, u; method = ExpProp)) < 1e-8

    # Problem-level parameters with a block that `H` does not hold
    q = [1.0, 2.0]
    u_problem = combine_parameters(q, u)
    G = GradGenerator(H, u_problem)
    @test G.wrt === u_problem
    @test length(G.derivs) == 5
    @test all(isnothing, G.derivs[1:2])
    @test isnothing(G.derivs[4])
    Ψ = propagate(GradVector(Ψ₀, 5), G, tlist; method = ExpProp)
    dΨ_fd = fd_derivs(Ψ₀, H, tlist, u_problem; method = ExpProp)
    @test all(norm(Ψ.grad_states[n]) == 0.0 for n in (1, 2, 4))
    @test max_rel_error(Ψ, dΨ_fd) < 1e-8

    # A single block
    G = GradGenerator(H, combine_parameters(p))
    @test length(G.derivs) == 3
    @test G.wrt.x === (p,)

end


@testset "CRAB parameters" begin

    rng = StableRNG(1002)
    Ĥ₀ = random_matrix(N; hermitian = true, rng)
    Ĥ₁ = random_matrix(N; hermitian = true, rng)
    Ψ₀ = random_state_vector(N; rng)
    tlist = collect(range(0, T; length = 201))
    shape(t) = 0.3 * sin(π * t / T)^2

    crab = CRABFunction(
        2;
        frequencies = [0.8, 1.9],
        guess = t -> 0.1,
        shape,
        rng,
        random_amplitude = true,
    )
    H = hamiltonian(Ĥ₀, (Ĥ₁, crab))
    u = get_parameters(H)
    @test length(u) == 5  # c₀, two cosine and two sine coefficients
    G = GradGenerator(H, u)
    @test G.wrt === u
    @test !any(isnothing, G.derivs)

    # Piecewise-constant propagation: the gradient is the exact derivative of
    # the propagated state
    Ψ = propagate(GradVector(Ψ₀, length(u)), G, tlist; method = ExpProp)
    @test max_rel_error(Ψ, fd_derivs(Ψ₀, H, tlist, u; method = ExpProp)) < 1e-8

    # Time-continuous propagation with an ODE solver
    ode_kwargs = (; method = OrdinaryDiffEq, alg = Vern9(), reltol = 1e-12, abstol = 1e-12)
    Ψ = propagate(GradVector(Ψ₀, length(u)), G, [0.0, T]; ode_kwargs...)
    dΨ_fd = fd_derivs(Ψ₀, H, [0.0, T], u; h = 1e-4, ode_kwargs...)
    @test max_rel_error(Ψ, dΨ_fd) < 1e-6

end


@testset "VariedFrequencyCRABFunction parameters" begin

    rng = StableRNG(1003)
    Ĥ₀ = random_matrix(N; hermitian = true, rng)
    Ĥ₁ = random_matrix(N; hermitian = true, rng)
    Ψ₀ = random_state_vector(N; rng)
    tlist = collect(range(0, T; length = 201))
    shape(t) = 0.3 * sin(π * t / T)^2

    crab = VariedFrequencyCRABFunction(2; frequencies = [0.8, 1.9], shape, rng)
    H = hamiltonian(Ĥ₀, (Ĥ₁, crab))
    u = get_parameters(H)
    @test length(u) == 6  # two cosine, two sine, and two frequency parameters
    G = GradGenerator(H, u)

    Ψ = propagate(GradVector(Ψ₀, length(u)), G, tlist; method = ExpProp)
    @test max_rel_error(Ψ, fd_derivs(Ψ₀, H, tlist, u; method = ExpProp)) < 1e-8

    ode_kwargs = (; method = OrdinaryDiffEq, alg = Vern9(), reltol = 1e-12, abstol = 1e-12)
    Ψ = propagate(GradVector(Ψ₀, length(u)), G, [0.0, T]; ode_kwargs...)
    dΨ_fd = fd_derivs(Ψ₀, H, [0.0, T], u; h = 1e-4, ode_kwargs...)
    @test max_rel_error(Ψ, dΨ_fd) < 1e-6

end


@testset "Controls in a different order, and subsets" begin

    # The scenario of GRAPE issue #129: the trajectories of a control problem
    # depend on the problem's controls in a different order, or on a subset of
    # them. The gradient states must be aligned with the problem's controls.

    rng = StableRNG(1004)
    Ĥ₀ = random_matrix(N; hermitian = true, rng)
    Ĥ₁ = random_matrix(N; hermitian = true, rng)
    Ĥ₂ = random_matrix(N; hermitian = true, rng)
    Ψ₀ = random_state_vector(N; rng)
    ϵ₁(t) = 0.2
    ϵ₂(t) = -0.3
    controls = (ϵ₁, ϵ₂)
    dt = 0.7
    tlist = [0.0, dt]
    vals = IdDict(ϵ₁ => 0.2, ϵ₂ => -0.3)

    # The derivative of a single time step with respect to each control value
    function fd_control_derivs(H; h = 1e-6)
        derivs = Vector{Vector{ComplexF64}}()
        for ϵ in controls
            vals_plus = IdDict(vals..., ϵ => vals[ϵ] + h)
            vals_minus = IdDict(vals..., ϵ => vals[ϵ] - h)
            U_plus = exp(-1im * Array(evaluate(H, tlist, 1; vals_dict = vals_plus)) * dt)
            U_minus = exp(-1im * Array(evaluate(H, tlist, 1; vals_dict = vals_minus)) * dt)
            push!(derivs, (U_plus - U_minus) * Ψ₀ / 2h)
        end
        return derivs
    end

    function propagate_step(G)
        Ψ = GradVector(Ψ₀, length(G.derivs))
        G_op = evaluate(G, tlist, 1; vals_dict = vals)
        Ψ_full = exp(-1im * Array(G_op) * dt) * convert(Vector{ComplexF64}, Ψ)
        return convert(typeof(Ψ), Ψ_full)
    end

    H_swapped = hamiltonian(Ĥ₀, (Ĥ₂, ϵ₂), (Ĥ₁, ϵ₁))
    @test get_controls(H_swapped) == (ϵ₂, ϵ₁)
    G = GradGenerator(H_swapped, controls)
    @test G.wrt == [ϵ₁, ϵ₂]
    @test G.wrt isa Vector
    @test G.derivs == [Ĥ₁, Ĥ₂]
    Ψ = propagate_step(G)
    @test max_rel_error(Ψ, fd_control_derivs(H_swapped)) < 1e-8
    # Without the controls, the gradient states are in the order of the
    # controls of the generator
    @test GradGenerator(H_swapped).derivs == [Ĥ₂, Ĥ₁]
    @test GradGenerator(H_swapped).wrt == [ϵ₂, ϵ₁]

    H_subset = hamiltonian(Ĥ₀, (Ĥ₂, ϵ₂))
    G = GradGenerator(H_subset, controls)
    @test length(G.derivs) == 2
    @test isnothing(G.derivs[1])
    @test G.derivs[2] == Ĥ₂
    Ψ = propagate_step(G)
    @test norm(Ψ.grad_states[1]) == 0.0
    @test max_rel_error(Ψ, fd_control_derivs(H_subset)) < 1e-8

    # Also with a vector of controls, and for a propagation with ExpProp (on
    # two time steps, which for constant controls is the same as one step)
    G = GradGenerator(H_subset, [ϵ₁, ϵ₂])
    @test G.wrt == [ϵ₁, ϵ₂]
    Ψ = propagate(GradVector(Ψ₀, 2), G, [0.0, dt / 2, dt]; method = ExpProp)
    @test norm(Ψ.grad_states[1]) == 0.0
    @test max_rel_error(Ψ, fd_control_derivs(H_subset)) < 1e-8

    @test check_generator(
        G;
        state = GradVector(Ψ₀, 2),
        tlist = collect(range(0, T; length = 11)),
        for_gradient_optimization = false
    )

end


@testset "Invalid wrt" begin

    rng = StableRNG(1005)
    Ĥ₀ = random_matrix(N; hermitian = true, rng)
    Ĥ₁ = random_matrix(N; hermitian = true, rng)
    pulse = rand(rng, 10)
    H = hamiltonian(Ĥ₀, (Ĥ₁, pulse))

    # A vector of real numbers (e.g., a single pulse, or a parameter block) is
    # neither a list of controls nor the parameters
    @test_throws ArgumentError GradGenerator(H, pulse)
    @test_throws "pass `(ϵ,)`" GradGenerator(H, pulse)
    @test_throws "combine_parameters(block)" GradGenerator(H, [0.1, 0.2])
    @test GradGenerator(H, (pulse,)).derivs == [Ĥ₁]

    @test_throws DimensionMismatch GradGenerator(H, [Ĥ₁], (pulse, pulse))
    G = GradGenerator(H, [Ĥ₁], (pulse,))
    @test G.wrt === (pulse,)

end


@testset "GradgenOperator with zero blocks" begin

    rng = StableRNG(1006)
    Ĥ = random_matrix(N; rng)
    μ̂₁ = random_matrix(N; rng)
    μ̂₃ = random_matrix(N; rng)
    Zero = zeros(ComplexF64, N, N)
    G = GradgenOperator(Ĥ, [μ̂₁, nothing, μ̂₃])
    G_ref = GradgenOperator(Ĥ, [μ̂₁, Zero, μ̂₃])
    @test G isa GradgenOperator{3,Matrix{ComplexF64},Union{Nothing,Matrix{ComplexF64}}}
    @test supports_matrix_interface(typeof(G))
    @test eltype(G) == ComplexF64
    @test size(G) == (4N, 4N)
    G_full = Array(G)
    @test G_full == Array(G_ref)
    @test convert(Matrix{ComplexF64}, G) == G_full
    @test all(G[i, j] == G_full[i, j] for i = 1:size(G, 1), j = 1:size(G, 2))
    @test collect(G) == vec(G_full)

    Ψ = GradVector(random_state_vector(N; rng), 3)
    Φ = GradVector(random_state_vector(N; rng), 3)
    for k = 1:3
        Ψ.grad_states[k] .= random_state_vector(N; rng)
        Φ.grad_states[k] .= random_state_vector(N; rng)
    end
    Ψ_full = convert(Vector{ComplexF64}, Ψ)

    @test norm(convert(Vector{ComplexF64}, G * Ψ) - G_full * Ψ_full) < 1e-12
    Θ = copy(Φ)
    α = 0.3 - 1.1im
    β = 0.7
    mul!(Θ, G, Ψ, α, β)
    Θ_ref = α * (G_full * Ψ_full) + β * convert(Vector{ComplexF64}, Φ)
    @test norm(convert(Vector{ComplexF64}, Θ) - Θ_ref) < 1e-12
    mul!(Θ, G, Ψ)
    @test norm(convert(Vector{ComplexF64}, Θ) - G_full * Ψ_full) < 1e-12
    # flat
    Φ_flat = convert(Vector{ComplexF64}, Φ)
    mul!(Φ_flat, G, Ψ_full, α, β)
    @test norm(Φ_flat - Θ_ref) < 1e-12

    @test dot(Φ, G, Ψ) ≈ dot(convert(Vector{ComplexF64}, Φ), G_full * Ψ_full)
    @test norm(Array(2.0 * G) - 2.0 * G_full) < 1e-12
    @test isnothing((G * 2.0).derivs[2])

    @test !isreal(G)
    @test isreal(GradgenOperator(real(Ĥ), [real(μ̂₁), nothing]))
    @test isreal(GradgenOperator(real(Ĥ), [nothing, nothing]))

    # With an `Operator` as the dynamical generator, and a control that it
    # does not depend on
    ϵ(t) = 0.4
    ϵ_other(t) = 0.0
    G_op = evaluate(GradGenerator(hamiltonian(Ĥ, (μ̂₁, ϵ)), (ϵ_other, ϵ)), 0.0)
    @test isnothing(G_op.derivs[1])
    Ψ₂ = GradVector(Ψ.state, 2)
    Φ₂ = GradVector(Φ.state, 2)
    for k = 1:2
        Ψ₂.grad_states[k] .= Ψ.grad_states[k]
        Φ₂.grad_states[k] .= Φ.grad_states[k]
    end
    @test dot(Φ₂, G_op, Ψ₂) ≈ dot(Φ₂, G_op * Ψ₂)
    @test dot(Φ₂, G_op, Ψ₂) ≈ dot(
        convert(Vector{ComplexF64}, Φ₂),
        Array(G_op) * convert(Vector{ComplexF64}, Ψ₂)
    )

    # All derivatives zero
    G₀ = GradgenOperator(Ĥ, [nothing, nothing])
    @test G₀ isa GradgenOperator{2,Matrix{ComplexF64},Nothing}
    @test supports_matrix_interface(typeof(G₀))
    @test eltype(G₀) == ComplexF64
    @test Array(G₀) == Array(GradgenOperator(Ĥ, [Zero, Zero]))
    Ψ₂ = GradVector(Ψ.state, 2)
    @test all(norm(Ψ′) == 0.0 for Ψ′ in (G₀ * Ψ₂).grad_states)

end
