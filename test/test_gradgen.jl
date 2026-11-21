# SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test
using LinearAlgebra
using QuantumPropagators: init_prop, prop_step!
using QuantumPropagators.Newton
using QuantumGradientGenerators: GradGenerator, GradgenOperator, GradVector, resetgradvec!
using QuantumControlTestUtils.RandomObjects: random_matrix, random_state_vector
using Zygote
using QuantumPropagators.Controls: evaluate


@testset "GradgenOperator" begin

    N = 10  # size of Hilbert space
    ρ = 1.0  # spectral radius
    # We'll test with non-Hermitian Hamiltonians
    Ĥ₀ = random_matrix(N)
    Ĥ₁ = random_matrix(N)
    Ĥ₂ = random_matrix(N)
    Zero = zeros(ComplexF64, N, N)
    ϵ₁(t) = 1.0
    ϵ₂(t) = 1.0
    Ĥ_of_t = (Ĥ₀, (Ĥ₁, ϵ₁), (Ĥ₂, ϵ₂))
    Ψ₀ = random_state_vector(N)
    ψ_max = maximum(abs.(Ψ₀))
    Ψtgt = random_state_vector(N)
    𝕚 = 1im
    dt = 1.25

    vals_dict = IdDict(ϵ₁ => 1.0, ϵ₂ => 1.0)

    G_of_t = GradGenerator(Ĥ_of_t)
    G = evaluate(G_of_t; vals_dict)
    Ĥ = evaluate(Ĥ_of_t; vals_dict)

    G2 = GradgenOperator(G.H, G.control_derivs)
    @test G2 == G

    Û_Ψ₀ = exp(-𝕚 * Ĥ * dt) * Ψ₀

    num_controls = length(Ĥ_of_t) - 1
    Ψ = GradVector(Ψ₀, num_controls)
    # did the initialization work?
    @test norm(Ψ.state - Ψ₀) < 1e-14
    @test norm(Ψ.grad_states[1]) == 0.0
    @test norm(Ψ.grad_states[2]) == 0.0
    @test length(Ψ.grad_states) == 2

    @test maximum(abs.(Ψ₀)) == ψ_max  # is Ψ₀ still exactly the same state?
    wrk = NewtonWrk(Ψ)
    newton!(Ψ, G, dt, wrk)
    Ψ_out = copy(Ψ)
    @test maximum(abs.(Ψ₀)) == ψ_max  # is Ψ₀ still exactly the same state?
    @test norm(Ψ_out.state - Û_Ψ₀) < 1e-12  # Ψ_out.state correct?
    @test norm(Ψ_out.grad_states[1]) > 0
    @test norm(Ψ_out.grad_states[2]) > 0
    # did the copy work?
    @test norm(Ψ_out.state - Ψ.state) == 0.0
    @test norm(Ψ_out.grad_states[1] - Ψ.grad_states[1]) == 0.0
    @test norm(Ψ_out.grad_states[2] - Ψ.grad_states[2]) == 0.0

    # reset Ψ to the original input (also to test the resetgradvec! method)
    resetgradvec!(Ψ, Ψ₀)
    # state should be the same as after previous initialization
    @test norm(Ψ.state - Ψ₀) < 1e-14
    @test norm(Ψ.grad_states[1]) == 0.0
    @test norm(Ψ.grad_states[2]) == 0.0

    @test norm(Ψ_out.state - Û_Ψ₀) < 1e-12  # Ψ_out.state still correct?

    ###########################################################################
    # Compare against explicit Grad-Gen
    # This checks whether the application of a GradgenOperator to a GradVector an
    # all the linear-algebra methods are implemented correctly

    G_full = [
         Ĥ    Zero  Ĥ₁
        Zero   Ĥ    Ĥ₂
        Zero  Zero  Ĥ
    ]

    Ψ_full = [
        Ψ.grad_states[1]
        Ψ.grad_states[2]
        Ψ.state
    ]

    @test size(G) == size(G_full)
    @test eltype(G) == eltype(G_full)

    # proper initialization? grad_states should be zero
    @test norm(Ψ_full) == norm(Ψ.state) == norm(Ψ₀)

    Ψ_out_full = exp(-𝕚 * G_full * dt) * Ψ_full
    # propagation correct?
    @test norm(Ψ_out_full[(2N+1):3N] - Û_Ψ₀) < 1e-10

    # do we get the same results as from newton?
    @test norm(Ψ_out_full[(2N+1):3N] - Ψ_out.state) < 1e-10
    @test norm(Ψ_out_full[1:N] - Ψ_out.grad_states[1]) < 1e-10
    @test norm(Ψ_out_full[(N+1):2N] - Ψ_out.grad_states[2]) < 1e-10

    ###########################################################################
    # Test custom expprop


    Ψ_full = [
        Ψ.grad_states[1]
        Ψ.grad_states[2]
        Ψ.state
    ]

    # proper initialization? grad_states should be zero
    @test norm(Ψ_full) == norm(Ψ.state) == norm(Ψ₀)

    Ψ_out_full = exp(-𝕚 * G_full * dt) * Ψ_full
    # propagation correct?
    @test norm(Ψ_out_full[(2N+1):3N] - Û_Ψ₀) < 1e-10

    # do we get the same results as from newton?
    @test norm(Ψ_out_full[(2N+1):3N] - Ψ_out.state) < 1e-10
    @test norm(Ψ_out_full[1:N] - Ψ_out.grad_states[1]) < 1e-10
    @test norm(Ψ_out_full[(N+1):2N] - Ψ_out.grad_states[2]) < 1e-10

    ###########################################################################
    # Test standard expprop

    propagator = init_prop(
        Ψ,
        G,
        [0, dt];
        method = :expprop,
        inplace = true,
        convert_state = Vector{ComplexF64},
        convert_operator = Matrix{ComplexF64}
    )
    Ψ_out_exp = prop_step!(propagator)
    @test norm(Ψ_out_exp - Ψ_out) < 1e-11
    resetgradvec!(Ψ, Ψ₀)


    ###########################################################################
    # Compare against explicit split-up Grad-Gens
    # This checks whether the extension of the Gradient-Vector to multiple
    # controls is correct (the literature generally only gives the expression
    # for a single control)

    G_full1 = [
         Ĥ   Ĥ₁
        Zero Ĥ
    ]
    G_full2 = [
         Ĥ   Ĥ₂
        Zero Ĥ
    ]
    Ψ_full1 = [
        Ψ.grad_states[1]
        Ψ.state
    ]

    @test maximum(abs.(Ψ₀)) == ψ_max  # is Ψ₀ still exactly the same state?
    @test norm(Ψ_full1) == norm(Ψ₀)  # initialization correct?
    Ψ_full2 = vcat(Ψ.grad_states[2], Ψ.state)
    @test norm(Ψ_full2) == norm(Ψ₀)  # initialization correct?
    Ψ_out_full1 = exp(-𝕚 * G_full1 * dt) * Ψ_full1
    Ψ_out_full2 = exp(-𝕚 * G_full2 * dt) * Ψ_full2

    # propagation correct?
    @test norm(Ψ_out_full1[(N+1):2N] - Û_Ψ₀) < 1e-12
    @test norm(Ψ_out_full2[(N+1):2N] - Û_Ψ₀) < 1e-12

    # do we get the same results as with the combined grad-gen?
    @test norm(Ψ_out_full1[1:N] - Ψ_out_full[1:N]) < 1e-12
    @test norm(Ψ_out_full2[1:N] - Ψ_out_full[(N+1):2N]) < 1e-12
    @test norm(Ψ_out_full1[(N+1):2N] - Ψ_out_full[(2N+1):3N]) < 1e-12
    @test norm(Ψ_out_full2[(N+1):2N] - Ψ_out_full[(2N+1):3N]) < 1e-12

    ###########################################################################
    # Compare against Zygote
    # This checks whether the gradients are correct, albeit a bit indirectly
    # (since Zygote can only calculate the gradient for a scalar function, in
    # this case the square-modulus of the overlap with a target)

    function F_sm(ϵ₁, ϵ₂)
        Û = exp(-𝕚 * (Ĥ₀ + ϵ₁ * Ĥ₁ + ϵ₂ * Ĥ₂) * dt)
        Ψ_T = Û * Ψ₀
        return abs2(Ψtgt ⋅ Ψ_T)
    end
    grad_zygote = collect(gradient(F_sm, 1.0, 1.0))

    @test norm(Ψ_out.state - Û_Ψ₀) < 1e-12  # still correct?
    τ = Ψtgt ⋅ Û_Ψ₀
    # `grad` is gradient of F_sm based on Newton-prop of GradgenOperator
    # For ∂F/∂τ see Eq. (3.47) of Phd Thesis of Michael Goerz
    grad = [
        2 * real(conj(τ) * (Ψtgt ⋅ Ψ_out.grad_states[1])),
        2 * real(conj(τ) * (Ψtgt ⋅ Ψ_out.grad_states[2]))
    ]
    @test abs(grad_zygote[1] - grad[1]) < 1e-9
    @test abs(grad_zygote[2] - grad[2]) < 1e-9

    ###########################################################################
    # Backward-propagation of GradVector
    # This is a change of perspective from τ = ⟨Ψtgt | ⋅ (Û |Ψ₀⟩)
    # to τ = (⟨Ψtgt | Û) ⋅ |Ψ₀⟩) = ⟨χ|Ψ₀⟩ with |χ⟩ = exp[i Ĥ⁺ dt] |Ψtgt⟩. That
    # is, instead of forward-propagating the initial state, we
    # backward-propagate the target state

    χ = GradVector(Ψtgt, num_controls)
    wrk_bw = NewtonWrk(χ)
    Ĥ_adj_of_t = (copy(Ĥ₀'), (copy(Ĥ₁'), ϵ₁), (copy(Ĥ₂'), ϵ₂))
    G_adj_of_t = GradGenerator(Ĥ_adj_of_t)
    G_adj = evaluate(G_adj_of_t; vals_dict)
    newton!(χ, G_adj, -dt, wrk)
    χ_out = copy(χ)
    Ψtgt_Û = χ_out.state
    τ_bw = Ψtgt_Û ⋅ Ψ₀
    @test τ_bw ≈ τ
    grad_bw = [
        2 * real(conj(τ_bw) * (χ_out.grad_states[1] ⋅ Ψ₀)),
        2 * real(conj(τ_bw) * (χ_out.grad_states[2] ⋅ Ψ₀))
    ]
    @test grad_bw[1] ≈ grad[1]
    @test grad_bw[2] ≈ grad[2]

    ###########################################################################
    # Compare against Taylor series
    # This checks the gradients more directly against an alternative method

    commutator(A, B) = A * B - B * A

    """Evaluate ∂/∂ϵ exp(-𝕚 Ĥ dt) via a Taylor expansion."""
    function U_grad(Ĥ, μ̂, dt)
        # See Eq. (14) in de Fouquieres et. al, JMR 212, 412 (2011)
        Û = exp(-𝕚 * Ĥ * dt)
        converged = false
        Ĉ = μ̂
        terms = [(-𝕚 * dt) * Ĉ]
        n = 2
        while !converged
            Ĉ = commutator(Ĥ, Ĉ)
            term = -((𝕚 * dt)^n / factorial(big(n))) * Ĉ
            push!(terms, term)
            converged = (norm(term) < 1e-12)
            n += 1
        end
        return Û * sum(terms)
    end

    grad_taylor = [U_grad(Ĥ, Ĥ₁, dt) * Ψ₀, U_grad(Ĥ, Ĥ₂, dt) * Ψ₀]

    @test norm(Ψ_out.grad_states[1] - grad_taylor[1]) < 1e-10
    @test norm(Ψ_out.grad_states[2] - grad_taylor[2]) < 1e-10

end
