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
    Ψ = random_state_vector(N)
    ψ_max = maximum(abs.(Ψ))
    Ψtgt = random_state_vector(N)
    𝕚 = 1im
    dt = 1.25

    vals_dict = IdDict(ϵ₁ => 1.0, ϵ₂ => 1.0)

    G̃_of_t = GradGenerator(Ĥ_of_t)
    G̃ = evaluate(G̃_of_t; vals_dict)
    Ĥ = evaluate(Ĥ_of_t; vals_dict)

    Û_Ψ = exp(-𝕚 * Ĥ * dt) * Ψ

    num_controls = length(Ĥ_of_t) - 1
    Ψ̃ = GradVector(Ψ, num_controls)
    # did the initialization work?
    @test norm(Ψ̃.state - Ψ) < 1e-14
    @test norm(Ψ̃.grad_states[1]) == 0.0
    @test norm(Ψ̃.grad_states[2]) == 0.0
    @test length(Ψ̃.grad_states) == 2

    @test maximum(abs.(Ψ)) == ψ_max  # is Ψ still exactly the same state?
    wrk = NewtonWrk(Ψ̃)
    newton!(Ψ̃, G̃, dt, wrk)
    Ψ̃_out = copy(Ψ̃)
    @test maximum(abs.(Ψ)) == ψ_max  # is Ψ still exactly the same state?
    @test norm(Ψ̃_out.state - Û_Ψ) < 1e-12  # Ψ̃_out.state correct?
    @test norm(Ψ̃_out.grad_states[1]) > 0
    @test norm(Ψ̃_out.grad_states[2]) > 0
    # did the copy work?
    @test norm(Ψ̃_out.state - Ψ̃.state) == 0.0
    @test norm(Ψ̃_out.grad_states[1] - Ψ̃.grad_states[1]) == 0.0
    @test norm(Ψ̃_out.grad_states[2] - Ψ̃.grad_states[2]) == 0.0

    # reset Ψ̃ to the original input (also to test the resetgradvec! method)
    resetgradvec!(Ψ̃, Ψ)
    # state should be the same as after previous initialization
    @test norm(Ψ̃.state - Ψ) < 1e-14
    @test norm(Ψ̃.grad_states[1]) == 0.0
    @test norm(Ψ̃.grad_states[2]) == 0.0

    @test norm(Ψ̃_out.state - Û_Ψ) < 1e-12  # Ψ̃_out.state still correct?

    ###########################################################################
    # Compare against explicit Grad-Gen
    # This checks whether the application of a GradgenOperator to a GradVector an
    # all the linear-algebra methods are implemented correctly

    G̃_full = [
         Ĥ    Zero  Ĥ₁
        Zero   Ĥ    Ĥ₂
        Zero  Zero  Ĥ
    ]

    Ψ̃_full = [
        Ψ̃.grad_states[1]
        Ψ̃.grad_states[2]
        Ψ̃.state
    ]

    @test size(G̃) == size(G̃_full)
    @test eltype(G̃) == eltype(G̃_full)

    # proper initialization? grad_states should be zero
    @test norm(Ψ̃_full) == norm(Ψ̃.state) == norm(Ψ)

    Ψ̃_out_full = exp(-𝕚 * G̃_full * dt) * Ψ̃_full
    # propagation correct?
    @test norm(Ψ̃_out_full[2N+1:3N] - Û_Ψ) < 1e-10

    # do we get the same results as from newton?
    @test norm(Ψ̃_out_full[2N+1:3N] - Ψ̃_out.state) < 1e-10
    @test norm(Ψ̃_out_full[1:N] - Ψ̃_out.grad_states[1]) < 1e-10
    @test norm(Ψ̃_out_full[N+1:2N] - Ψ̃_out.grad_states[2]) < 1e-10

    ###########################################################################
    # Test custom expprop


    Ψ̃_full = [
        Ψ̃.grad_states[1]
        Ψ̃.grad_states[2]
        Ψ̃.state
    ]

    # proper initialization? grad_states should be zero
    @test norm(Ψ̃_full) == norm(Ψ̃.state) == norm(Ψ)

    Ψ̃_out_full = exp(-𝕚 * G̃_full * dt) * Ψ̃_full
    # propagation correct?
    @test norm(Ψ̃_out_full[2N+1:3N] - Û_Ψ) < 1e-10

    # do we get the same results as from newton?
    @test norm(Ψ̃_out_full[2N+1:3N] - Ψ̃_out.state) < 1e-10
    @test norm(Ψ̃_out_full[1:N] - Ψ̃_out.grad_states[1]) < 1e-10
    @test norm(Ψ̃_out_full[N+1:2N] - Ψ̃_out.grad_states[2]) < 1e-10

    ###########################################################################
    # Test standard expprop

    propagator = init_prop(
        Ψ̃,
        G̃,
        [0, dt];
        method=:expprop,
        inplace=true,
        convert_state=Vector{ComplexF64},
        convert_operator=Matrix{ComplexF64}
    )
    Ψ̃_out_exp = prop_step!(propagator)
    @test norm(Ψ̃_out_exp - Ψ̃_out) < 1e-11
    resetgradvec!(Ψ̃, Ψ)


    ###########################################################################
    # Compare against explicit split-up Grad-Gens
    # This checks whether the extension of the Gradient-Vector to multiple
    # controls is correct (the literature generally only gives the expression
    # for a single control)

    G̃_full1 = [
         Ĥ   Ĥ₁
        Zero Ĥ
    ]
    G̃_full2 = [
         Ĥ   Ĥ₂
        Zero Ĥ
    ]
    Ψ̃_full1 = [
        Ψ̃.grad_states[1]
        Ψ̃.state
    ]

    @test maximum(abs.(Ψ)) == ψ_max  # is Ψ still exactly the same state?
    @test norm(Ψ̃_full1) == norm(Ψ)  # initialization correct?
    Ψ̃_full2 = vcat(Ψ̃.grad_states[2], Ψ̃.state)
    @test norm(Ψ̃_full2) == norm(Ψ)  # initialization correct?
    Ψ̃_out_full1 = exp(-𝕚 * G̃_full1 * dt) * Ψ̃_full1
    Ψ̃_out_full2 = exp(-𝕚 * G̃_full2 * dt) * Ψ̃_full2

    # propagation correct?
    @test norm(Ψ̃_out_full1[N+1:2N] - Û_Ψ) < 1e-12
    @test norm(Ψ̃_out_full2[N+1:2N] - Û_Ψ) < 1e-12

    # do we get the same results as with the combined grad-gen?
    @test norm(Ψ̃_out_full1[1:N] - Ψ̃_out_full[1:N]) < 1e-12
    @test norm(Ψ̃_out_full2[1:N] - Ψ̃_out_full[N+1:2N]) < 1e-12
    @test norm(Ψ̃_out_full1[N+1:2N] - Ψ̃_out_full[2N+1:3N]) < 1e-12
    @test norm(Ψ̃_out_full2[N+1:2N] - Ψ̃_out_full[2N+1:3N]) < 1e-12

    ###########################################################################
    # Compare against Zygote
    # This checks whether the gradients are correct, albeit a bit indirectly
    # (since Zygote can only calculate the gradient for a scalar function, in
    # this case the square-modulus of the overlap with a target)

    function F_sm(ϵ₁, ϵ₂)
        Û = exp(-𝕚 * (Ĥ₀ + ϵ₁ * Ĥ₁ + ϵ₂ * Ĥ₂) * dt)
        Ψ_T = Û * Ψ
        return abs2(Ψtgt ⋅ Ψ_T)
    end
    grad_zygote = collect(gradient(F_sm, 1.0, 1.0))

    @test norm(Ψ̃_out.state - Û_Ψ) < 1e-12  # still correct?
    τ = Ψtgt ⋅ Û_Ψ
    # `grad` is gradient of F_sm based on Newton-prop of GradgenOperator
    # For ∂F/∂τ see Eq. (3.47) of Phd Thesis of Michael Goerz
    grad = [
        2 * real(conj(τ) * (Ψtgt ⋅ Ψ̃_out.grad_states[1])),
        2 * real(conj(τ) * (Ψtgt ⋅ Ψ̃_out.grad_states[2]))
    ]
    @test abs(grad_zygote[1] - grad[1]) < 1e-9
    @test abs(grad_zygote[2] - grad[2]) < 1e-9

    ###########################################################################
    # Backward-propagation of GradVector
    # This is a change of perspective from τ = ⟨Ψtgt | ⋅ (Û |Ψ⟩)
    # to τ = (⟨Ψtgt | Û) ⋅ |Ψ⟩) = ⟨χ|Ψ⟩ with |χ⟩ = exp[i Ĥ⁺ dt] |Ψtgt⟩. That
    # is, instead of forward-propagating the initial state, we
    # backward-propagate the target state

    χ̃ = GradVector(Ψtgt, num_controls)
    wrk_bw = NewtonWrk(χ̃)
    Ĥ_adj_of_t = (copy(Ĥ₀'), (copy(Ĥ₁'), ϵ₁), (copy(Ĥ₂'), ϵ₂))
    G̃_adj_of_t = GradGenerator(Ĥ_adj_of_t)
    G̃_adj = evaluate(G̃_adj_of_t; vals_dict)
    newton!(χ̃, G̃_adj, -dt, wrk)
    χ̃_out = copy(χ̃)
    Ψtgt_Û = χ̃_out.state
    τ_bw = Ψtgt_Û ⋅ Ψ
    @test τ_bw ≈ τ
    grad_bw = [
        2 * real(conj(τ_bw) * (χ̃_out.grad_states[1] ⋅ Ψ)),
        2 * real(conj(τ_bw) * (χ̃_out.grad_states[2] ⋅ Ψ))
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

    grad_taylor = [U_grad(Ĥ, Ĥ₁, dt) * Ψ, U_grad(Ĥ, Ĥ₂, dt) * Ψ]

    @test norm(Ψ̃_out.grad_states[1] - grad_taylor[1]) < 1e-10
    @test norm(Ψ̃_out.grad_states[2] - grad_taylor[2]) < 1e-10

end
