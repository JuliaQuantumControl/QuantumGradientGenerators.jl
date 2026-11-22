# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test
using LinearAlgebra: norm, mul!, lmul!
using QuantumPropagators: hamiltonian, propagate, init_prop, prop_step!, ExpProp, evaluate
using QuantumGradientGenerators:
    GradGenerator, GradVector, frozen_internalnorm, flat_internalnorm
import OrdinaryDiffEq
# Only algorithms that OrdinaryDiffEq 6 and 7 both export
using OrdinaryDiffEq: ODEProblem, solve, Tsit5, Vern7, Vern9
import DiffEqBase
using ForwardDiff: Dual, Tag, value, partials


const Ĥ₀ = ComplexF64[0 0 0; 0 1.0 0; 0 0 2.1]
const Ĥ₁ = ComplexF64[0 1 0; 1 0 √2; 0 √2 0]
const T = 5.0
ϵ(t) = 0.3 * sin(π * t / T)^2
const Ψ₀ = ComplexF64[1, 0, 0]
const tlist = [0.0, T]

const H = hamiltonian(Ĥ₀, (Ĥ₁, ϵ))
const G = GradGenerator(H)

# The exact derivative of the propagated state with respect to a static
# offset of the control, from the block exponential on a fine time grid
const Ψ_exact =
    propagate(GradVector(Ψ₀, 1), G, collect(range(0, T; length = 2001)); method = ExpProp)


# Propagate with OrdinaryDiffEq via QuantumPropagators; return the result and
# the number of accepted time steps
function propagate_ode(Ψ, generator; kwargs...)
    propagator = init_prop(Ψ, generator, tlist; method = OrdinaryDiffEq, kwargs...)
    prop_step!(propagator)
    return copy(propagator.state), propagator.integrator.stats.naccept
end


# Propagate a gradient vector directly with OrdinaryDiffEq; return the result
# and the time grid
function propagate_gradvec(alg; kwargs...)
    function f!(du, u, p, t)
        mul!(du, evaluate(G, t), u)
        lmul!(-1im, du)
    end
    sol = solve(ODEProblem(f!, GradVector(Ψ₀, 1), (0.0, T)), alg; kwargs...)
    return sol.u[end], sol.t
end


# Propagate a state of ForwardDiff dual numbers for the Hamiltonian at the
# control ϵ(t) + δ, directly with OrdinaryDiffEq. Return the value and the
# derivative with respect to δ, and the number of accepted time steps.
function propagate_dual(alg; kwargs...)
    # A proper tag (not `Nothing`), so that the dual numbers can be ordered
    # with the tags of the automatic differentiation in DiffEqBase
    δ = Dual{typeof(Tag(propagate_dual, Float64))}(0.0, 1.0)
    f!(du, u, p, t) = (du .= -1im .* ((Ĥ₀ .+ (ϵ(t) + p) .* Ĥ₁) * u))
    u₀ = Complex{typeof(δ)}.(Ψ₀)
    sol = solve(ODEProblem(f!, u₀, (0.0, T), δ), alg; save_everystep = false, kwargs...)
    u = sol.u[end]
    Ψ = value.(real.(u)) .+ 1im .* value.(imag.(u))
    ∂Ψ = partials.(real.(u), 1) .+ 1im .* partials.(imag.(u), 1)
    return Ψ, ∂Ψ, sol.stats.naccept
end


@testset "Blockwise broadcasting" begin

    Ψ = GradVector(ComplexF64[1, 2, 3], 2)
    Ψ.grad_states[1] .= ComplexF64[4, 5, 6]
    Ψ.grad_states[2] .= ComplexF64[7, 8, 9]
    Φ = GradVector(ComplexF64[1im, 0, 0], 2)
    Φ.grad_states[2] .= 1.0

    Θ = Ψ .+ 2 .* Φ
    @test Θ isa GradVector{2,Vector{ComplexF64}}
    @test Θ.state == Ψ.state .+ 2 .* Φ.state
    @test Θ.grad_states[1] == Ψ.grad_states[1] .+ 2 .* Φ.grad_states[1]
    @test Θ.grad_states[2] == Ψ.grad_states[2] .+ 2 .* Φ.grad_states[2]

    Θ = similar(Ψ)
    Θ .= Ψ .* Φ .- 1im
    @test Θ.state == Ψ.state .* Φ.state .- 1im
    @test Θ.grad_states[2] == Ψ.grad_states[2] .* Φ.grad_states[2] .- 1im

    Θ .= 0
    @test norm(Θ) == 0.0

    A = abs2.(Ψ)
    @test A isa GradVector{2,Vector{Float64}}
    @test A.grad_states[1] == [16.0, 25.0, 36.0]

    # Together with an array, a gradient vector is a flat vector with the
    # gradient states first and the state last (the layout of `getindex`)
    flat = [Ψ.grad_states[1]; Ψ.grad_states[2]; Ψ.state]
    v = collect(ComplexF64, 1:9)
    @test (Ψ .+ v) isa Vector{ComplexF64}
    @test (Ψ .+ v) == flat .+ v
    w = zeros(ComplexF64, 9)
    w .= Ψ ./ 2
    @test w == flat ./ 2
    Θ = similar(Ψ)
    Θ .= v .* Ψ
    @test Θ.grad_states[2] == v[4:6] .* Ψ.grad_states[2]
    @test Θ.state == v[7:9] .* Ψ.state
    @test_throws DimensionMismatch (Θ .= ones(ComplexF64, 4))

    # Buffers for the error estimate of ODE solvers keep the block structure
    R = similar(Ψ, Float64)
    @test R isa GradVector{2,Vector{Float64}}
    @test length(R.state) == 3

end


@testset "ODE error norms" begin

    Ψ = GradVector(ComplexF64[3, 4, 0, 0], 1)
    Ψ.grad_states[1] .= ComplexF64[0, 0, 12, 0]
    # root mean square of the state: √(25/4) = 2.5
    @test frozen_internalnorm(Ψ, 0.0) ≈ 2.5
    @test DiffEqBase.ODE_DEFAULT_NORM(Ψ, 0.0) ≈ 2.5
    # root mean square of all 8 elements: √(169/8)
    @test flat_internalnorm(Ψ, 0.0) ≈ √(169 / 8)
    flat_vector = [Ψ.grad_states[1]; Ψ.state]
    @test flat_internalnorm(Ψ, 0.0) ≈ DiffEqBase.ODE_DEFAULT_NORM(flat_vector, 0.0)
    @test frozen_internalnorm(-2.0 + 0im, 0.0) == 2.0
    @test flat_internalnorm(-2.0, 0.0) == 2.0

end


@testset "OrdinaryDiffEq: frozen semantics by default" begin

    tol = (; reltol = 1e-6, abstol = 1e-6)
    Ψ_bare, n_bare = propagate_ode(Ψ₀, H; tol...)
    Ψ, n = propagate_ode(GradVector(Ψ₀, 1), G; tol...)
    Ψ_frozen, n_frozen =
        propagate_ode(GradVector(Ψ₀, 1), G; tol..., internalnorm = frozen_internalnorm)

    # The time steps, and therefore the state, are those of the propagation
    # without gradients
    @test n == n_bare
    @test norm(Ψ.state - Ψ_bare) < 1e-14
    # The default is `frozen_internalnorm`
    @test n_frozen == n
    @test norm(Ψ_frozen.state - Ψ.state) == 0.0
    @test norm(Ψ_frozen.grad_states[1] - Ψ.grad_states[1]) == 0.0

    # The gradient approximates the exact derivative within the tolerance
    @test norm(Ψ.grad_states[1] - Ψ_exact.grad_states[1]) < 1e-4

    # `propagate` passes its interface check for the gradient vector
    Ψ = propagate(GradVector(Ψ₀, 1), G, tlist; method = OrdinaryDiffEq, tol...)
    @test norm(Ψ.state - Ψ_bare) < 1e-14

end


@testset "OrdinaryDiffEq: default norm of complex dual numbers" begin

    # For a complex state of ForwardDiff dual numbers, the default error norm
    # of OrdinaryDiffEq is a dual number whose value is the norm of the values
    # only, so the time steps do not depend on the dual parts
    TT = typeof(Tag(propagate_dual, Float64))
    vals = ComplexF64[1+2im, -0.5im, 3.0, 0.1-0.2im]
    duals = ComplexF64[10im, 5.0, -3+1im, 2.0]
    u = [
        Complex(Dual{TT}(real(v), real(d)), Dual{TT}(imag(v), imag(d))) for
        (v, d) in zip(vals, duals)
    ]
    nrm = DiffEqBase.ODE_DEFAULT_NORM(u, 0.0)
    @test value(nrm) ≈ DiffEqBase.ODE_DEFAULT_NORM(vals, 0.0)
    @test !(value(nrm) ≈ DiffEqBase.ODE_DEFAULT_NORM([vals; duals], 0.0))

end


@testset "OrdinaryDiffEq: frozen semantics = ForwardDiff ($(nameof(typeof(alg))), tol=$tol)" for alg in
                                                                                                 (
        Tsit5(),
        Vern7(),
        Vern9()
    ),
    tol in (1e-4, 1e-8)

    # On the time grid of the gradient vector in the frozen semantics, the
    # gradient is the derivative of the computed state (ForwardDiff)
    Ψ, ts = propagate_gradvec(alg; reltol = tol, abstol = tol)
    Ψ_dual, ∂Ψ_dual, _ = propagate_dual(alg; adaptive = false, dt = T, tstops = ts)
    @test norm(Ψ.state - Ψ_dual) < 1e-13
    @test norm(Ψ.grad_states[1] - ∂Ψ_dual) < 1e-13

    # With adaptive time steps, the dual numbers take the same time steps, but
    # the default norm of OrdinaryDiffEq sums with `@fastmath`, so its
    # rounding errors differ between `ComplexF64` and dual numbers, and
    # between CPUs. This can flip the acceptance of a time step at the
    # threshold of the tolerance, after which the time steps and the results
    # differ by much less than the tolerance.
    Ψ_dual, ∂Ψ_dual, n_dual = propagate_dual(alg; reltol = tol, abstol = tol)
    @test abs((length(ts) - 1) - n_dual) <= 1
    @test norm(Ψ.state - Ψ_dual) < tol
    @test norm(Ψ.grad_states[1] - ∂Ψ_dual) < tol

end


@testset "OrdinaryDiffEq: flat semantics" begin

    tol = (; reltol = 1e-6, abstol = 1e-6)
    Ψ_bare, n_bare = propagate_ode(Ψ₀, H; tol...)
    Ψ, n = propagate_ode(GradVector(Ψ₀, 1), G; tol..., internalnorm = flat_internalnorm)

    # Same as the default norm for a flat vector of all elements, up to
    # rounding errors in the error estimate (`flat_internalnorm` sums block by
    # block), which can flip the acceptance of a time step at the threshold
    Ψ_flat = [zeros(ComplexF64, 3); Ψ₀]
    Ψ_flat, n_flat = propagate_ode(Ψ_flat, G; tol...)
    @test abs(n - n_flat) <= 1
    @test norm(Ψ.state - Ψ_flat[4:6]) < 1e-6
    @test norm(Ψ.grad_states[1] - Ψ_flat[1:3]) < 1e-6

    # The time steps take into account the gradient state, which changes the
    # state within the tolerance
    @test n > n_bare
    @test 0 < norm(Ψ.state - Ψ_bare) < 1e-5
    @test norm(Ψ.grad_states[1] - Ψ_exact.grad_states[1]) < 1e-4

end


@testset "OrdinaryDiffEq: fixed time steps" begin

    # With fixed time steps, the frozen and the flat semantics coincide, and
    # the gradient is the derivative of the propagated state (ForwardDiff)
    kw = (; alg = Tsit5(), adaptive = false, dt = 0.05)
    Ψ_dual, ∂Ψ_dual, _ = propagate_dual(kw.alg; adaptive = false, dt = 0.05)
    Ψ_frozen, _ = propagate_ode(GradVector(Ψ₀, 1), G; kw...)
    Ψ_flat, _ = propagate_ode(GradVector(Ψ₀, 1), G; kw..., internalnorm = flat_internalnorm)
    @test norm(Ψ_frozen.grad_states[1] - ∂Ψ_dual) < 1e-13
    @test norm(Ψ_flat.grad_states[1] - ∂Ψ_dual) < 1e-13
    @test norm(Ψ_frozen.state - Ψ_dual) < 1e-13

end
