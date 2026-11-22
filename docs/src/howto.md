```@meta
# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
# SPDX-License-Identifier: MIT OR CC-BY-4.0
CurrentModule = QuantumGradientGenerators
```

# [How-to Guides](@id QGG-Howto)

**Contents**

```@contents
Pages = ["howto.md"]
Depth = 2:2
```


## [How to propagate a gradient vector backward in time](@id Howto-Backward)

A backward propagation, as in GRAPE, propagates a state ``|χ⟩`` from ``t = T`` to ``t = 0`` under the adjoint generator ``\hat{H}^\dagger(t)``. To get the derivative of each backward time step with respect to the control value on that time step, propagate a gradient vector under the gradient generator ``G[\hat{H}^\dagger]`` of the adjoint generator.

Start from the dynamical generator, here a non-Hermitian three-level system with a decaying upper level:

```@example backward
using LinearAlgebra
using QuantumPropagators: hamiltonian, init_prop, prop_step!, ExpProp
using QuantumGradientGenerators: GradGenerator, GradVector, resetgradvec!

Ĥ₀ = ComplexF64[0 0 0; 0 1.0 0; 0 0 (2.1-0.05im)]
Ĥ₁ = ComplexF64[0 1 0; 1 0 √2; 0 √2 0]
T = 5.0
tlist = collect(range(0, T; length = 51))
ϵ(t) = 0.2 + 0.1 * sin(2π * t / T)
H = hamiltonian(Ĥ₀, (Ĥ₁, ϵ))
nothing # hide
```

Construct the adjoint generator by taking the adjoint of each operator. The control is real, so the control function stays the same:

```@example backward
H_adj = hamiltonian(copy(Ĥ₀'), (copy(Ĥ₁'), ϵ))
nothing # hide
```

With [QuantumControl](@extref QuantumControl :doc:`index`), `adjoint(traj)` of a `Trajectory` `traj` returns a trajectory with the adjoint generator in the same way.

Build the gradient generator from the adjoint generator. Its control derivative is ``\hat{μ}^\dagger = \hat{H}_1^\dagger``:

```@example backward
G̃ = GradGenerator(H_adj)
nothing # hide
```

Initialize a backward propagator for a gradient vector that holds the boundary condition ``|χ(T)⟩`` as its state:

```@example backward
χ_T = ComplexF64[0, 0, 1]
propagator = init_prop(GradVector(χ_T, 1), G̃, tlist; method = ExpProp, backward = true)
nothing # hide
```

Propagate one time step at a time, resetting the gradient states with [`resetgradvec!`](@ref) before each step. After the step for the interval ``n``, the gradient state is the derivative of that backward step with respect to the control value ``ϵ_n`` on the interval, applied to ``|χ(t_n)⟩``:

```@example backward
N_T = length(tlist) - 1
χ_states = Vector{Vector{ComplexF64}}(undef, N_T + 1)  # |χ(tₙ)⟩ at tlist[n]
dχ = Vector{Vector{ComplexF64}}(undef, N_T)  # derivative of backward step n
χ_states[N_T+1] = χ_T
for n = N_T:-1:1
    resetgradvec!(propagator.state)  # keeps the state, zeroes the gradient
    χ = prop_step!(propagator)
    χ_states[n] = copy(χ.state)
    dχ[n] = copy(χ.grad_states[1])
end
```

This is how GRAPE evaluates its gradient with `gradient_method = :gradgen`. Without the call to `resetgradvec!`, the gradient state instead accumulates the derivative with respect to a parameter that all time steps depend on, see [Non-zero gradient states and multiple time steps](@ref Background-Exponential).

To check the result, compare with a finite-difference derivative of a single backward step, ``e^{-\ii \hat{H}^\dagger(ϵ_n) (-dt)}``, applied to ``|χ(t_{n+1})⟩``:

```@example backward
n = 20
ϵₙ = propagator.parameters[ϵ][n]
dt = tlist[n+1] - tlist[n]
Û_bw(e) = exp(1im * (Ĥ₀ + e * Ĥ₁)' * dt)
h = 1e-5
dχ_fd = (Û_bw(ϵₙ + h) - Û_bw(ϵₙ - h)) * χ_states[n+1] / 2h
norm(dχ[n] - dχ_fd)
```

!!! warning "Not the adjoint of the gradient generator"

    The gradient generator for the backward propagation is ``G[\hat{H}^\dagger]``, with ``\hat{μ}^\dagger`` in its upper right block. This is not the adjoint ``G[\hat{H}]^\dagger`` of the forward gradient generator, which is block *lower*-triangular. Propagating ``(0, |χ⟩)`` under ``G[\hat{H}]^\dagger`` leaves the gradient state at zero. For this reason, `QuantumGradientGenerators` does not define `adjoint` for a gradient generator. See [Properties of the gradient generator](@ref Background-Properties) for details.


## [How to propagate gradient vectors with the Chebychev propagator](@id Howto-Cheby)

The [Chebychev propagator](@extref QuantumPropagators :label:`method_cheby`) works for gradient vectors without any extra configuration. The gradient states it produces are the exact derivative of the Chebychev propagation of the state, even when that propagation is inaccurate: a polynomial in the gradient generator with coefficients that do not depend on the control maps a gradient vector to the derivative of the same polynomial applied to the state ([Corollary 7](@ref Background-Numerical-Methods)). This guide shows how to propagate a gradient vector and how to verify the gradient against finite differences.

Define a Hermitian generator. Here, the parameter ``u`` with respect to which we take the derivative is a static offset of the control, ``ϵ(t) + u``, so that ``\partial \hat{H} / \partial u = \hat{H}_1``:

```@example cheby
using LinearAlgebra
using QuantumPropagators: hamiltonian, propagate, Cheby, ExpProp
using QuantumGradientGenerators: GradGenerator, GradVector

Ĥ₀ = ComplexF64[0 0 0; 0 1.0 0; 0 0 2.1]
Ĥ₁ = ComplexF64[0 1 0; 1 0 √2; 0 √2 0]
T = 5.0
tlist = collect(range(0, T; length = 51))
ϵ(t) = 0.3 * sin(π * t / T)^2
Ψ₀ = ComplexF64[1, 0, 0]

H(u) = hamiltonian(Ĥ₀ + u * Ĥ₁, (Ĥ₁, ϵ))
G = GradGenerator(H(0.0))
nothing # hide
```

Propagate a gradient vector with `method = Cheby`:

```@example cheby
Ψ = propagate(GradVector(Ψ₀, 1), G, tlist; method = Cheby)
nothing # hide
```

The gradient generator has the same eigenvalues as ``\hat{H}``, each with multiplicity ``L+1`` ([Proposition 8](@ref Background-Properties)), so the propagator obtains its spectral range from ``\hat{H}``. It uses the same Chebychev coefficients as a propagation of the state alone, so the state component is identical:

```@example cheby
Ψ.state == propagate(Ψ₀, H(0.0), tlist; method = Cheby)
```

The gradient vector is propagated without resetting the gradient state, so `Ψ.grad_states[1]` is the derivative ``\partial_u |Ψ(T)⟩`` of the final state with respect to the offset ``u``.

Do not pass `check_normalization = true` when propagating a gradient vector. That check compares Rayleigh quotients with the spectral range, and those of a gradient generator can lie outside the spectral range of ``\hat{H}`` ([Non-normality](@ref Background-Properties)), so the check fails even when the spectral range is correct.

To verify the gradient with finite differences, the Chebychev propagations for ``u`` and ``u \pm h`` must use the same coefficients. By default, the spectral range, and thus the coefficients, are determined from the generator, and so they change with ``u``. Fix the spectral range with `specrange_method = :manual` and `E_min`, `E_max` (a range that contains the spectrum of ``\hat{H}`` for all values of ``u`` and of the control). To demonstrate that the gradient is the derivative of the Chebychev propagation even when the propagation is inaccurate, also truncate the Chebychev series early with `cheby_coeffs_limit = 1e-2`:

```@example cheby
cheby_kwargs = (;
    method = Cheby,
    specrange_method = :manual,
    E_min = -0.5,
    E_max = 2.5,
    cheby_coeffs_limit = 1e-2,
)
Ψ = propagate(GradVector(Ψ₀, 1), G, tlist; cheby_kwargs...)

h = 1e-6
dΨ_fd =
    (
        propagate(Ψ₀, H(h), tlist; cheby_kwargs...) -
        propagate(Ψ₀, H(-h), tlist; cheby_kwargs...)
    ) / 2h
norm(Ψ.grad_states[1] - dΨ_fd)
```

The gradient matches the finite-difference derivative of the same Chebychev propagation to the precision of the finite differences. For comparison, the exact derivative of the piecewise-constant time evolution follows from propagating the gradient vector with `ExpProp`:

```@example cheby
Ψ_exact = propagate(GradVector(Ψ₀, 1), G, tlist; method = ExpProp)
(
    state_error = norm(Ψ.state - Ψ_exact.state),
    gradient_error = norm(Ψ.grad_states[1] - Ψ_exact.grad_states[1]),
)
```

Both the state and the gradient deviate from the exact result at the level of the truncation error of the Chebychev series. With the default `cheby_coeffs_limit = 1e-12`, both are accurate to near machine precision.


## [How to control the time steps of an adaptive ODE solver for gradient vectors](@id Howto-ODE)

When a gradient vector is propagated with an adaptive ODE solver, the error estimate that determines the time steps can include the gradient states or ignore them. With the default error norm, it includes them (the *flat* semantics). With an error norm that looks only at the state, the solver takes exactly the time steps it would take for the state alone (the *frozen* semantics). See [Flat, frozen, and dual semantics](@ref Background-Semantics) for the theory. This guide shows how to select either behavior and what the choice does to the cost and the accuracy of the gradient.

The ODE solvers in [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl) require the state to be an array. A [`GradVector`](@ref) is not an array, so represent the gradient vector as a flat vector with the gradient states first and the state last, ``(|Ψ^\prime_1⟩, \dots, |Ψ^\prime_L⟩, |Ψ⟩)``. A [`GradgenOperator`](@ref) applies to such a vector block by block.

```@example ode
using LinearAlgebra
using Printf
using OrdinaryDiffEq
using DiffEqBase: ODE_DEFAULT_NORM
using QuantumPropagators: hamiltonian, propagate, init_prop, prop_step!
using QuantumGradientGenerators: GradGenerator

Ĥ₀ = ComplexF64[0 0 0; 0 1.0 0; 0 0 2.1]
Ĥ₁ = ComplexF64[0 1 0; 1 0 √2; 0 √2 0]
T = 5.0
tlist = [0.0, T]
ϵ(t) = 0.3 * sin(π * t / T)^2
Ψ₀ = ComplexF64[1, 0, 0]

H(u) = hamiltonian(Ĥ₀ + u * Ĥ₁, (Ĥ₁, ϵ))
G = GradGenerator(H(0.0))

N = length(Ψ₀)  # dimension of the Hilbert space
L = 1  # number of gradient states
Ψ_flat = [zeros(ComplexF64, L * N); Ψ₀]
state_idxs = (L*N+1):((L+1)*N)
grad_idxs = 1:N  # first gradient state
nothing # hide
```

As in the [Chebychev guide](@ref Howto-Cheby), the derivative is with respect to a static offset ``u`` of the control.

For the flat semantics, propagate with the default error norm. The flat vector does not pass the interface check that [`propagate`](@extref QuantumPropagators.propagate) runs by default (`GradgenOperator` implements `mul!` for a flat vector, but not `*`), so disable the check with `check = false`:

```@example ode
Ψ = propagate(
    Ψ_flat,
    G,
    tlist;
    method = OrdinaryDiffEq,
    check = false,
    reltol = 1e-6,
    abstol = 1e-6
)
nothing # hide
```

For the frozen semantics, pass an `internalnorm` that evaluates the default norm on the state component only. OrdinaryDiffEq also calls the norm on individual numbers, which requires a second method:

```@example ode
state_norm(u::AbstractVector, t) = ODE_DEFAULT_NORM(view(u, state_idxs), t)
state_norm(u::Number, t) = ODE_DEFAULT_NORM(u, t)

Ψ = propagate(
    Ψ_flat,
    G,
    tlist;
    method = OrdinaryDiffEq,
    check = false,
    reltol = 1e-6,
    abstol = 1e-6,
    internalnorm = state_norm
)
nothing # hide
```

In either case, the state is `Ψ[state_idxs]` and the gradient state is `Ψ[grad_idxs]`.

To decide between the two, compare them over a range of tolerances. The following propagates the state alone, and the gradient vector in both modes, and reports the number of time steps (from the underlying OrdinaryDiffEq integrator), the deviation of the state from the propagation of the state alone, and the error of the gradient relative to a reference propagation with a tight tolerance:

```@example ode
function propagate_ode(Ψ, generator; kwargs...)
    propagator = init_prop(Ψ, generator, tlist; method = OrdinaryDiffEq, kwargs...)
    prop_step!(propagator)
    return copy(propagator.state), propagator.integrator.stats.naccept
end

Ψ_ref, _ = propagate_ode(Ψ_flat, G; alg = Vern9(), reltol = 1e-14, abstol = 1e-14)

@printf("%7s | %5s | %-27s | %-27s\n", "", "bare", "flat", "frozen")
@printf(
    "%7s | %5s | %5s %10s %10s | %5s %10s %10s\n",
    "tol",
    "steps",
    "steps",
    "Δ state",
    "grad err",
    "steps",
    "Δ state",
    "grad err"
)
for tol in (1e-4, 1e-6, 1e-8, 1e-10)
    kw = (; reltol = tol, abstol = tol)
    Ψ_bare, n_bare = propagate_ode(Ψ₀, H(0.0); kw...)
    Ψ_flat_T, n_flat = propagate_ode(Ψ_flat, G; kw...)
    Ψ_frozen_T, n_frozen = propagate_ode(Ψ_flat, G; internalnorm = state_norm, kw...)
    @printf(
        "%7.0e | %5d | %5d %10.2e %10.2e | %5d %10.2e %10.2e\n",
        tol,
        n_bare,
        n_flat,
        norm(Ψ_flat_T[state_idxs] - Ψ_bare),
        norm(Ψ_flat_T[grad_idxs] - Ψ_ref[grad_idxs]),
        n_frozen,
        norm(Ψ_frozen_T[state_idxs] - Ψ_bare),
        norm(Ψ_frozen_T[grad_idxs] - Ψ_ref[grad_idxs]),
    )
end
```

With the frozen norm, the solver takes the same time steps as for the state alone, and the state component is identical to the propagation of the state without gradients. The gradient is the derivative of the computed state, except for the dependence of the time steps on ``u``, and its error is of the order of the propagation error. With the default (flat) norm, the gradient states contribute to the error estimate. The solver then takes more time steps, the state deviates from the propagation of the state alone within the tolerance, and the gradient is more accurate at the same tolerance.

Use the frozen norm when the state must agree exactly with a propagation without gradients, e.g., when a functional is evaluated from separately propagated states and its gradient from gradient vectors, or when the cost of each time step matters more than the accuracy of the gradient at a given tolerance. Use the default norm when the accuracy of the gradient should be controlled directly by the tolerance. The default norm of DifferentialEquations for the dual numbers of [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) also includes the dual parts, so forward-mode automatic differentiation through the solver follows the flat semantics as well.
