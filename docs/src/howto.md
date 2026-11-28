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
G = GradGenerator(H_adj)
nothing # hide
```

Initialize a backward propagator for a gradient vector that holds the boundary condition ``|χ(T)⟩`` as its state:

```@example backward
χ_T = ComplexF64[0, 0, 1]
propagator = init_prop(GradVector(χ_T, 1), G, tlist; method = ExpProp, backward = true)
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

This is how GRAPE evaluates its gradient with `gradient_method = :gradgen`. Without the call to `resetgradvec!`, the gradient state instead accumulates the derivative with respect to a parameter that all time steps depend on, see [Non-zero gradient states and multiple time steps](@ref Background-cor-gradient-state).

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

    The gradient generator for the backward propagation is ``G[\hat{H}^\dagger]``, with ``\hat{μ}^\dagger`` in its upper right block. This is not the adjoint ``G[\hat{H}]^\dagger`` of the forward gradient generator, which is block *lower*-triangular. Propagating ``(0, |χ⟩)`` under ``G[\hat{H}]^\dagger`` leaves the gradient state at zero. For this reason, `QuantumGradientGenerators` does not define `adjoint` for a gradient generator. See [Adjoint](@ref Background-Adjoint) on the Background page for details.


## [How to compute gradients with respect to control parameters](@id Howto-Parameters)

The controls of a dynamical generator may depend on [control parameters](@extref QuantumControl :label:`Control-Parameters`) ``u_1, \dots, u_M``, e.g., the coefficients of a CRAB expansion, or a tunable frequency of the drift. Propagating a gradient vector under the gradient generator for the parameters over the entire time interval, without resetting the gradient states, yields the derivatives ``\partial_{u_n} |Ψ(T)⟩`` of the final state with respect to all parameters at once. This is the basis of GOAT [MachnesPRL2018](@cite), see [Parameterized controls](@ref Background-Parameters) on the Background page.

Define a generator with two kinds of parameters: the coefficients of a [`CRABFunction`](@extref QuantumPropagators.ParameterizedFunctions.CRABFunction) as the control, and the energy of the upper level as a [`StaticParameter`](@extref QuantumPropagators.Controls.StaticParameter):

```@example parameters
using LinearAlgebra
using QuantumPropagators: hamiltonian, propagate, ExpProp
using QuantumPropagators.Controls: get_parameters, combine_parameters, StaticParameter
using QuantumPropagators.ParameterizedFunctions: CRABFunction
using QuantumGradientGenerators: GradGenerator, GradVector

Ĥ₀ = ComplexF64[0 0 0; 0 1.0 0; 0 0 0]
Ĥ₁ = ComplexF64[0 1 0; 1 0 √2; 0 √2 0]
P̂₂ = ComplexF64[0 0 0; 0 0 0; 0 0 1]  # projector onto the upper level
T = 5.0
tlist = collect(range(0, T; length = 201))
shape(t) = 0.3 * sin(π * t / T)^2
# coefficients c⁽⁺⁾₁, c⁽⁺⁾₂, c⁽⁻⁾₁, c⁽⁻⁾₂
crab = CRABFunction(2; frequencies = [0.6, 1.3], shape, parameters = [0.5, -0.3, 0.2, 0.4])
w = [2.1]  # energy of the upper level
H = hamiltonian(Ĥ₀, (P̂₂, StaticParameter(w, 1)), (Ĥ₁, crab))
u = get_parameters(H)
```

The parameters `u` are a `RecursiveArrayTools.ArrayPartition` of the [parameter blocks](@extref QuantumPropagators :label:`ParameterBlocks`) `w` and `crab.parameters`. They are views into the blocks: changing `u[n]` changes the parameters of `H`. The gradient generator for these parameters is

```@example parameters
G = GradGenerator(H, u)
nothing # hide
```

with the derivative ``\hat{μ}_n = \partial \hat{H} / \partial u_n`` in `G.derivs[n]`, as obtained from [`get_parameter_derivs`](@extref QuantumControl.Controls.get_parameter_derivs). The first derivative, with respect to the energy `w[1]`, is the static operator ``\hat{P}_2``. The derivatives with respect to the CRAB coefficients are time-dependent generators:

```@example parameters
G.derivs[1] == P̂₂
```

```@example parameters
G.derivs[3]
```

The parameters are available as `G.wrt`, which is `u` itself:

```@example parameters
G.wrt === u
```

Propagate a gradient vector with one gradient state per parameter:

```@example parameters
Ψ₀ = ComplexF64[1, 0, 0]
Ψ = propagate(GradVector(Ψ₀, length(u)), G, tlist; method = ExpProp)
nothing # hide
```

The gradient state `Ψ.grad_states[n]` is the derivative ``\partial_{u_n} |Ψ(T)⟩`` of the final state. For the functional ``J_T = 1 - |τ|^2`` with the overlap ``τ = ⟨Ψ_{\text{tgt}}|Ψ(T)⟩``, the gradient is ``\partial J_T / \partial u_n = -2 \Re (τ^* ⟨Ψ_{\text{tgt}}|Ψ^\prime_n(T)⟩)``:

```@example parameters
Ψ_tgt = ComplexF64[0, 0, 1]
τ = dot(Ψ_tgt, Ψ.state)
∇J = [-2 * real(conj(τ) * dot(Ψ_tgt, Ψ′)) for Ψ′ in Ψ.grad_states]
```

To check the gradient, compare it with central finite differences. Changing an element of `u` changes the parameters of `H`:

```@example parameters
function J_T(H)
    τ = dot(Ψ_tgt, propagate(Ψ₀, H, tlist; method = ExpProp))
    return 1 - abs2(τ)
end

function fd_gradient(H, u; h = 1e-6)
    ∇J_fd = zeros(length(u))
    for n in eachindex(u)
        u₀ = u[n]
        u[n] = u₀ + h
        J_plus = J_T(H)
        u[n] = u₀ - h
        J_minus = J_T(H)
        u[n] = u₀
        ∇J_fd[n] = (J_plus - J_minus) / 2h
    end
    return ∇J_fd
end

norm(∇J - fd_gradient(H, u))
```

The gradient generator reads the values of the parameters when it is evaluated, so it remains valid when an optimizer changes the parameters in place. There is no need to construct a new `GradGenerator` in each iteration:

```@example parameters
u .= u .+ 0.1  # e.g., an update by an optimizer
Ψ = propagate(GradVector(Ψ₀, length(u)), G, tlist; method = ExpProp)
τ = dot(Ψ_tgt, Ψ.state)
∇J = [-2 * real(conj(τ) * dot(Ψ_tgt, Ψ′)) for Ψ′ in Ψ.grad_states]
norm(∇J - fd_gradient(H, u))
```

The piecewise-constant propagation with `ExpProp` gives the exact derivative of the piecewise-constant approximation of the dynamics. For the time-continuous dynamics, propagate with an ODE solver over the entire time interval instead. See the [how-to guide for ODE solvers](@ref Howto-ODE) for the choice of the error norm:

```@example parameters
using OrdinaryDiffEq: OrdinaryDiffEq, Vern9

Ψ = propagate(
    GradVector(Ψ₀, length(u)),
    G,
    [0.0, T];
    method = OrdinaryDiffEq,
    alg = Vern9(),
    reltol = 1e-10,
    abstol = 1e-10
)
τ = dot(Ψ_tgt, Ψ.state)
∇J_ode = [-2 * real(conj(τ) * dot(Ψ_tgt, Ψ′)) for Ψ′ in Ψ.grad_states]
norm(∇J_ode - ∇J)
```

This gradient differs from the gradient `∇J` of the piecewise-constant propagation by the time discretization error of `tlist`.

### Parameters of a control problem

In a control problem with several trajectories, the gradient must be with respect to the parameters of the problem, while each trajectory may depend only on some of them. Pass the parameters of the problem to `GradGenerator` for the generator of each trajectory. With [QuantumControl](@extref QuantumControl :doc:`index`), these are `get_parameters(problem)`. Here, we combine the parameters of `H` with those of a second generator `H2`, which has its own CRAB control and the same upper-level energy `w`:

```@example parameters
crab2 = CRABFunction(1; frequencies = [0.9], shape, parameters = [0.3, -0.2])
H2 = hamiltonian(Ĥ₀, (P̂₂, StaticParameter(w, 1)), (Ĥ₁, crab2))
u_problem = combine_parameters(u, get_parameters(H2))
G2 = GradGenerator(H2, u_problem)
[isnothing(μ) for μ in G2.derivs]
```

The derivative of `H2` with respect to a parameter that it does not depend on is `nothing`, which the gradient generator treats as a zero block. The corresponding gradient state remains zero, so `Ψ.grad_states[n]` is the derivative with respect to `u_problem[n]` for every trajectory:

```@example parameters
Ψ = propagate(GradVector(Ψ₀, length(u_problem)), G2, tlist; method = ExpProp)
[norm(Ψ′) for Ψ′ in Ψ.grad_states]
```

The same applies to the derivatives with respect to the controls of a control problem: `GradGenerator(H, controls)` takes the derivatives with respect to the given `controls`, in the given order, with `nothing` for any control that `H` does not depend on.

!!! warning "Not a single parameter block"

    The second argument of `GradGenerator` must be a list of controls or the result of `get_parameters`. A vector of numbers, such as a single parameter block `w`, is neither, and throws an `ArgumentError`. For the parameters in a single block, pass [`combine_parameters(w)`](@extref QuantumPropagators.Controls.combine_parameters).


## [How to propagate gradient vectors with the Chebychev propagator](@id Howto-Cheby)

The [Chebychev propagator](@extref QuantumPropagators :label:`method_cheby`) works for gradient vectors without any extra configuration. The gradient states it produces are the exact derivative of the Chebychev propagation of the state, even when that propagation is inaccurate: a polynomial in the gradient generator with coefficients that do not depend on the control maps a gradient vector to the derivative of the same polynomial applied to the state ([Corollary 7](@ref Background-cor-polynomial)). This guide shows how to propagate a gradient vector and how to verify the gradient against finite differences.

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

The gradient generator has the same eigenvalues as ``\hat{H}``, each with multiplicity ``L+1`` ([Proposition 8](@ref Background-prop-charpoly)), so the propagator obtains its spectral range from ``\hat{H}``. It uses the same Chebychev coefficients as a propagation of the state alone, so the state component is identical:

```@example cheby
Ψ.state == propagate(Ψ₀, H(0.0), tlist; method = Cheby)
```

The gradient vector is propagated without resetting the gradient state, so `Ψ.grad_states[1]` is the derivative ``\partial_u |Ψ(T)⟩`` of the final state with respect to the offset ``u``.

Do not pass `check_normalization = true` when propagating a gradient vector. That check compares expectation values ``⟨v|G|v⟩`` with the spectral range, and for a gradient generator, these can lie outside the spectral range of ``\hat{H}`` ([Non-normality](@ref Background-Non-Normality)), so the check fails even when the spectral range is correct.

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

When a gradient vector is propagated with an adaptive ODE solver, the error estimate that determines the time steps can ignore the gradient states or include them. By default, it ignores them, and the solver takes exactly the time steps it would take for the state alone (the *frozen* semantics). With [`flat_internalnorm`](@ref) as the error norm, it includes them (the *flat* semantics). See [Flat, frozen, and dual semantics](@ref Background-Semantics) for the theory. This guide shows how to select either behavior and what the choice does to the cost and the accuracy of the gradient.

A [`GradVector`](@ref) can be propagated with the ODE solvers in [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl) like any other state:

```@example ode
using LinearAlgebra
using Printf
using OrdinaryDiffEq
using QuantumPropagators: hamiltonian, propagate, init_prop, prop_step!
using QuantumGradientGenerators: GradGenerator, GradVector, flat_internalnorm

Ĥ₀ = ComplexF64[0 0 0; 0 1.0 0; 0 0 2.1]
Ĥ₁ = ComplexF64[0 1 0; 1 0 √2; 0 √2 0]
T = 5.0
tlist = [0.0, T]
ϵ(t) = 0.3 * sin(π * t / T)^2
Ψ₀ = ComplexF64[1, 0, 0]

H(u) = hamiltonian(Ĥ₀ + u * Ĥ₁, (Ĥ₁, ϵ))
G = GradGenerator(H(0.0))
nothing # hide
```

As in the [Chebychev guide](@ref Howto-Cheby), the derivative is with respect to a static offset ``u`` of the control.

For the frozen semantics, propagate with the default error norm:

```@example ode
Ψ = propagate(GradVector(Ψ₀, 1), G, tlist; method = OrdinaryDiffEq, reltol = 1e-6, abstol = 1e-6)
nothing # hide
```

The default error norm for a gradient vector is [`frozen_internalnorm`](@ref), the default norm of OrdinaryDiffEq (the root mean square of all elements) evaluated for `Ψ.state` only.

For the flat semantics, pass `internalnorm = flat_internalnorm`, the root mean square of all elements of the state and the gradient states:

```@example ode
Ψ = propagate(
    GradVector(Ψ₀, 1),
    G,
    tlist;
    method = OrdinaryDiffEq,
    reltol = 1e-6,
    abstol = 1e-6,
    internalnorm = flat_internalnorm
)
nothing # hide
```

To decide between the two, compare them over a range of tolerances. The following propagates the state alone ("bare"), and the gradient vector with both norms, and reports the number of time steps (from the underlying OrdinaryDiffEq integrator), the deviation of the state from the propagation of the state alone, and the error of the gradient relative to a reference propagation with a tight tolerance:

```@example ode
function propagate_ode(Ψ, generator; kwargs...)
    propagator = init_prop(Ψ, generator, tlist; method = OrdinaryDiffEq, kwargs...)
    prop_step!(propagator)
    return copy(propagator.state), propagator.integrator.stats.naccept
end

Ψ_ref, _ = propagate_ode(GradVector(Ψ₀, 1), G; alg = Vern9(), reltol = 1e-14, abstol = 1e-14)

@printf("%7s | %5s | %-27s | %-27s\n", "", "bare", "frozen", "flat")
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
    Ψ_frozen, n_frozen = propagate_ode(GradVector(Ψ₀, 1), G; kw...)
    Ψ_flat, n_flat = propagate_ode(GradVector(Ψ₀, 1), G; internalnorm = flat_internalnorm, kw...)
    @printf(
        "%7.0e | %5d | %5d %10.2e %10.2e | %5d %10.2e %10.2e\n",
        tol,
        n_bare,
        n_frozen,
        norm(Ψ_frozen.state - Ψ_bare),
        norm(Ψ_frozen.grad_states[1] - Ψ_ref.grad_states[1]),
        n_flat,
        norm(Ψ_flat.state - Ψ_bare),
        norm(Ψ_flat.grad_states[1] - Ψ_ref.grad_states[1]),
    )
end
```

With the default (frozen) norm, the solver takes the same time steps as for the state alone, and the state component is identical to the propagation of the state without gradients. The gradient is the derivative of the computed state, except for the dependence of the time steps on ``u``. Its error is a few times the tolerance. With the flat norm, the gradient states contribute to the error estimate. The solver then takes more time steps, the state deviates from the propagation of the state alone within the tolerance, and the error of the gradient is about the tolerance. The flat norm is not more efficient per time step, though: in this example, tightening the tolerance of the frozen propagation by a factor of about 3 gives a gradient about as accurate as the flat one, at about the same number of time steps.

Keep the default when the state must agree exactly with a propagation without gradients, e.g., when a functional is evaluated from separately propagated states and its gradient from gradient vectors. Use the flat norm when the tolerance should bound the error of the gradient directly, without calibrating it.

The default is also what forward-mode automatic differentiation with the dual numbers of [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) does: for a complex state of dual numbers, the default error norm of OrdinaryDiffEq is itself a dual number, whose value is the norm of the values of the state, and the solver chooses its time steps from that value. The result is the same as for a gradient vector with the default norm, except that rounding errors in the error estimate can occasionally change the acceptance of a time step at the threshold of the tolerance, and with it the subsequent time steps.
