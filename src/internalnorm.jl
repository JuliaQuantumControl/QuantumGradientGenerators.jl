# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

# The methods are defined in the `QuantumGradientGeneratorsDiffEqBaseExt`
# extension, since they build on the default norm of DiffEqBase


@doc raw"""Error norm of an ODE solver for the frozen semantics.

```julia
nrm = frozen_internalnorm(Ψ::GradVector, t)
```

returns the default norm of OrdinaryDiffEq (`DiffEqBase.ODE_DEFAULT_NORM`, the
root mean square of all elements) of the state `Ψ.state`, ignoring the
gradient states. It is meant to be passed as the `internalnorm` of an ODE
solver:

```julia
using OrdinaryDiffEq
Ψ = propagate(GradVector(Ψ₀, L), G, tlist; method = OrdinaryDiffEq, internalnorm = frozen_internalnorm)
```

The adaptive ODE solver then evaluates its error estimate from the state
component of the gradient vector only, and its time steps are the same as in the propagation of `Ψ.state` without gradients, so `Ψ.state` is the
result of that propagation (up to floating-point rounding), and the gradient
states are the derivative of that result with the time steps held fixed. This
is the ["frozen" semantics](@ref Background-Semantics) of propagating a
gradient vector. It is also what ForwardDiff dual numbers in a complex state
produce in OrdinaryDiffEq.

This is the default for a [`GradVector`](@ref): passing
`internalnorm = frozen_internalnorm` gives the same result as passing no
`internalnorm`. For the alternative, see [`flat_internalnorm`](@ref).

The methods of this function are only defined when DiffEqBase (a dependency
of OrdinaryDiffEq) is loaded.

```julia
frozen_internalnorm(u::Number, t)
```

is the default norm of OrdinaryDiffEq, which ODE solvers use for scalars.
"""
function frozen_internalnorm end


@doc raw"""Error norm of an ODE solver for the flat semantics.

```julia
nrm = flat_internalnorm(Ψ::GradVector, t)
```

returns the root mean square of all elements of the state `Ψ.state` and the
gradient states `Ψ.grad_states`. It is meant to be passed as the
`internalnorm` of an ODE solver:

```julia
using OrdinaryDiffEq
Ψ = propagate(GradVector(Ψ₀, L), G, tlist; method = OrdinaryDiffEq, internalnorm = flat_internalnorm)
```

The adaptive ODE solver then evaluates its error estimate from all components
of the gradient vector, as with the default norm of OrdinaryDiffEq for a
`Vector` of all the elements of the gradient vector. The time steps then depend
on the gradient states, so `Ψ.state` deviates from the propagation of the
state without gradients, within the tolerance of the solver. The gradient
states approximate the exact derivative of the propagated state, to the same
tolerance. This is the ["flat" semantics](@ref Background-Semantics) of
propagating a gradient vector.

Without `internalnorm`, an ODE solver uses [`frozen_internalnorm`](@ref) for
a [`GradVector`](@ref).

The methods of this function are only defined when DiffEqBase (a dependency
of OrdinaryDiffEq) is loaded.

```julia
flat_internalnorm(u::Number, t)
```

is the default norm of OrdinaryDiffEq, which ODE solvers use for scalars.
"""
function flat_internalnorm end
