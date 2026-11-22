# SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Random: GLOBAL_RNG
import QuantumControl.QuantumPropagators: _exp_prop_convert_operator
import QuantumControl.QuantumPropagators.Controls: get_controls
import QuantumControl.QuantumPropagators.SpectralRange: random_state
import QuantumControl.QuantumPropagators.Interfaces:
    supports_inplace, supports_matrix_interface


@doc raw"""Static gradient generator.

```julia
using QuantumPropagators.Controls: evaluate

G = evaluate(G_of_t::GradGenerator, args...; vals_dict)

G = GradgenOperator(H, control_derivs)
```

is the gradient generator ``G[\hat{H}]`` for specific values of all controls,
obtained by evaluating a time-dependent [`GradGenerator`](@ref), or directly
from a static dynamical generator `H` (``\hat{H}``) and a vector
`control_derivs` of static control derivatives ``\hat{μ}_l``. These are
available as `G.H` and `G.control_derivs`.

The resulting object can be multiplied directly with a gradient vector
[`GradVector`](@ref), e.g., in the process of evaluating a piecewise-constant
time propagation.
"""
struct GradgenOperator{num_controls,HT,CDT}
    H::HT
    control_derivs::Vector{CDT}
end


function GradgenOperator(H, control_derivs)
    num_controls = length(control_derivs)
    HT = typeof(H)
    CDT = eltype(control_derivs)
    return GradgenOperator{num_controls,HT,CDT}(H, control_derivs)
end


function get_controls(::GradgenOperator)
    return Tuple([])
end


function random_state(G::GradgenOperator; rng = GLOBAL_RNG, _...)
    state = random_state(G.H; rng)
    num_controls = length(G.control_derivs)
    grad_states = [random_state(G.H; rng) for _ in eachindex(G.control_derivs)]
    return GradVector{num_controls,typeof(state)}(state, grad_states)
end


_exp_prop_convert_operator(::GradgenOperator) = Matrix{ComplexF64}

supports_inplace(::Type{GradgenOperator{N,HT,CDT}}) where {N,HT,CDT} = supports_inplace(HT)
# Note: `evaluate!` for `GradgenOperator` reassigns `control_derivs[l]` via
# vector element assignment, so the in-place capability of `CDT` is irrelevant.
# Only the dynamical generator `HT` needs to support in-place evaluation.

supports_matrix_interface(::Type{<:GradgenOperator{N,HT,CDT}}) where {N,HT,CDT} =
    supports_matrix_interface(HT) && supports_matrix_interface(CDT)
