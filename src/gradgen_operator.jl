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

G = GradgenOperator(H, derivs)
```

is the gradient generator ``G[\hat{H}]`` for specific values of all controls,
obtained by evaluating a time-dependent [`GradGenerator`](@ref), or directly
from a static dynamical generator `H` (``\hat{H}``) and a vector `derivs` of
static derivatives ``\hat{μ}_k`` of ``\hat{H}`` with respect to real
variables ``x_k`` (controls or control parameters). These are available as
`G.H` and `G.derivs`. An element of `derivs` may be `nothing`, for a variable
that ``\hat{H}`` does not depend on. It is treated as a zero block.

The resulting object can be multiplied directly with a gradient vector
[`GradVector`](@ref), e.g., in the process of evaluating a piecewise-constant
time propagation.

Since ``G`` is block upper-triangular with ``\hat{H}`` in all diagonal blocks,
its eigenvalues are those of ``\hat{H}``. Accordingly, `eigvals`, `tr`, `det`,
and `specrange` (and thus the spectral range used by the Chebychev propagator)
are evaluated from `G.H`, see the
[Properties of the gradient generator](@ref Background-Properties).
"""
struct GradgenOperator{num_derivs,HT,DT}
    H::HT
    derivs::Vector{DT}
end


function GradgenOperator(H, derivs)
    num_derivs = length(derivs)
    HT = typeof(H)
    DT = eltype(derivs)
    return GradgenOperator{num_derivs,HT,DT}(H, derivs)
end


function get_controls(::GradgenOperator)
    return Tuple([])
end


function random_state(G::GradgenOperator; rng = GLOBAL_RNG, _...)
    state = random_state(G.H; rng)
    num_derivs = length(G.derivs)
    grad_states = [random_state(G.H; rng) for _ in eachindex(G.derivs)]
    return GradVector{num_derivs,typeof(state)}(state, grad_states)
end


# The type of the derivatives in `G.derivs` that are not `nothing` (zero
# blocks), for the element type `DT` of `G.derivs`. This is `Union{}` if all
# derivatives are `nothing`.
_nonnothing_type(::Type{T}) where {T} = T
_nonnothing_type(::Type{Nothing}) = Union{}
_nonnothing_type(::Type{Union{Nothing,T}}) where {T} = T


_exp_prop_convert_operator(::GradgenOperator) = Matrix{ComplexF64}

supports_inplace(::Type{GradgenOperator{N,HT,DT}}) where {N,HT,DT} = supports_inplace(HT)
# Note: `evaluate!` for `GradgenOperator` reassigns `derivs[k]` via vector
# element assignment, so the in-place capability of `DT` is irrelevant. Only
# the dynamical generator `HT` needs to support in-place evaluation.

function supports_matrix_interface(::Type{<:GradgenOperator{N,HT,DT}}) where {N,HT,DT}
    DT_blocks = _nonnothing_type(DT)  # `nothing` is a zero block
    return supports_matrix_interface(HT) &&
           ((DT_blocks === Union{}) || supports_matrix_interface(DT_blocks))
end
