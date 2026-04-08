using Random: GLOBAL_RNG
import QuantumControl.QuantumPropagators: _exp_prop_convert_operator
import QuantumControl.QuantumPropagators.Controls: get_controls
import QuantumControl.QuantumPropagators.SpectralRange: random_state
import QuantumControl.QuantumPropagators.Interfaces:
    supports_inplace, supports_matrix_interface


"""Static generator for the dynamic gradient.

```julia
using QuantumPropagators.Controls: evaluate

G::GradgenOperator = evaluate(gradgen::GradGenerator; vals_dict)

G = GradgenOperator(G.G, G.control_deriv_ops)
```

is the result of plugging in specific values for all controls in a
[`GradGenerator`](@ref).

The resulting object can be multiplied directly with a [`GradVector`](@ref),
e.g., in the process of evaluating a piecewise-constant time propagation.
"""
struct GradgenOperator{num_controls,GT,CGT}
    G::GT
    control_deriv_ops::Vector{CGT}
end


function GradgenOperator(G, control_deriv_ops)
    num_controls = length(control_deriv_ops)
    GT = typeof(G)
    CGT = eltype(control_deriv_ops)
    return GradgenOperator{num_controls,GT,CGT}(G, control_deriv_ops)
end


function get_controls(O1::GradgenOperator)
    return Tuple([])
end


function random_state(H::GradgenOperator; rng = GLOBAL_RNG, _...)
    state = random_state(H.G; rng)
    num_controls = length(H.control_deriv_ops)
    grad_states = [random_state(H.G; rng) for _ in eachindex(H.control_deriv_ops)]
    return GradVector{num_controls,typeof(state)}(state, grad_states)
end


_exp_prop_convert_operator(::GradgenOperator) = Matrix{ComplexF64}

supports_inplace(::Type{GradgenOperator{N,GT,CGT}}) where {N,GT,CGT} =
    (supports_inplace(GT) && supports_inplace(CGT))

supports_matrix_interface(::Type{<:GradgenOperator{N,GT,CGT}}) where {N,GT,CGT} =
    supports_matrix_interface(GT) && supports_matrix_interface(CGT)
