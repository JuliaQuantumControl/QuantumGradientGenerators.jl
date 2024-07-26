using Random: GLOBAL_RNG
import QuantumControlBase.QuantumPropagators: _exp_prop_convert_operator
import QuantumControlBase.QuantumPropagators.Controls: get_controls
import QuantumControlBase.QuantumPropagators.SpectralRange: random_state
import QuantumControlBase.QuantumPropagators.Interfaces: supports_inplace


"""Static generator for the dynamic gradient.

```julia
using QuantumPropagators.Controls: evaluate

G::GradgenOperator = evaluate(gradgen::GradGenerator; vals_dict)
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


function get_controls(O1::GradgenOperator)
    return Tuple([])
end


function random_state(H::GradgenOperator; rng=GLOBAL_RNG, _...)
    state = random_state(H.G; rng)
    num_controls = length(H.control_deriv_ops)
    grad_states = [random_state(H.G; rng) for i ∈ eachindex(H.control_deriv_ops)]
    return GradVector{num_controls,typeof(state)}(state, grad_states)
end


_exp_prop_convert_operator(::GradgenOperator) = Matrix{ComplexF64}

supports_inplace(::GradgenOperator) = true
