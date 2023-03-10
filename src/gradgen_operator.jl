import QuantumControlBase.QuantumPropagators: has_real_eigvals, _exp_prop_convert_operator
import QuantumControlBase.QuantumPropagators.Controls: get_controls
import QuantumControlBase.QuantumPropagators.SpectralRange: random_state


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


# Upper triangular block matrices have eigenvalues only from the diagonal
# blocks. This is an example for a matrix that has real eigenvalues despite not
# being Hermitian
has_real_eigvals(G::GradgenOperator) = has_real_eigvals(G.G)


function random_state(H::GradgenOperator)
    state = random_state(H.G)
    num_controls = length(H.control_deriv_ops)
    grad_states = [random_state(H.G) for i ∈ eachindex(H.control_deriv_ops)]
    return GradVector{num_controls,typeof(state)}(state, grad_states)
end


_exp_prop_convert_operator(::GradgenOperator) = Matrix{ComplexF64}
