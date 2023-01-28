import QuantumPropagators
using QuantumControlBase: get_control_derivs
import QuantumPropagators.Controls: get_controls


@doc raw"""Extended generator for the standard dynamic gradient.

```julia
G̃ = GradGenerator(G)
```

contains the original time-dependent generator `G` (a Hamiltonian or
Liouvillian) in `G̃.G`, a vector of control derivatives ``∂G/∂ϵₗ(t)`` in
`G̃.control_derivs`, and the controls in `G̃.controls`.

For a generator ``G = Ĥ(t) = Ĥ₀ + ϵ₁(t) Ĥ₁ + … +  ϵₙ(t) Ĥₙ``, this extended
generator encodes the block-matrix

```math
G̃ = \begin{pmatrix}
         Ĥ(t)  &  0    &  \dots   &  0     &  Ĥ₁     \\
         0     &  Ĥ(t) &  \dots   &  0     &  Ĥ₂     \\
    \vdots     &       &  \ddots  &        &  \vdots \\
         0     &  0    &  \dots   &  Ĥ(t)  &  Ĥₙ     \\
         0     &  0    &  \dots   &  0     &  Ĥ(t)
\end{pmatrix}
```

Note that the ``∂G/∂ϵₗ(t)`` (``Ĥₗ`` in the above example) may be
time-dependent, to account for the possibility of non-linear control terms.
"""
struct GradGenerator{GT,CDT,CT}
    G::GT
    control_derivs::Vector{CDT}
    controls::Vector{CT}

    function GradGenerator(G::GT) where {GT}
        controls = collect(get_controls(G))
        control_derivs = get_control_derivs(G, controls)
        CT = eltype(controls)
        CDT = eltype(control_derivs)
        new{GT,CDT,CT}(G, control_derivs, controls)
    end

end


function get_controls(gradgen::GradGenerator)
    return get_controls(gradgen.G)
end


QuantumPropagators._exp_prop_convert_operator(::GradGenerator) = Matrix{ComplexF64}
