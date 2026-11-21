# SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using QuantumControl.Controls: get_control_derivs
import QuantumControl.QuantumPropagators.Controls: get_controls
import QuantumControl.QuantumPropagators.Controls: get_parameters
import QuantumControl.QuantumPropagators: _exp_prop_convert_operator


@doc raw"""Time-dependent gradient generator.

```julia
G_of_t = GradGenerator(H)
```

for a time-dependent dynamical generator `H` (a Hamiltonian or Liouvillian
``\hat{H}(t)``) that depends on the controls ``\epsilon_1(t), \dots, \epsilon_L(t)``,
constructs the gradient generator ``G[\hat{H}](t)``. It contains `H` in
`G_of_t.H`, the control derivatives
``\hat{\mu}_l(t) = \partial \hat{H}/\partial \epsilon_l(t)`` in
`G_of_t.control_derivs`, and the controls in `G_of_t.controls`.

For a dynamical generator ``\hat{H}(t) = \hat{H}_0 + \sum_l \epsilon_l(t) \hat{H}_l``,
with ``\hat{\mu}_l = \hat{H}_l``, the gradient generator encodes the block
matrix

```math
G[\hat{H}] = \begin{pmatrix}
    \hat{H}  &  0       &  \dots   &  0        &  \hat{\mu}_1 \\
    0        &  \hat{H} &  \dots   &  0        &  \hat{\mu}_2 \\
    \vdots   &          &  \ddots  &           &  \vdots      \\
    0        &  0       &  \dots   &  \hat{H}  &  \hat{\mu}_L \\
    0        &  0       &  \dots   &  0        &  \hat{H}
\end{pmatrix}\,.
```

The control derivatives ``\hat{\mu}_l`` may be time-dependent, to account for
non-linear controls.

Plugging in values for the controls with
[`evaluate`](@extref QuantumPropagators.Controls.evaluate) results in a static
[`GradgenOperator`](@ref).
"""
struct GradGenerator{HT,CDT,CT}
    H::HT
    control_derivs::Vector{CDT}
    controls::Vector{CT}

    function GradGenerator(H::HT) where {HT}
        controls = collect(get_controls(H))
        control_derivs = get_control_derivs(H, controls)
        CT = eltype(controls)
        CDT = eltype(control_derivs)
        new{HT,CDT,CT}(H, control_derivs, controls)
    end

end


function get_controls(G_of_t::GradGenerator)
    return get_controls(G_of_t.H)
end


function get_parameters(G_of_t::GradGenerator)
    return get_parameters(G_of_t.H)
end


_exp_prop_convert_operator(::GradGenerator) = Matrix{ComplexF64}
