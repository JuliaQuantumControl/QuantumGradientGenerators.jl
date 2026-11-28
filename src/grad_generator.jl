# SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using QuantumControl.Controls: get_control_derivs, get_parameter_derivs
using RecursiveArrayTools: ArrayPartition
import QuantumControl.QuantumPropagators.Controls: get_controls
import QuantumControl.QuantumPropagators.Controls: get_parameters
import QuantumControl.QuantumPropagators: _exp_prop_convert_operator


@doc raw"""Time-dependent gradient generator.

```julia
G_of_t = GradGenerator(H)
G_of_t = GradGenerator(H, controls)
G_of_t = GradGenerator(H, parameters)
```

for a time-dependent dynamical generator `H` (a Hamiltonian or Liouvillian
``\hat{H}(t)``) constructs the gradient generator ``G[\hat{H}](t)`` for the
derivatives with respect to a list of real variables ``x_1, \dots, x_K``.
These are either controls ``ϵ_l(t)`` or control parameters ``u_n``. The
resulting object contains `H` in `G_of_t.H`, the derivatives
``\hat{μ}_k(t) = \partial \hat{H}/\partial x_k`` in `G_of_t.derivs`, and the
variables in `G_of_t.wrt`.

With a single argument, the variables are the controls of `H`, as obtained
from [`get_controls(H)`](@extref QuantumPropagators.Controls.get_controls).

With `controls` (a tuple or vector of controls) as the second argument, the
derivatives are the control derivatives from
[`get_control_derivs(H, controls)`](@extref QuantumControl.Controls.get_control_derivs),
and `G_of_t.wrt` is `collect(controls)`. This is for gradients with respect to
the controls of a control problem, where each trajectory may depend on a
different subset of the controls, in a different order.

With `parameters` (a `RecursiveArrayTools.ArrayPartition`, as returned by
[`get_parameters`](@extref QuantumPropagators.Controls.get_parameters)) as the
second argument, the derivatives are the parameter derivatives from
[`get_parameter_derivs(H, parameters)`](@extref QuantumControl.Controls.get_parameter_derivs),
and `G_of_t.wrt` is `parameters` itself. This is for gradients with respect
to the control parameters, e.g., `get_parameters(H)`, or
`get_parameters(problem)` for a control problem. The derivatives evaluate to
the current values of the parameters, so `G_of_t` remains valid when the
values in `parameters` change, e.g., in an optimization. To obtain the
derivatives with respect to the parameters in a single parameter block
`block`, pass `combine_parameters(block)`.

A derivative ``\hat{μ}_k`` is `nothing` if `H` does not depend on ``x_k``,
e.g., for a control or a parameter of another trajectory of a control problem.
In the gradient generator, it is a zero block, and the corresponding gradient
state remains zero. Thus, the index ``k`` of a gradient state always
identifies the variable `G_of_t.wrt[k]`.

For ``K`` derivatives, the gradient generator encodes the block matrix

```math
G[\hat{H}] = \begin{pmatrix}
    \hat{H}  &  0       &  \dots   &  0        &  \hat{μ}_1 \\
    0        &  \hat{H} &  \dots   &  0        &  \hat{μ}_2 \\
    \vdots   &          &  \ddots  &           &  \vdots      \\
    0        &  0       &  \dots   &  \hat{H}  &  \hat{μ}_K \\
    0        &  0       &  \dots   &  0        &  \hat{H}
\end{pmatrix}\,.
```

For a dynamical generator ``\hat{H}(t) = \hat{H}_0 + \sum_l ϵ_l(t) \hat{H}_l``
and derivatives with respect to the controls, ``\hat{μ}_l = \hat{H}_l``. The
derivatives may be time-dependent, e.g., for non-linear controls, and the
derivatives with respect to control parameters generally are.

Plugging in values for the controls with
[`evaluate`](@extref QuantumPropagators.Controls.evaluate) results in a static
[`GradgenOperator`](@ref).

```julia
G_of_t = GradGenerator(H, derivs, wrt)
```

constructs the gradient generator directly from a vector `derivs` of
derivatives (`nothing`, static operators, or time-dependent generators that
can be evaluated with
[`evaluate`](@extref QuantumPropagators.Controls.evaluate)), and the
corresponding variables `wrt`, which are stored as-is.
"""
struct GradGenerator{HT,DT,WT}
    H::HT
    derivs::Vector{DT}
    wrt::WT

    function GradGenerator(H::HT, derivs::AbstractVector, wrt::WT) where {HT,WT}
        if length(derivs) != length(wrt)
            msg = "The number of derivatives ($(length(derivs))) does not match the number of variables in `wrt` ($(length(wrt)))"
            throw(DimensionMismatch(msg))
        end
        derivs = collect(derivs)
        DT = eltype(derivs)
        new{HT,DT,WT}(H, derivs, wrt)
    end

end


GradGenerator(H) = GradGenerator(H, get_controls(H))


function GradGenerator(H, controls)
    controls = collect(controls)
    return GradGenerator(H, get_control_derivs(H, controls), controls)
end


function GradGenerator(H, parameters::ArrayPartition)
    return GradGenerator(H, get_parameter_derivs(H, parameters), parameters)
end

# Without this method, `GradGenerator(H, parameters)` would be ambiguous
# for the (usual) `ArrayPartition` of real numbers
function GradGenerator(H, parameters::ArrayPartition{<:Real})
    return GradGenerator(H, get_parameter_derivs(H, parameters), parameters)
end


function GradGenerator(H, values::AbstractVector{<:Real})
    msg = "`GradGenerator(H, wrt)` requires `wrt` to be a tuple or vector of controls, or the control parameters as returned by `get_parameters`, but received a vector of $(length(values)) real numbers. For a single control `ϵ`, pass `(ϵ,)`. For the parameters in a single parameter block `block`, pass `combine_parameters(block)`."
    throw(ArgumentError(msg))
end


function get_controls(G_of_t::GradGenerator)
    return get_controls(G_of_t.H)
end


function get_parameters(G_of_t::GradGenerator)
    return get_parameters(G_of_t.H)
end


_exp_prop_convert_operator(::GradGenerator) = Matrix{ComplexF64}
