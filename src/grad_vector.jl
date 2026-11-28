# SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

import QuantumControl.QuantumPropagators: _exp_prop_convert_state
import QuantumControl.QuantumPropagators.Interfaces:
    supports_inplace, supports_vector_interface


@doc raw"""Gradient vector for the dynamic gradient.

```julia
Ψ = GradVector(state, num_derivs)
```

creates a gradient vector ``Ψ`` for the derivatives with respect to ``K`` =
`num_derivs` real variables ``x_1, \dots, x_K`` (controls or control
parameters, see [`GradGenerator`](@ref)). It corresponds to the block column
vector

```math
Ψ = \begin{pmatrix}
|Ψ^\prime_1⟩ \\ \vdots \\ |Ψ^\prime_K⟩ \\ |Ψ⟩
\end{pmatrix}\,,
```

with the state ``|Ψ⟩`` in `Ψ.state` and the gradient states ``|Ψ^\prime_k⟩``
in `Ψ.grad_states[k]`. Note that the gradient vector ``Ψ`` is written without
a ket, to distinguish it from the state ``|Ψ⟩``. Upon initialization,
``|Ψ⟩`` is a copy of `state`, and the gradient states are zero.

With a matching gradient generator ``G = G[\hat{H}]`` for the derivatives
``\hat{μ}_k = \partial \hat{H} / \partial x_k``, see
[`GradGenerator`](@ref), we have

```math
G Ψ = \begin{pmatrix}
\hat{H} |Ψ^\prime_1⟩ + \hat{μ}_1 |Ψ⟩ \\
\vdots \\
\hat{H} |Ψ^\prime_K⟩ + \hat{μ}_K |Ψ⟩ \\
\hat{H} |Ψ⟩
\end{pmatrix}
```

and

```math
e^{-i G dt} \begin{pmatrix} 0 \\ \vdots \\ 0 \\ |Ψ⟩ \end{pmatrix}
= \begin{pmatrix}
\frac{\partial}{\partial x_1} e^{-i \hat{H} dt} |Ψ⟩ \\
\vdots \\
\frac{\partial}{\partial x_K} e^{-i \hat{H} dt} |Ψ⟩ \\
e^{-i \hat{H} dt} |Ψ⟩
\end{pmatrix}\,.
```
"""
struct GradVector{num_derivs,T}
    state::T
    grad_states::Vector{T}
end

function GradVector(state::T, num_derivs::Int64) where {T}
    grad_states = [zero(state) for _ = 1:num_derivs]
    GradVector{num_derivs,T}(copy(state), grad_states)
end


"""Reset the given gradient vector for a new gradient evaluation.

```julia
resetgradvec!(Ψ::GradVector)
```

zeroes out the gradient states `Ψ.grad_states` but leaves `Ψ.state`
unaffected. This is possible whether or not `Ψ` supports in-place operations
([`QuantumPropagators.Interfaces.supports_inplace`](@ref))

```julia
resetgradvec!(Ψ::GradVector, state)
```

additionally sets `Ψ.state` to `state`, which requires that `Ψ.state` supports
in-place operations.

Returns `Ψ`.
"""
function resetgradvec!(Ψ::T) where {T<:GradVector}
    if supports_inplace(T)
        for Ψ′ in Ψ.grad_states
            fill!(Ψ′, 0.0)
        end
    else
        for k in eachindex(Ψ.grad_states)
            Ψ.grad_states[k] = zero(Ψ.state)
        end
    end
    return Ψ
end

function resetgradvec!(Ψ::GradVector{num_derivs,T}, state::T) where {num_derivs,T}
    copyto!(Ψ.state, state)
    resetgradvec!(Ψ)
end


_exp_prop_convert_state(::GradVector) = Vector{ComplexF64}


function Base.convert(
    ::Type{GradVector{num_derivs,T2}},
    Ψ::GradVector{num_derivs,T1}
) where {T1,T2,num_derivs}
    # Used, e.g., for Cheby initialization
    return GradVector{num_derivs,T2}(
        convert(T2, Ψ.state),
        T2[convert(T2, Ψ′) for Ψ′ in Ψ.grad_states]
    )
end


supports_inplace(::Type{GradVector{N,T}}) where {N,T} = supports_inplace(T)

supports_vector_interface(::Type{GradVector{N,T}}) where {N,T} =
    supports_vector_interface(T)
