# SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

import QuantumControl.QuantumPropagators: _exp_prop_convert_state
import QuantumControl.QuantumPropagators.Interfaces:
    supports_inplace, supports_vector_interface


@doc raw"""Gradient vector for the dynamic gradient.

```julia
Ψ = GradVector(state, num_controls)
```

creates a gradient vector ``\Psi`` for ``L`` = `num_controls` controls. It
corresponds to the block column vector

```math
\Psi = \begin{pmatrix}
|\Psi^\prime_1⟩ \\ \vdots \\ |\Psi^\prime_L⟩ \\ |\Psi⟩
\end{pmatrix}\,,
```

with the state ``|\Psi⟩`` in `Ψ.state` and the gradient states ``|\Psi^\prime_l⟩``
in `Ψ.grad_states[l]`. Note that the gradient vector ``\Psi`` is written without
a ket, to distinguish it from the state ``|\Psi⟩``. Upon initialization,
``|\Psi⟩`` is a copy of `state`, and the gradient states are zero.

With a matching gradient generator ``G = G[\hat{H}]``, see
[`GradGenerator`](@ref), we have

```math
G \Psi = \begin{pmatrix}
\hat{H} |\Psi^\prime_1⟩ + \hat{\mu}_1 |\Psi⟩ \\
\vdots \\
\hat{H} |\Psi^\prime_L⟩ + \hat{\mu}_L |\Psi⟩ \\
\hat{H} |\Psi⟩
\end{pmatrix}
```

and

```math
e^{-i G dt} \begin{pmatrix} 0 \\ \vdots \\ 0 \\ |\Psi⟩ \end{pmatrix}
= \begin{pmatrix}
\frac{\partial}{\partial \epsilon_1} e^{-i \hat{H} dt} |\Psi⟩ \\
\vdots \\
\frac{\partial}{\partial \epsilon_L} e^{-i \hat{H} dt} |\Psi⟩ \\
e^{-i \hat{H} dt} |\Psi⟩
\end{pmatrix}\,.
```
"""
struct GradVector{num_controls,T}
    state::T
    grad_states::Vector{T}
end

function GradVector(state::T, num_controls::Int64) where {T}
    grad_states = [zero(state) for _ = 1:num_controls]
    GradVector{num_controls,T}(copy(state), grad_states)
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
        for l in eachindex(Ψ.grad_states)
            Ψ.grad_states[l] = zero(Ψ.state)
        end
    end
    return Ψ
end

function resetgradvec!(Ψ::GradVector{num_controls,T}, state::T) where {num_controls,T}
    copyto!(Ψ.state, state)
    resetgradvec!(Ψ)
end


_exp_prop_convert_state(::GradVector) = Vector{ComplexF64}


function Base.convert(
    ::Type{GradVector{num_controls,T2}},
    Ψ::GradVector{num_controls,T1}
) where {T1,T2,num_controls}
    # Used, e.g., for Cheby initialization
    return GradVector{num_controls,T2}(
        convert(T2, Ψ.state),
        T2[convert(T2, Ψ′) for Ψ′ in Ψ.grad_states]
    )
end


supports_inplace(::Type{GradVector{N,T}}) where {N,T} = supports_inplace(T)

supports_vector_interface(::Type{GradVector{N,T}}) where {N,T} =
    supports_vector_interface(T)
