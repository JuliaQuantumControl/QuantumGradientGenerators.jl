import QuantumControlBase.QuantumPropagators: _exp_prop_convert_state
import QuantumControlBase.QuantumPropagators.Interfaces: supports_inplace


@doc raw"""Extended state-vector for the dynamic gradient.

```julia
Ψ̃ = GradVector(Ψ, num_controls)
```

for an initial state `Ψ` and `num_controls` control fields.

The `GradVector` conceptually corresponds to a direct-sum (block) column-vector
``Ψ̃ = (|Ψ̃₁⟩, |Ψ̃₂⟩, … |Ψ̃ₙ⟩, |Ψ⟩)^T``, where ``n`` is `num_controls`. With a
matching ``G̃`` as in the documentation of [`GradGenerator`](@ref),
we have

```math
G̃ Ψ̃ = \begin{pmatrix}
Ĥ |Ψ̃₁⟩ + Ĥ₁|Ψ⟩ \\
\vdots \\
Ĥ |Ψ̃ₙ⟩ + Ĥₙ|Ψ⟩ \\
Ĥ |Ψ⟩
\end{pmatrix}
```

and

```math
e^{-i G̃ dt} \begin{pmatrix} 0 \\ \vdots \\ 0 \\ |Ψ⟩ \end{pmatrix}
= \begin{pmatrix}
\frac{∂}{∂ϵ₁} e^{-i Ĥ dt} |Ψ⟩ \\
\vdots \\
\frac{∂}{∂ϵₙ} e^{-i Ĥ dt} |Ψ⟩ \\
e^{-i Ĥ dt} |Ψ⟩
\end{pmatrix}.
```

Upon initialization, ``|Ψ̃₁⟩…|Ψ̃ₙ⟩`` are zero.
"""
struct GradVector{num_controls,T}
    state::T
    grad_states::Vector{T}
end

function GradVector(Ψ::T, num_controls::Int64) where {T}
    grad_states = [zero(Ψ) for _ = 1:num_controls]
    GradVector{num_controls,T}(copy(Ψ), grad_states)
end


"""Reset the given gradient vector for a new gradient evaluation.

```julia
resetgradvec!(Ψ̃::GradVector)
```

zeroes out `Ψ̃.grad_states` but leaves `Ψ̃.state` unaffected. This is possible
whether or not Ψ̃ supports in-place operations
([`QuantumPropagators.Interfaces.supports_inplace`](@ref))

```julia
resetgradvec!(Ψ̃::GradVector, Ψ)
```

additionally sets `Ψ̃.state` to `Ψ`, which requires that `Ψ̃.state` supports
in-place operations.

Returns `Ψ̃`.
"""
function resetgradvec!(Ψ̃::GradVector)
    if supports_inplace(Ψ̃)
        for i in eachindex(Ψ̃.grad_states)
            fill!(Ψ̃.grad_states[i], 0.0)
        end
    else
        for i in eachindex(Ψ̃.grad_states)
            Ψ̃.grad_states[i] = zero(Ψ̃.state)
        end
    end
    return Ψ̃
end

function resetgradvec!(Ψ̃::GradVector{num_controls,T}, Ψ::T) where {num_controls,T}
    copyto!(Ψ̃.state, Ψ)
    resetgradvec!(Ψ̃)
end


_exp_prop_convert_state(::GradVector) = Vector{ComplexF64}

supports_inplace(Ψ̃::GradVector) = supports_inplace(Ψ̃.state)
