# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

# Broadcasting over gradient vectors and scalars operates block by block:
# `Ψ .+ c .* Φ` is evaluated as `Ψ.state .+ c .* Φ.state` and
# `Ψ.grad_states[l] .+ c .* Φ.grad_states[l]` for each `l`. This is how ODE
# solvers (OrdinaryDiffEq) update their state, and it avoids indexing into the
# gradient vector element by element. All `GradVector` arguments must have the
# same number of gradient states.
#
# Broadcasting a gradient vector together with an array treats the gradient
# vector as a flat vector with the gradient states first and the state last
# (the layout of `getindex`), e.g., `V[:, 1] .= Ψ ./ β` in the Krylov methods of
# ExponentialUtilities. This requires the vector interface
# (`supports_vector_interface`) and indexes the gradient vector element by
# element.

using Base.Broadcast: Broadcasted, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle


struct GradVectorStyle <: BroadcastStyle end

Base.BroadcastStyle(::Type{<:GradVector}) = GradVectorStyle()

# With scalars (and `Ref`): block by block
Base.BroadcastStyle(s::GradVectorStyle, ::AbstractArrayStyle{0}) = s

# With arrays: flat
Base.BroadcastStyle(::GradVectorStyle, s::AbstractArrayStyle) = s

Base.broadcastable(Ψ::GradVector) = Ψ

Base.ndims(::Type{<:GradVector}) = 1

# The blocks are only known when the broadcast is materialized
Base.Broadcast.instantiate(bc::Broadcasted{GradVectorStyle}) = bc

# Element access for flat broadcasting
Base.getindex(Ψ::GradVector, I::CartesianIndex{1}) = Ψ[I[1]]


# The block `l` of a broadcast argument: the state for `l = 0`, the gradient
# state `l` otherwise
_block(Ψ::GradVector, l) = (l == 0) ? Ψ.state : Ψ.grad_states[l]
_block(bc::Broadcasted, l) = Broadcasted(bc.f, map(arg -> _block(arg, l), bc.args))
_block(x, l) = x

_num_grad_states(bc::Broadcasted) = _num_grad_states(bc.args...)
_num_grad_states(Ψ::GradVector, args...) = length(Ψ.grad_states)
_num_grad_states(bc::Broadcasted, args...) =
    something(_num_grad_states(bc), _num_grad_states(args...))
_num_grad_states(x, args...) = _num_grad_states(args...)
_num_grad_states() = nothing

_materialize_block(bc, l) = Base.Broadcast.materialize(_block(bc, l))


function Base.copy(bc::Broadcasted{GradVectorStyle})
    L = _num_grad_states(bc)
    state = _materialize_block(bc, 0)
    grad_states = [_materialize_block(bc, l) for l = 1:L]
    return GradVector{L,typeof(state)}(state, grad_states)
end


function Base.copyto!(dest::GradVector, bc::Broadcasted{GradVectorStyle})
    Base.Broadcast.materialize!(dest.state, _block(bc, 0))
    for l in eachindex(dest.grad_states)
        Base.Broadcast.materialize!(dest.grad_states[l], _block(bc, l))
    end
    return dest
end


# Only scalars on the right-hand side, e.g., `Ψ .= 0`
function Base.copyto!(dest::GradVector, bc::Broadcasted{<:AbstractArrayStyle{0}})
    copyto!(dest.state, bc)
    for l in eachindex(dest.grad_states)
        copyto!(dest.grad_states[l], bc)
    end
    return dest
end


# Arrays on the right-hand side: flat
function Base.copyto!(dest::GradVector, bc::Broadcasted{<:AbstractArrayStyle})
    return _copy_flat!(dest, Base.Broadcast.materialize(bc))
end


# Copy a flat vector (gradient states first, state last) into a gradient vector
function _copy_flat!(Ψ::GradVector, v::AbstractVector)
    N = length(Ψ.state)
    L = length(Ψ.grad_states)
    (length(v) == (L + 1) * N) || throw(
        DimensionMismatch(
            "A flat vector of length $(length(v)) does not match a GradVector of length $((L + 1) * N)"
        )
    )
    for l = 1:L
        copyto!(Ψ.grad_states[l], view(v, ((l-1)*N+1):(l*N)))
    end
    copyto!(Ψ.state, view(v, (L*N+1):((L+1)*N)))
    return Ψ
end
