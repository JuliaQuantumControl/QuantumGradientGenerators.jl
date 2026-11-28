# SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

import LinearAlgebra
import Base: +, -, *


# Add `α μ Ψ` to `Φ`, for a derivative `μ` that may be `nothing` (zero block)
_add_deriv_mul!(Φ, μ, Ψ, α) = LinearAlgebra.mul!(Φ, μ, Ψ, α, true)
_add_deriv_mul!(Φ, ::Nothing, Ψ, α) = Φ


function LinearAlgebra.mul!(Φ::GradVector, G::GradgenOperator, Ψ::GradVector, α, β)
    LinearAlgebra.mul!(Φ.state, G.H, Ψ.state, α, β)
    for k = 1:length(Ψ.grad_states)
        LinearAlgebra.mul!(Φ.grad_states[k], G.H, Ψ.grad_states[k], α, β)
        _add_deriv_mul!(Φ.grad_states[k], G.derivs[k], Ψ.state, α)
    end
    return Φ
end


function LinearAlgebra.mul!(Φ::GradVector, G::GradgenOperator, Ψ::GradVector)
    return LinearAlgebra.mul!(Φ, G, Ψ, true, false)
end


# Flat-vector dispatch used by ExponentialUtilities' Arnoldi iteration, which
# builds its Krylov matrix via similar(b, T, (n, m+1)) → plain Matrix{T} and
# then calls mul!(view(V,:,j+1), A, view(V,:,j)). The GradVector is treated as
# a packed flat vector: blocks [Ψ′₁ | Ψ′₂ | … | Ψ′_K | Ψ_state], each of length N.
function LinearAlgebra.mul!(
    Φ::AbstractVector,
    G::GradgenOperator{num_derivs,HT,DT},
    Ψ::AbstractVector,
    α,
    β
) where {num_derivs,HT,DT}
    N = size(G.H, 1)
    K = num_derivs
    Ψ_state = view(Ψ, (K*N+1):((K+1)*N))
    for k = 1:K
        Φ′ₖ = view(Φ, ((k-1)*N+1):(k*N))
        Ψ′ₖ = view(Ψ, ((k-1)*N+1):(k*N))
        LinearAlgebra.mul!(Φ′ₖ, G.H, Ψ′ₖ, α, β)
        _add_deriv_mul!(Φ′ₖ, G.derivs[k], Ψ_state, α)
    end
    Φ_state = view(Φ, (K*N+1):((K+1)*N))
    LinearAlgebra.mul!(Φ_state, G.H, Ψ_state, α, β)
    return Φ
end


function LinearAlgebra.mul!(Φ::AbstractVector, G::GradgenOperator, Ψ::AbstractVector)
    return LinearAlgebra.mul!(Φ, G, Ψ, true, false)
end


function LinearAlgebra.lmul!(c, Ψ::GradVector)
    LinearAlgebra.lmul!(c, Ψ.state)
    for k ∈ eachindex(Ψ.grad_states)
        LinearAlgebra.lmul!(c, Ψ.grad_states[k])
    end
    return Ψ
end


function LinearAlgebra.axpy!(a, X::GradVector, Y::GradVector)
    LinearAlgebra.axpy!(a, X.state, Y.state)
    for k ∈ eachindex(X.grad_states)
        LinearAlgebra.axpy!(a, X.grad_states[k], Y.grad_states[k])
    end
    return Y
end


# The Euclidean norm of the block vector, consistent with `dot`
function LinearAlgebra.norm(Ψ::GradVector)
    nrm_sq = LinearAlgebra.norm(Ψ.state)^2
    for k ∈ eachindex(Ψ.grad_states)
        nrm_sq += LinearAlgebra.norm(Ψ.grad_states[k])^2
    end
    return sqrt(nrm_sq)
end


function LinearAlgebra.dot(Ψ::GradVector, Φ::GradVector)
    c::ComplexF64 = LinearAlgebra.dot(Ψ.state, Φ.state)
    for k ∈ eachindex(Ψ.grad_states)
        c += LinearAlgebra.dot(Ψ.grad_states[k], Φ.grad_states[k])
    end
    return c
end


# ⟨Ψ|μ|Φ⟩ for a derivative `μ` that may be `nothing` (zero block)
_dot_deriv(Ψ, μ, Φ) = LinearAlgebra.dot(Ψ, μ, Φ)
_dot_deriv(Ψ, ::Nothing, Φ) = false  # strong zero


function LinearAlgebra.dot(Ψ::GradVector, G::GradgenOperator, Φ::GradVector)
    # Compute dot(Ψ, G * Φ) without forming the intermediate GradVector G * Φ
    c = LinearAlgebra.dot(Ψ.state, G.H, Φ.state)
    for k ∈ eachindex(Ψ.grad_states)
        c += LinearAlgebra.dot(Ψ.grad_states[k], G.H, Φ.grad_states[k])
        c += _dot_deriv(Ψ.grad_states[k], G.derivs[k], Φ.state)
    end
    return c
end


# A gradient generator is not normal unless all derivatives are zero
LinearAlgebra.ishermitian(G::GradgenOperator) = false


function Base.isreal(G::GradgenOperator)
    return (isreal(G.H) && all(isnothing(μ) || isreal(μ) for μ in G.derivs))
end


function Base.copyto!(dest::GradVector, src::GradVector)
    copyto!(dest.state, src.state)
    for k = 1:length(src.grad_states)
        copyto!(dest.grad_states[k], src.grad_states[k])
    end
    return dest
end


function Base.copy(Ψ::GradVector{num_derivs,T}) where {num_derivs,T}
    return GradVector{num_derivs,T}(copy(Ψ.state), [copy(Ψ′) for Ψ′ in Ψ.grad_states])
end


# === Vector interface for GradVector ===
#
# The following methods are part of the vector interface and are only
# meaningful when `supports_vector_interface` is true for the state type T.
# Each method delegates to a private `_name(::Val{supports}, ...)` function:
# the Val{true} method contains the implementation, and the Val{false} method
# throws an error.

function _length(::Val{true}, Ψ::GradVector)
    return length(Ψ.state) * (1 + length(Ψ.grad_states))
end

function _length(::Val{false}, Ψ::GradVector)
    error("$(typeof(Ψ)) does not support the vector interface")
end

function Base.length(Ψ::T) where {T<:GradVector}
    return _length(Val(supports_vector_interface(T)), Ψ)
end


function _size(::Val{true}, Ψ::GradVector{num_derivs,T}) where {num_derivs,T}
    return ((num_derivs + 1) * length(Ψ.state),)
end

function _size(::Val{false}, Ψ::GradVector)
    error("$(typeof(Ψ)) does not support the vector interface")
end

function Base.size(Ψ::T) where {T<:GradVector}
    return _size(Val(supports_vector_interface(T)), Ψ)
end


function _getindex(::Val{true}, Ψ::GradVector{num_derivs,T}, k::Int) where {num_derivs,T}
    N = length(Ψ.state)
    K = num_derivs
    block = (k - 1) ÷ N + 1
    local_k = (k - 1) % N + 1
    if block <= K
        return Ψ.grad_states[block][local_k]
    else
        return Ψ.state[local_k]
    end
end

function _getindex(::Val{false}, Ψ::GradVector, k::Int)
    error("$(typeof(Ψ)) does not support the vector interface")
end

function Base.getindex(Ψ::T, k::Int) where {T<:GradVector}
    return _getindex(Val(supports_vector_interface(T)), Ψ, k)
end


function _setindex!(
    ::Val{true},
    Ψ::GradVector{num_derivs,T},
    v,
    k::Int
) where {num_derivs,T}
    N = length(Ψ.state)
    K = num_derivs
    block = (k - 1) ÷ N + 1
    local_k = (k - 1) % N + 1
    if block <= K
        Ψ.grad_states[block][local_k] = v
    else
        Ψ.state[local_k] = v
    end
    return Ψ
end

function _setindex!(::Val{false}, Ψ::GradVector, v, k::Int)
    error("$(typeof(Ψ)) does not support the vector interface")
end

function Base.setindex!(Ψ::T, v, k::Int) where {T<:GradVector}
    return _setindex!(Val(supports_vector_interface(T)), Ψ, v, k)
end


function _iterate(::Val{true}, Ψ::GradVector, k)
    k > length(Ψ) && return nothing
    return (Ψ[k], k + 1)
end

function _iterate(::Val{false}, Ψ::GradVector, k)
    error("$(typeof(Ψ)) does not support the vector interface")
end

function Base.iterate(Ψ::T, k = 1) where {T<:GradVector}
    return _iterate(Val(supports_vector_interface(T)), Ψ, k)
end


function _eachindex(::Val{true}, Ψ::GradVector)
    return Base.OneTo(length(Ψ))
end

function _eachindex(::Val{false}, Ψ::GradVector)
    error("$(typeof(Ψ)) does not support the vector interface")
end

# ExponentialUtilities ≥ 1.34 loops over `eachindex(b)` in `firststep!`
function Base.eachindex(Ψ::T) where {T<:GradVector}
    return _eachindex(Val(supports_vector_interface(T)), Ψ)
end


function Base.similar(Ψ::GradVector{num_derivs,T}) where {num_derivs,T}
    state_sim = similar(Ψ.state)
    grad_states_sim = [similar(Ψ′) for Ψ′ ∈ Ψ.grad_states]
    return GradVector{num_derivs,typeof(state_sim)}(state_sim, grad_states_sim)
end

# A gradient vector with element type S. ODE solvers (OrdinaryDiffEq) use this
# for the buffer of their error estimate, so it must have the block structure
# of `Ψ` for the error norm to distinguish the state from the gradient states.
function Base.similar(Ψ::GradVector{num_derivs,T}, ::Type{S}) where {num_derivs,T,S}
    state_sim = similar(Ψ.state, S)
    grad_states_sim = [similar(Ψ′, S) for Ψ′ ∈ Ψ.grad_states]
    return GradVector{num_derivs,typeof(state_sim)}(state_sim, grad_states_sim)
end

# similar(Ψ, dims) calls eltype(Ψ) but not length/size, so no vector interface needed
Base.similar(Ψ::GradVector, dims::Tuple{Vararg{Int}}) = Array{eltype(Ψ)}(undef, dims)

# These definitions of `similar` exist to make ExponentialUtilities happy, but
# it's not clear at all that `similar` with a custom shape really makes sense
Base.similar(::GradVector, ::Type{T}, dims::Tuple{Int,Int}) where {T} =
    Matrix{T}(undef, dims...)

Base.similar(::GradVector, ::Type{T}, dims::Tuple{Int}) where {T} =
    Vector{T}(undef, dims[1])


function Base.fill!(Ψ::GradVector, v)
    Base.fill!(Ψ.state, v)
    for k = 1:length(Ψ.grad_states)
        Base.fill!(Ψ.grad_states[k], v)
    end
    return Ψ
end


function Base.size(G::GradgenOperator{num_derivs,HT,DT}) where {num_derivs,HT,DT}
    return (num_derivs + 1) .* size(G.H)
end

function Base.size(
    G::GradgenOperator{num_derivs,HT,DT},
    dim::Integer
) where {num_derivs,HT,DT}
    return (num_derivs + 1) * size(G.H, dim)
end


# === Matrix interface for GradgenOperator ===
#
# The following methods are part of the matrix interface and are only
# meaningful when `supports_matrix_interface` is true for both component types.
# Each method delegates to a private `_name(::Val{supports}, ...)` function:
# the Val{true} method contains the implementation, and the Val{false} method
# throws an error. Note that this does not include `size`, which must be
# defined for _all_ operators, whether or not they define the full matrix
# interface.


# As for an `Operator`, we implement `similar` to return a standard `Array`
# because `GradgenOperator` does not `setindex!`, so it's arguably not a
# "mutable array" even if its components are mutable.
Base.similar(G::GradgenOperator) = Array{eltype(G)}(undef, size(G))

Base.similar(G::GradgenOperator, ::Type{S}) where {S} = Array{S}(undef, size(G))
Base.similar(G::GradgenOperator, dims::Tuple{Vararg{Int}}) = Array{eltype(G)}(undef, dims)
Base.similar(G::GradgenOperator, ::Type{S}, dims::Tuple{Vararg{Int}}) where {S} =
    Array{S}(undef, dims)


function Base.eltype(::Type{GradgenOperator{num_derivs,HT,DT}}) where {num_derivs,HT,DT}
    DT_blocks = _nonnothing_type(DT)  # `nothing` is a zero block
    (DT_blocks === Union{}) && return eltype(HT)
    return promote_type(eltype(HT), eltype(DT_blocks))
end


function _getindex(
    ::Val{true},
    G::GradgenOperator{num_derivs,HT,DT},
    row::Int,
    col::Int
) where {num_derivs,HT,DT}
    T = eltype(G)
    N, M = size(G.H)
    K = num_derivs
    block_row = (row - 1) ÷ N + 1
    block_col = (col - 1) ÷ M + 1
    local_row = (row - 1) % N + 1
    local_col = (col - 1) % M + 1
    if block_row == block_col
        return convert(T, G.H[local_row, local_col])
    elseif block_col == K + 1 && block_row <= K
        μ = G.derivs[block_row]
        isnothing(μ) && return zero(T)
        return convert(T, μ[local_row, local_col])
    else
        return zero(T)
    end
end

function _getindex(::Val{false}, G::GradgenOperator, row::Int, col::Int)
    error("$(typeof(G)) does not support the matrix interface")
end

function Base.getindex(G::T, row::Int, col::Int) where {T<:GradgenOperator}
    return _getindex(Val(supports_matrix_interface(T)), G, row, col)
end


function _length(::Val{true}, G::GradgenOperator)
    return prod(size(G))
end

function _length(::Val{false}, G::GradgenOperator)
    error("$(typeof(G)) does not support the matrix interface")
end

function Base.length(G::T) where {T<:GradgenOperator}
    return _length(Val(supports_matrix_interface(T)), G)
end


function _iterate(::Val{true}, G::GradgenOperator, k)
    n = length(G)
    k > n && return nothing
    n_rows = size(G, 1)
    i = (k - 1) % n_rows + 1
    j = (k - 1) ÷ n_rows + 1
    return (G[i, j], k + 1)
end

function _iterate(::Val{false}, G::GradgenOperator, k)
    error("$(typeof(G)) does not support the matrix interface")
end

function Base.iterate(G::T, k = 1) where {T<:GradgenOperator}
    return _iterate(Val(supports_matrix_interface(T)), G, k)
end


function Base.eltype(::Type{GradVector{num_derivs,T}}) where {num_derivs,T}
    return eltype(T)
end

function Base.copyto!(dest::GradgenOperator, src::GradgenOperator)
    copyto!(dest.H, src.H)
    copyto!(dest.derivs, src.derivs)
end


function Base.zero(Ψ::GradVector{num_derivs,T}) where {num_derivs,T}
    return GradVector{num_derivs,T}(zero(Ψ.state), [zero(Ψ′) for Ψ′ ∈ Ψ.grad_states])
end


function -(Ψ::GradVector{num_derivs,T}, Φ::GradVector{num_derivs,T}) where {num_derivs,T}
    return GradVector{num_derivs,T}(
        Ψ.state - Φ.state,
        [a - b for (a, b) in zip(Ψ.grad_states, Φ.grad_states)]
    )
end


function +(Ψ::GradVector{num_derivs,T}, Φ::GradVector{num_derivs,T}) where {num_derivs,T}
    return GradVector{num_derivs,T}(
        Ψ.state + Φ.state,
        [a + b for (a, b) in zip(Ψ.grad_states, Φ.grad_states)]
    )
end


function *(α::Number, Ψ::GradVector{num_derivs,T}) where {num_derivs,T}
    return GradVector{num_derivs,T}(α * Ψ.state, [(α * Ψ′) for Ψ′ in Ψ.grad_states])
end


function *(Ψ::GradVector{num_derivs,T}, α::Number) where {num_derivs,T}
    return α * Ψ
end


_scale_deriv(μ, α) = μ * α
_scale_deriv(::Nothing, α) = nothing


function *(G::GradgenOperator{num_derivs,HT,DT}, α::Number) where {num_derivs,HT,DT}
    GradgenOperator{num_derivs,HT,DT}(G.H * α, [_scale_deriv(μ, α) for μ in G.derivs])
end

*(α::Number, G::GradgenOperator) = *(G::GradgenOperator, α::Number)


function *(
    G::GradgenOperator{num_derivs,HT,DT},
    Ψ::GradVector{num_derivs,ST}
) where {num_derivs,HT,DT,ST}
    state = G.H * Ψ.state
    grad_states = [G.H * Ψ′ for Ψ′ in Ψ.grad_states]
    for (k, μ) in enumerate(G.derivs)
        isnothing(μ) && continue  # zero block
        grad_states[k] += μ * Ψ.state
    end
    return GradVector{num_derivs,ST}(state, grad_states)
end


@inline function convert_gradvec_to_dense(Ψ)
    N = length(Ψ.state)
    K = length(Ψ.grad_states)
    Ψ_full = zeros(ComplexF64, N * (K + 1))
    convert_gradvec_to_dense!(Ψ_full, Ψ)
end


@inline function convert_gradvec_to_dense!(Ψ_full, Ψ)
    N = length(Ψ.state)
    K = length(Ψ.grad_states)
    @inbounds for k = 1:K
        Ψ_full[((k-1)*N+1):(k*N)] .= Ψ.grad_states[k]
    end
    @inbounds Ψ_full[(K*N+1):((K+1)*N)] .= Ψ.state
    return Ψ_full
end


@inline function convert_dense_to_gradvec!(Ψ, Ψ_full)
    N = length(Ψ.state)
    K = length(Ψ.grad_states)
    @inbounds for k = 1:K
        Ψ.grad_states[k] .= Ψ_full[((k-1)*N+1):(k*N)]
    end
    @inbounds Ψ.state .= Ψ_full[(K*N+1):((K+1)*N)]
    return Ψ
end


function Base.convert(::Type{Vector{ComplexF64}}, gradvec::GradVector)
    convert_gradvec_to_dense(gradvec)
end


function Base.convert(
    ::Type{GradVector{num_derivs,T}},
    vec::AbstractVector
) where {num_derivs,T}
    K = num_derivs
    N = length(vec) ÷ (K + 1)  # dimension of state
    @assert length(vec) == (K + 1) * N
    grad_states = [convert(T, vec[((k-1)*N+1):(k*N)]) for k = 1:K]
    state = convert(T, vec[(K*N+1):((K+1)*N)])
    return GradVector{num_derivs,T}(state, grad_states)
end

function Base.Array{T}(G::GradgenOperator) where {T}
    N, M = size(G.H)
    K = length(G.derivs)
    𝟘 = zeros(T, N, M)
    μ = [isnothing(μₖ) ? 𝟘 : Array{T}(μₖ) for μₖ in G.derivs]  # `nothing` is a zero block
    block_rows = [
        hcat([𝟘 for j = 1:(k-1)]..., Array{T}(G.H), [𝟘 for j = (k+1):K]..., μ[k]) for
        k = 1:K
    ]
    last_block_row = hcat([𝟘 for j = 1:K]..., Array{T}(G.H))
    return Base.Array{T}(vcat(block_rows..., last_block_row))
end


Base.Array(G::GradgenOperator) = Array{ComplexF64}(G)


function Base.convert(::Type{MT}, G::GradgenOperator) where {MT<:Matrix}
    Base.convert(MT, Base.Array(G))
end
