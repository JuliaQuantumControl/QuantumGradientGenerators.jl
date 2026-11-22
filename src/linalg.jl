# SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

import LinearAlgebra
import Base: +, -, *


function LinearAlgebra.mul!(Φ::GradVector, G::GradgenOperator, Ψ::GradVector, α, β)
    LinearAlgebra.mul!(Φ.state, G.H, Ψ.state, α, β)
    for l = 1:length(Ψ.grad_states)
        LinearAlgebra.mul!(Φ.grad_states[l], G.H, Ψ.grad_states[l], α, β)
        LinearAlgebra.mul!(Φ.grad_states[l], G.control_derivs[l], Ψ.state, α, 1)
    end
    return Φ
end


function LinearAlgebra.mul!(Φ::GradVector, G::GradgenOperator, Ψ::GradVector)
    return LinearAlgebra.mul!(Φ, G, Ψ, true, false)
end


# Flat-vector dispatch used by ExponentialUtilities' Arnoldi iteration, which
# builds its Krylov matrix via similar(b, T, (n, m+1)) → plain Matrix{T} and
# then calls mul!(view(V,:,j+1), A, view(V,:,j)). The GradVector is treated as
# a packed flat vector: blocks [Ψ′₁ | Ψ′₂ | … | Ψ′_L | Ψ_state], each of length N.
function LinearAlgebra.mul!(
    Φ::AbstractVector,
    G::GradgenOperator{num_controls,HT,CDT},
    Ψ::AbstractVector,
    α,
    β
) where {num_controls,HT,CDT}
    N = size(G.H, 1)
    L = num_controls
    Ψ_state = view(Ψ, (L*N+1):((L+1)*N))
    for l = 1:L
        Φ′ₗ = view(Φ, ((l-1)*N+1):(l*N))
        Ψ′ₗ = view(Ψ, ((l-1)*N+1):(l*N))
        LinearAlgebra.mul!(Φ′ₗ, G.H, Ψ′ₗ, α, β)
        LinearAlgebra.mul!(Φ′ₗ, G.control_derivs[l], Ψ_state, α, 1)
    end
    Φ_state = view(Φ, (L*N+1):((L+1)*N))
    LinearAlgebra.mul!(Φ_state, G.H, Ψ_state, α, β)
    return Φ
end


function LinearAlgebra.mul!(Φ::AbstractVector, G::GradgenOperator, Ψ::AbstractVector)
    return LinearAlgebra.mul!(Φ, G, Ψ, true, false)
end


function LinearAlgebra.lmul!(c, Ψ::GradVector)
    LinearAlgebra.lmul!(c, Ψ.state)
    for l ∈ eachindex(Ψ.grad_states)
        LinearAlgebra.lmul!(c, Ψ.grad_states[l])
    end
    return Ψ
end


function LinearAlgebra.axpy!(a, X::GradVector, Y::GradVector)
    LinearAlgebra.axpy!(a, X.state, Y.state)
    for l ∈ eachindex(X.grad_states)
        LinearAlgebra.axpy!(a, X.grad_states[l], Y.grad_states[l])
    end
    return Y
end


# The Euclidean norm of the block vector, consistent with `dot`
function LinearAlgebra.norm(Ψ::GradVector)
    nrm_sq = LinearAlgebra.norm(Ψ.state)^2
    for l ∈ eachindex(Ψ.grad_states)
        nrm_sq += LinearAlgebra.norm(Ψ.grad_states[l])^2
    end
    return sqrt(nrm_sq)
end


function LinearAlgebra.dot(Ψ::GradVector, Φ::GradVector)
    c::ComplexF64 = LinearAlgebra.dot(Ψ.state, Φ.state)
    for l ∈ eachindex(Ψ.grad_states)
        c += LinearAlgebra.dot(Ψ.grad_states[l], Φ.grad_states[l])
    end
    return c
end


function LinearAlgebra.dot(Ψ::GradVector, G::GradgenOperator, Φ::GradVector)
    return LinearAlgebra.dot(Ψ, G * Φ)
end


# A gradient generator is not normal unless all control derivatives are zero
LinearAlgebra.ishermitian(G::GradgenOperator) = false


function Base.isreal(G::GradgenOperator)
    return (isreal(G.H) && all(isreal(μ) for μ in G.control_derivs))
end


function Base.copyto!(dest::GradVector, src::GradVector)
    copyto!(dest.state, src.state)
    for l = 1:length(src.grad_states)
        copyto!(dest.grad_states[l], src.grad_states[l])
    end
    return dest
end


function Base.copy(Ψ::GradVector{num_controls,T}) where {num_controls,T}
    return GradVector{num_controls,T}(copy(Ψ.state), [copy(Ψ′) for Ψ′ in Ψ.grad_states])
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


function _size(::Val{true}, Ψ::GradVector{num_controls,T}) where {num_controls,T}
    return ((num_controls + 1) * length(Ψ.state),)
end

function _size(::Val{false}, Ψ::GradVector)
    error("$(typeof(Ψ)) does not support the vector interface")
end

function Base.size(Ψ::T) where {T<:GradVector}
    return _size(Val(supports_vector_interface(T)), Ψ)
end


function _getindex(
    ::Val{true},
    Ψ::GradVector{num_controls,T},
    k::Int
) where {num_controls,T}
    N = length(Ψ.state)
    L = num_controls
    block = (k - 1) ÷ N + 1
    local_k = (k - 1) % N + 1
    if block <= L
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
    Ψ::GradVector{num_controls,T},
    v,
    k::Int
) where {num_controls,T}
    N = length(Ψ.state)
    L = num_controls
    block = (k - 1) ÷ N + 1
    local_k = (k - 1) % N + 1
    if block <= L
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


function Base.similar(Ψ::GradVector{num_controls,T}) where {num_controls,T}
    state_sim = similar(Ψ.state)
    grad_states_sim = [similar(Ψ′) for Ψ′ ∈ Ψ.grad_states]
    return GradVector{num_controls,typeof(state_sim)}(state_sim, grad_states_sim)
end

# similar(Ψ, S) calls length(Ψ), which will error if !supports_vector_interface
Base.similar(Ψ::GradVector, ::Type{S}) where {S} = Vector{S}(undef, length(Ψ))

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
    for l = 1:length(Ψ.grad_states)
        Base.fill!(Ψ.grad_states[l], v)
    end
    return Ψ
end


function Base.size(G::GradgenOperator{num_controls,HT,CDT}) where {num_controls,HT,CDT}
    return (num_controls + 1) .* size(G.H)
end

function Base.size(
    G::GradgenOperator{num_controls,HT,CDT},
    dim::Integer
) where {num_controls,HT,CDT}
    return (num_controls + 1) * size(G.H, dim)
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


function Base.eltype(
    ::Type{GradgenOperator{num_controls,HT,CDT}}
) where {num_controls,HT,CDT}
    return promote_type(eltype(HT), eltype(CDT))
end


function _getindex(
    ::Val{true},
    G::GradgenOperator{num_controls,HT,CDT},
    row::Int,
    col::Int
) where {num_controls,HT,CDT}
    T = eltype(G)
    N, M = size(G.H)
    L = num_controls
    block_row = (row - 1) ÷ N + 1
    block_col = (col - 1) ÷ M + 1
    local_row = (row - 1) % N + 1
    local_col = (col - 1) % M + 1
    if block_row == block_col
        return convert(T, G.H[local_row, local_col])
    elseif block_col == L + 1 && block_row <= L
        return convert(T, G.control_derivs[block_row][local_row, local_col])
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


function Base.eltype(::Type{GradVector{num_controls,T}}) where {num_controls,T}
    return eltype(T)
end

function Base.copyto!(dest::GradgenOperator, src::GradgenOperator)
    copyto!(dest.H, src.H)
    copyto!(dest.control_derivs, src.control_derivs)
end


function Base.zero(Ψ::GradVector{num_controls,T}) where {num_controls,T}
    return GradVector{num_controls,T}(zero(Ψ.state), [zero(Ψ′) for Ψ′ ∈ Ψ.grad_states])
end


function -(
    Ψ::GradVector{num_controls,T},
    Φ::GradVector{num_controls,T}
) where {num_controls,T}
    return GradVector{num_controls,T}(
        Ψ.state - Φ.state,
        [a - b for (a, b) in zip(Ψ.grad_states, Φ.grad_states)]
    )
end


function +(
    Ψ::GradVector{num_controls,T},
    Φ::GradVector{num_controls,T}
) where {num_controls,T}
    return GradVector{num_controls,T}(
        Ψ.state + Φ.state,
        [a + b for (a, b) in zip(Ψ.grad_states, Φ.grad_states)]
    )
end


function *(α::Number, Ψ::GradVector{num_controls,T}) where {num_controls,T}
    return GradVector{num_controls,T}(α * Ψ.state, [(α * Ψ′) for Ψ′ in Ψ.grad_states])
end


function *(Ψ::GradVector{num_controls,T}, α::Number) where {num_controls,T}
    return α * Ψ
end


function *(G::GradgenOperator{num_controls,HT,CDT}, α::Number) where {num_controls,HT,CDT}
    GradgenOperator{num_controls,HT,CDT}(G.H * α, [μ * α for μ in G.control_derivs])
end

*(α::Number, G::GradgenOperator) = *(G::GradgenOperator, α::Number)


function *(
    G::GradgenOperator{num_controls,HT,CDT},
    Ψ::GradVector{num_controls,ST}
) where {num_controls,HT,CDT,ST}
    state = G.H * Ψ.state
    grad_states = [G.H * Ψ′ for Ψ′ in Ψ.grad_states]
    for (l, μ) in enumerate(G.control_derivs)
        grad_states[l] += μ * Ψ.state
    end
    return GradVector{num_controls,ST}(state, grad_states)
end


@inline function convert_gradgen_to_dense(G::GradGenerator)
    N = size(G.H)[1]
    L = length(G.control_derivs)
    G_full = zeros(eltype(G.H), N * (L + 1), N * (L + 1))
    convert_gradgen_to_dense!(G_full, G)
end


@inline function convert_gradgen_to_dense!(G_full, G::GradGenerator)
    N = size(G.H)[1]
    L = length(G.control_derivs)
    @inbounds for i = 1:(L+1)
        G_full[((i-1)*N+1):(i*N), ((i-1)*N+1):(i*N)] .= G.H
    end
    # Set the control-derivatives in the last (block-)column
    @inbounds for l = 1:L
        G_full[((l-1)*N+1):(l*N), (L*N+1):((L+1)*N)] .= G.control_derivs[l]
    end
    return G_full
end


@inline function convert_gradvec_to_dense(Ψ)
    N = length(Ψ.state)
    L = length(Ψ.grad_states)
    Ψ_full = zeros(ComplexF64, N * (L + 1))
    convert_gradvec_to_dense!(Ψ_full, Ψ)
end


@inline function convert_gradvec_to_dense!(Ψ_full, Ψ)
    N = length(Ψ.state)
    L = length(Ψ.grad_states)
    @inbounds for l = 1:L
        Ψ_full[((l-1)*N+1):(l*N)] .= Ψ.grad_states[l]
    end
    @inbounds Ψ_full[(L*N+1):((L+1)*N)] .= Ψ.state
    return Ψ_full
end


@inline function convert_dense_to_gradvec!(Ψ, Ψ_full)
    N = length(Ψ.state)
    L = length(Ψ.grad_states)
    @inbounds for l = 1:L
        Ψ.grad_states[l] .= Ψ_full[((l-1)*N+1):(l*N)]
    end
    @inbounds Ψ.state .= Ψ_full[(L*N+1):((L+1)*N)]
    return Ψ
end


function Base.convert(::Type{Vector{ComplexF64}}, gradvec::GradVector)
    convert_gradvec_to_dense(gradvec)
end


function Base.convert(
    ::Type{GradVector{num_controls,T}},
    vec::AbstractVector
) where {num_controls,T}
    L = num_controls
    N = length(vec) ÷ (L + 1)  # dimension of state
    @assert length(vec) == (L + 1) * N
    grad_states = [convert(T, vec[((l-1)*N+1):(l*N)]) for l = 1:L]
    state = convert(T, vec[(L*N+1):((L+1)*N)])
    return GradVector{num_controls,T}(state, grad_states)
end

function Base.Array{T}(G::GradgenOperator) where {T}
    N, M = size(G.H)
    L = length(G.control_derivs)
    𝟘 = zeros(T, N, M)
    μ = G.control_derivs
    block_rows = [
        hcat([𝟘 for j = 1:(l-1)]..., Array{T}(G.H), [𝟘 for j = (l+1):L]..., Array{T}(μ[l])) for l = 1:L
    ]
    last_block_row = hcat([𝟘 for j = 1:L]..., Array{T}(G.H))
    return Base.Array{T}(vcat(block_rows..., last_block_row))
end


Base.Array(G::GradgenOperator) = Array{ComplexF64}(G)


function Base.convert(::Type{MT}, G::GradgenOperator) where {MT<:Matrix}
    Base.convert(MT, Base.Array(G))
end
