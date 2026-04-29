import LinearAlgebra
import Base: +, -, *


function LinearAlgebra.mul!(Φ::GradVector, G::GradgenOperator, Ψ::GradVector, α, β)
    LinearAlgebra.mul!(Φ.state, G.G, Ψ.state, α, β)
    for i = 1:length(Ψ.grad_states)
        LinearAlgebra.mul!(Φ.grad_states[i], G.G, Ψ.grad_states[i], α, β)
        LinearAlgebra.mul!(Φ.grad_states[i], G.control_deriv_ops[i], Ψ.state, α, 1)
    end
    return Φ
end


function LinearAlgebra.mul!(Φ::GradVector, G::GradgenOperator, Ψ::GradVector)
    return LinearAlgebra.mul!(Φ, G, Ψ, true, false)
end


# Flat-vector dispatch used by ExponentialUtilities' Arnoldi iteration, which
# builds its Krylov matrix via similar(b, T, (n, m+1)) → plain Matrix{T} and
# then calls mul!(view(V,:,j+1), A, view(V,:,j)). The GradVector is treated as
# a packed flat vector: blocks [grad_1 | grad_2 | … | state], each of length N.
function LinearAlgebra.mul!(
    Φ::AbstractVector,
    G::GradgenOperator{num_controls,GT,CGT},
    Ψ::AbstractVector,
    α,
    β
) where {num_controls,GT,CGT}
    N = size(G.G, 1)
    L = num_controls
    Ψ_state = view(Ψ, (L*N+1):((L+1)*N))
    for i = 1:L
        Φ_grad_i = view(Φ, ((i-1)*N+1):(i*N))
        Ψ_grad_i = view(Ψ, ((i-1)*N+1):(i*N))
        LinearAlgebra.mul!(Φ_grad_i, G.G, Ψ_grad_i, α, β)
        LinearAlgebra.mul!(Φ_grad_i, G.control_deriv_ops[i], Ψ_state, α, 1)
    end
    Φ_state = view(Φ, (L*N+1):((L+1)*N))
    LinearAlgebra.mul!(Φ_state, G.G, Ψ_state, α, β)
    return Φ
end


function LinearAlgebra.mul!(Φ::AbstractVector, G::GradgenOperator, Ψ::AbstractVector)
    return LinearAlgebra.mul!(Φ, G, Ψ, true, false)
end


function LinearAlgebra.lmul!(c, Ψ::GradVector)
    LinearAlgebra.lmul!(c, Ψ.state)
    for i ∈ eachindex(Ψ.grad_states)
        LinearAlgebra.lmul!(c, Ψ.grad_states[i])
    end
    return Ψ
end


function LinearAlgebra.axpy!(a, X::GradVector, Y::GradVector)
    LinearAlgebra.axpy!(a, X.state, Y.state)
    for i ∈ eachindex(X.grad_states)
        LinearAlgebra.axpy!(a, X.grad_states[i], Y.grad_states[i])
    end
    return Y
end


function LinearAlgebra.norm(Ψ::GradVector)
    nrm = LinearAlgebra.norm(Ψ.state)
    for i ∈ eachindex(Ψ.grad_states)
        nrm += LinearAlgebra.norm(Ψ.grad_states[i])
    end
    return nrm
end


function LinearAlgebra.dot(Ψ::GradVector, Φ::GradVector)
    c::ComplexF64 = LinearAlgebra.dot(Ψ.state, Φ.state)
    for i ∈ eachindex(Ψ.grad_states)
        c += LinearAlgebra.dot(Ψ.grad_states[i], Φ.grad_states[i])
    end
    return c
end


function LinearAlgebra.dot(Ψ::GradVector, G::GradgenOperator, Φ::GradVector)
    return LinearAlgebra.dot(Ψ, G * Φ)
end


LinearAlgebra.ishermitian(G::GradgenOperator) = false


function Base.isreal(G::GradgenOperator)
    return (isreal(G.G) && all(isreal(D for D in G.control_deriv_ops)))
end


function Base.copyto!(dest::GradVector, src::GradVector)
    copyto!(dest.state, src.state)
    for i = 1:length(src.grad_states)
        copyto!(dest.grad_states[i], src.grad_states[i])
    end
    return dest
end


function Base.copy(Ψ::GradVector{num_controls,T}) where {num_controls,T}
    return GradVector{num_controls,T}(copy(Ψ.state), [copy(ϕ) for ϕ in Ψ.grad_states])
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


function Base.similar(Ψ::GradVector{num_controls,T}) where {num_controls,T}
    state_sim = similar(Ψ.state)
    grad_states_sim = [similar(ϕ) for ϕ ∈ Ψ.grad_states]
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
    for i = 1:length(Ψ.grad_states)
        Base.fill!(Ψ.grad_states[i], v)
    end
    return Ψ
end


function Base.size(O::GradgenOperator{num_controls,GT,CGT}) where {num_controls,GT,CGT}
    return (num_controls + 1) .* size(O.G)
end

function Base.size(
    O::GradgenOperator{num_controls,GT,CGT},
    dim::Integer
) where {num_controls,GT,CGT}
    return (num_controls + 1) * size(O.G, dim)
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

Base.similar(O::GradgenOperator, ::Type{S}) where {S} = Array{S}(undef, size(O))
Base.similar(O::GradgenOperator, dims::Tuple{Vararg{Int}}) = Array{eltype(O)}(undef, dims)
Base.similar(O::GradgenOperator, ::Type{S}, dims::Tuple{Vararg{Int}}) where {S} =
    Array{S}(undef, dims)


function Base.eltype(
    ::Type{GradgenOperator{num_controls,GT,CGT}}
) where {num_controls,GT,CGT}
    return promote_type(eltype(GT), eltype(CGT))
end


function _getindex(
    ::Val{true},
    O::GradgenOperator{num_controls,GT,CGT},
    row::Int,
    col::Int
) where {num_controls,GT,CGT}
    T = eltype(O)
    N, M = size(O.G)
    L = num_controls
    block_row = (row - 1) ÷ N + 1
    block_col = (col - 1) ÷ M + 1
    local_row = (row - 1) % N + 1
    local_col = (col - 1) % M + 1
    if block_row == block_col
        return convert(T, O.G[local_row, local_col])
    elseif block_col == L + 1 && block_row <= L
        return convert(T, O.control_deriv_ops[block_row][local_row, local_col])
    else
        return zero(T)
    end
end

function _getindex(::Val{false}, O::GradgenOperator, row::Int, col::Int)
    error("$(typeof(O)) does not support the matrix interface")
end

function Base.getindex(O::T, row::Int, col::Int) where {T<:GradgenOperator}
    return _getindex(Val(supports_matrix_interface(T)), O, row, col)
end


function _length(::Val{true}, O::GradgenOperator)
    return prod(size(O))
end

function _length(::Val{false}, O::GradgenOperator)
    error("$(typeof(O)) does not support the matrix interface")
end

function Base.length(O::T) where {T<:GradgenOperator}
    return _length(Val(supports_matrix_interface(T)), O)
end


function _iterate(::Val{true}, O::GradgenOperator, k)
    n = length(O)
    k > n && return nothing
    n_rows = size(O, 1)
    i = (k - 1) % n_rows + 1
    j = (k - 1) ÷ n_rows + 1
    return (O[i, j], k + 1)
end

function _iterate(::Val{false}, O::GradgenOperator, k)
    error("$(typeof(O)) does not support the matrix interface")
end

function Base.iterate(O::T, k = 1) where {T<:GradgenOperator}
    return _iterate(Val(supports_matrix_interface(T)), O, k)
end


function Base.eltype(::Type{GradVector{num_controls,T}}) where {num_controls,T}
    return eltype(T)
end

function Base.copyto!(dest::GradgenOperator, src::GradgenOperator)
    copyto!(dest.G, src.G)
    copyto!(dest.control_deriv_ops, src.control_deriv_ops)
end


function Base.zero(Ψ::GradVector{num_controls,T}) where {num_controls,T}
    return GradVector{num_controls,T}(zero(Ψ.state), [zero(ϕ) for ϕ ∈ Ψ.grad_states])
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


function *(α::Number, v::GradVector{num_controls,T}) where {num_controls,T}
    return GradVector{num_controls,T}(α * v.state, [(α * ϕ) for ϕ in v.grad_states])
end


function *(v::GradVector{num_controls,T}, α::Number) where {num_controls,T}
    return α * v
end


function *(G::GradgenOperator{num_controls,GT,CGT}, α::Number) where {num_controls,GT,CGT}
    GradgenOperator{num_controls,GT,CGT}(G.G * α, [CG * α for CG in G.control_deriv_ops])
end

*(α::Number, G::GradgenOperator) = *(G::GradgenOperator, α::Number)


function *(
    G::GradgenOperator{num_controls,GT,CGT},
    Ψ::GradVector{num_controls,ST}
) where {num_controls,GT,CGT,ST}
    state = G.G * Ψ.state
    grad_states = [G.G * ϕ for ϕ in Ψ.grad_states]
    for (i, Hₙ) in enumerate(G.control_deriv_ops)
        grad_states[i] += Hₙ * Ψ.state
    end
    return GradVector{num_controls,ST}(state, grad_states)
end


@inline function convert_gradgen_to_dense(G::GradGenerator)
    N = size(G.G)[1]
    L = length(G.control_derivs)
    G_full = zeros(eltype(G.G), N * (L + 1), N * (L + 1))
    convert_gradgen_to_dense!(G_full, G)
end


@inline function convert_gradgen_to_dense!(G_full, G::GradGenerator)
    N = size(G.G)[1]
    L = length(G.control_derivs)
    @inbounds for i = 1:(L+1)
        G_full[((i-1)*N+1):(i*N), ((i-1)*N+1):(i*N)] .= G.G
    end
    # Set the control-derivatives in the last (block-)column
    @inbounds for i = 1:L
        G_full[((i-1)*N+1):(i*N), (L*N+1):((L+1)*N)] .= G.control_derivs[i]
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
    @inbounds for i = 1:L
        Ψ_full[((i-1)*N+1):(i*N)] .= Ψ.grad_states[i]
    end
    @inbounds Ψ_full[(L*N+1):((L+1)*N)] .= Ψ.state
    return Ψ_full
end


@inline function convert_dense_to_gradvec!(Ψ, Ψ_full)
    N = length(Ψ.state)
    L = length(Ψ.grad_states)
    @inbounds for i = 1:L
        Ψ.grad_states[i] .= Ψ_full[((i-1)*N+1):(i*N)]
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
    grad_states = [convert(T, vec[((i-1)*N+1):(i*N)]) for i = 1:L]
    state = convert(T, vec[(L*N+1):((L+1)*N)])
    return GradVector{num_controls,T}(state, grad_states)
end

function Base.Array{T}(G::GradgenOperator) where {T}
    N, M = size(G.G)
    L = length(G.control_deriv_ops)
    𝟘 = zeros(T, N, M)
    μ = G.control_deriv_ops
    block_rows = [
        hcat([𝟘 for j = 1:(i-1)]..., Array{T}(G.G), [𝟘 for j = (i+1):L]..., Array{T}(μ[i])) for i = 1:L
    ]
    last_block_row = hcat([𝟘 for j = 1:L]..., Array{T}(G.G))
    return Base.Array{T}(vcat(block_rows..., last_block_row))
end


Base.Array(G::GradgenOperator) = Array{ComplexF64}(G)


function Base.convert(::Type{MT}, G::GradgenOperator) where {MT<:Matrix}
    Base.convert(MT, Base.Array(G))
end
