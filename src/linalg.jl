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


function Base.length(Ψ::GradVector)
    return length(Ψ.state) * (1 + length(Ψ.grad_states))
end


function Base.size(Ψ::GradVector{num_controls,T}) where {num_controls,T}
    return ((num_controls + 1) * length(Ψ.state),)
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


function Base.similar(Ψ::GradVector{num_controls,T}) where {num_controls,T}
    state_sim = similar(Ψ.state)
    grad_states_sim = [similar(ϕ) for ϕ ∈ Ψ.grad_states]
    return GradVector{num_controls,typeof(state_sim)}(state_sim, grad_states_sim)
end

Base.similar(Ψ::GradVector, ::Type{S}) where {S} = Vector{S}(undef, length(Ψ))

Base.similar(Ψ::GradVector, dims::Tuple{Vararg{Int}}) = Array{eltype(Ψ)}(undef, dims)

# These definitions of `similar` exist to make ExponentialUtilities happy, but
# it's not clear at all that `similar` with a custom shape really makes sense
Base.similar(::GradVector, ::Type{T}, dims::Tuple{Int,Int}) where {T} =
    Matrix{T}(undef, dims...)

Base.similar(::GradVector, ::Type{T}, dims::Tuple{Int}) where {T} =
    Vector{T}(undef, dims[1])

function Base.getindex(Ψ::GradVector{num_controls,T}, k::Int) where {num_controls,T}
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

function Base.setindex!(Ψ::GradVector{num_controls,T}, v, k::Int) where {num_controls,T}
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

function Base.iterate(Ψ::GradVector, k = 1)
    k > length(Ψ) && return nothing
    return (Ψ[k], k + 1)
end

# As for an `Operator`, we implement `similar` to return a standard `Array`
# because `GradgenOperator` does not `setindex!`, so it's arguable not a
# "mutable array"even if its components are mutable.
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

function Base.getindex(
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

Base.length(O::GradgenOperator) = prod(size(O))

function Base.iterate(O::GradgenOperator, k = 1)
    n = length(O)
    k > n && return nothing
    n_rows = size(O, 1)
    i = (k - 1) % n_rows + 1
    j = (k - 1) ÷ n_rows + 1
    return (O[i, j], k + 1)
end

function Base.eltype(::Type{GradVector{num_controls,T}}) where {num_controls,T}
    return eltype(T)
end

function Base.copyto!(dest::GradgenOperator, src::GradgenOperator)
    copyto!(dest.G, src.G)
    copyto!(dest.control_deriv_ops, src.control_deriv_ops)
end


function Base.fill!(Ψ::GradVector, v)
    Base.fill!(Ψ.state, v)
    for i = 1:length(Ψ.grad_states)
        Base.fill!(Ψ.grad_states[i], v)
    end
    return Ψ
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
