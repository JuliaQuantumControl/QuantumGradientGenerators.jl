import LinearAlgebra
import Base: -, *


function LinearAlgebra.mul!(Φ::GradVector, G::GradgenOperator, Ψ::GradVector)
    LinearAlgebra.mul!(Φ.state, G.G, Ψ.state)
    for i = 1:length(Ψ.grad_states)
        LinearAlgebra.mul!(Φ.grad_states[i], G.G, Ψ.grad_states[i])
        LinearAlgebra.mul!(Φ.grad_states[i], G.control_deriv_ops[i], Ψ.state, 1, 1)
    end
end


function LinearAlgebra.lmul!(c, Ψ::GradVector)
    LinearAlgebra.lmul!(c, Ψ.state)
    for i ∈ eachindex(Ψ.grad_states)
        LinearAlgebra.lmul!(c, Ψ.grad_states[i])
    end
end


function LinearAlgebra.axpy!(a, X::GradVector, Y::GradVector)
    LinearAlgebra.axpy!(a, X.state, Y.state)
    for i ∈ eachindex(X.grad_states)
        LinearAlgebra.axpy!(a, X.grad_states[i], Y.grad_states[i])
    end
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


function Base.copy(Ψ::GradVector)
    Φ = GradVector(Ψ.state, length(Ψ.grad_states))
    for i = 1:length(Ψ.grad_states)
        copyto!(Φ.grad_states[i], Ψ.grad_states[i])
    end
    return Φ
end


function Base.length(Ψ::GradVector)
    return length(Ψ.state) * (1 + length(Ψ.grad_states))
end


function Base.size(G::GradgenOperator)
    return Base.size(G.G)
end


function Base.similar(Ψ::GradVector{num_controls,T}) where {num_controls,T}
    return GradVector{num_controls,T}(similar(Ψ.state), [similar(ϕ) for ϕ ∈ Ψ.grad_states])
end

function Base.similar(G::GradgenOperator{num_controls,GT,CGT}) where {num_controls,GT,CGT}
    return GradgenOperator{num_controls,GT,CGT}(similar(G.G), similar(G.control_deriv_ops))
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
end


function -(Ψ::GradVector, Φ::GradVector)
    res = copy(Ψ)
    LinearAlgebra.axpy!(-1, Φ.state, res.state)
    for i = 1:length(Ψ.grad_states)
        LinearAlgebra.axpy!(-1, Φ.grad_states[i], res.grad_states[i])
    end
    return res
end


function *(G::GradgenOperator{num_controls,GT,CGT}, α::Number) where {num_controls,GT,CGT}
    GradgenOperator{num_controls,GT,CGT}(G.G * α, [CG * α for CG in G.control_deriv_ops])
end


*(α::Number, G::GradgenOperator) = *(G::GradgenOperator, α::Number)


@inline function convert_gradgen_to_dense(G)
    N = size(G.G)[1]
    L = length(G.control_derivs)
    G_full = zeros(eltype(G.G), N * (L + 1), N * (L + 1))
    convert_gradgen_to_dense!(G_full, G)
end


@inline function convert_gradgen_to_dense!(G_full, G)
    N = size(G.G)[1]
    L = length(G.control_derivs)
    @inbounds for i = 1:L+1
        G_full[(i-1)*N+1:i*N, (i-1)*N+1:i*N] .= G.G
    end
    # Set the control-derivatives in the last (block-)column
    @inbounds for i = 1:L
        G_full[(i-1)*N+1:i*N, L*N+1:(L+1)*N] .= G.control_derivs[i]
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
        Ψ_full[(i-1)*N+1:i*N] .= Ψ.grad_states[i]
    end
    @inbounds Ψ_full[L*N+1:(L+1)*N] .= Ψ.state
    return Ψ_full
end


@inline function convert_dense_to_gradvec!(Ψ, Ψ_full)
    N = length(Ψ.state)
    L = length(Ψ.grad_states)
    @inbounds for i = 1:L
        Ψ.grad_states[i] .= Ψ_full[(i-1)*N+1:i*N]
    end
    @inbounds Ψ.state .= Ψ_full[L*N+1:(L+1)*N]
    return Ψ
end


function Base.convert(::Type{Vector{ComplexF64}}, gradvec::GradVector)
    convert_gradvec_to_dense(gradvec)
end


function Base.convert(::Type{GradVector{num_controls,T}}, vec::T) where {num_controls,T}
    L = num_controls
    N = length(vec) ÷ (L + 1)  # dimension of state
    @assert length(vec) == (L + 1) * N
    grad_states = [vec[(i-1)*N+1:i*N] for i = 1:L]
    state = vec[L*N+1:(L+1)*N]
    return GradVector{num_controls,T}(state, grad_states)
end


function Base.Array{T}(G::GradgenOperator) where {T}
    N, M = size(G.G)
    L = length(G.control_deriv_ops)
    𝟘 = zeros(T, N, M)
    μ = G.control_deriv_ops
    block_rows = [
        hcat([𝟘 for j = 1:i-1]..., Array{T}(G.G), [𝟘 for j = i+1:L]..., Array{T}(μ[i]))
        for i = 1:L
    ]
    last_block_row = hcat([𝟘 for j = 1:L]..., Array{T}(G.G))
    return Base.Array{T}(vcat(block_rows..., last_block_row))
end


Base.Array(G::GradgenOperator) = Array{ComplexF64}(G)


function Base.convert(::Type{MT}, G::GradgenOperator) where {MT<:Matrix}
    Base.convert(MT, Base.Array(G))
end
