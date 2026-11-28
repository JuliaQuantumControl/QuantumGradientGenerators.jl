# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

# Spectral properties of a gradient generator, evaluated from the dynamical
# generator `G.H` alone. The gradient generator is block upper-triangular with
# `G.H` in all diagonal blocks, so its characteristic polynomial is that of
# `G.H` to the power `K+1` for `K` derivatives, see the "Properties of the
# gradient generator" section of the Background page in the documentation.

import LinearAlgebra
import QuantumControl.QuantumPropagators.SpectralRange: specrange


_num_blocks(::GradgenOperator{num_derivs}) where {num_derivs} = num_derivs + 1


function specrange(G::GradgenOperator, method::Val{:auto}; kwargs...)
    return specrange(G.H, method; _specrange_kwargs(kwargs)...)
end

function specrange(G::GradgenOperator, method::Val{:diag}; kwargs...)
    return specrange(G.H, method; _specrange_kwargs(kwargs)...)
end

function specrange(G::GradgenOperator, method::Val{:arnoldi}; kwargs...)
    return specrange(G.H, method; _specrange_kwargs(kwargs)...)
end

# A `GradVector` given as the starting vector for the Arnoldi iteration
# contributes its state
function _specrange_kwargs(kwargs)
    if haskey(kwargs, :state) && (kwargs[:state] isa GradVector)
        return (; kwargs..., state = kwargs[:state].state)
    end
    return kwargs
end


function LinearAlgebra.eigvals(G::GradgenOperator)
    return repeat(LinearAlgebra.eigvals(Array(G.H)); inner = _num_blocks(G))
end


LinearAlgebra.tr(G::GradgenOperator) = _num_blocks(G) * LinearAlgebra.tr(G.H)


LinearAlgebra.det(G::GradgenOperator) = LinearAlgebra.det(G.H)^_num_blocks(G)
