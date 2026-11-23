# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

module QuantumGradientGeneratorsExponentialUtilitiesExt

# The Krylov propagation of ExponentialUtilities builds its Krylov subspace from
# a `GradVector` via the vector interface (`eachindex`, `getindex`) and the
# flat `mul!` of a `GradgenOperator`, but `expv!` only accepts an
# `AbstractVector` as its output `w`. A `GradVector` is not an `AbstractVector`,
# and `similar(Ψ, T)` returns a `GradVector` (which ODE solvers rely on), so
# `expv(t, A, Ψ)` passes a `GradVector` as `w`. We compute the result in a flat
# `Vector`, with the gradient states first and the state last (the layout of
# `getindex` for a `GradVector`), and copy it into `w` block by block.

import ExponentialUtilities: expv!, KrylovSubspace
import QuantumGradientGenerators: GradVector, _copy_flat!


# This allocates the flat buffer in every call. The out-of-place `expv`
# allocates its result anyway. For in-place propagation, a propagator should
# keep a flat buffer as a workspace, call `expv!` for that buffer, and copy it
# into the gradient vector (`Ψ .= buffer`), which avoids this method.
function expv!(w::GradVector, t::Number, Ks::KrylovSubspace; kwargs...)
    w_flat = Vector{eltype(w)}(undef, length(w))
    expv!(w_flat, t, Ks; kwargs...)
    return _copy_flat!(w, w_flat)
end


end
