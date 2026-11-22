# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

module QuantumGradientGeneratorsDiffEqBaseExt

import DiffEqBase: ODE_DEFAULT_NORM
import QuantumGradientGenerators: GradVector, frozen_internalnorm, flat_internalnorm


frozen_internalnorm(Ψ::GradVector, t) = ODE_DEFAULT_NORM(Ψ.state, t)
frozen_internalnorm(u::Number, t) = ODE_DEFAULT_NORM(u, t)


function flat_internalnorm(Ψ::GradVector, t)
    # The root mean square of all elements, from the root mean square of each
    # block: (Σ_blocks n_b · rms_b²) / (Σ_blocks n_b)
    n = length(Ψ.state)
    sum_abs2 = n * ODE_DEFAULT_NORM(Ψ.state, t)^2
    for Ψ′ in Ψ.grad_states
        n′ = length(Ψ′)
        sum_abs2 += n′ * ODE_DEFAULT_NORM(Ψ′, t)^2
        n += n′
    end
    return sqrt(sum_abs2 / max(n, 1))
end

flat_internalnorm(u::Number, t) = ODE_DEFAULT_NORM(u, t)


# Default: frozen semantics. Without this method, OrdinaryDiffEq would fall
# back to `norm(Ψ)`, the Euclidean norm, which unlike the root mean square
# scales with the dimension.
ODE_DEFAULT_NORM(Ψ::GradVector, t) = frozen_internalnorm(Ψ, t)

end
