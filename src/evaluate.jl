# SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

import QuantumControl.QuantumPropagators.Controls: evaluate, evaluate!


function evaluate(G::GradgenOperator, args...; kwargs...)
    return G
end


function evaluate!(G::GradgenOperator, G_of_t::GradGenerator, args...; vals_dict = IdDict())
    evaluate!(G.H, G_of_t.H, args...; vals_dict)
    for (k, μ) in enumerate(G_of_t.derivs)
        G.derivs[k] = evaluate(μ, args...; vals_dict)
        # In most cases (for linear controls), evaluate(μ, ...) = μ
        # Hence, we're not using `copyto!`. A `nothing` (zero block) evaluates
        # to `nothing`.
    end
    return G
end


function evaluate(G_of_t::GradGenerator, args...; vals_dict = IdDict())
    H = evaluate(G_of_t.H, args...; vals_dict)
    derivs = [evaluate(μ, args...; vals_dict) for μ ∈ G_of_t.derivs]
    return GradgenOperator(H, derivs)
end
