# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

using Test
using LinearAlgebra: norm
using QuantumPropagators: hamiltonian, propagate, ExpProp
using QuantumGradientGenerators: GradGenerator, GradVector
import ExponentialUtilities


@testset "ExponentialUtilities (inplace=$inplace)" for inplace in (true, false)

    # The Krylov propagation of ExponentialUtilities allocates its result with
    # `similar(Ψ, T)`, which is a `GradVector`, and passes it to `expv!`
    Ĥ₀ = ComplexF64[0 0 0; 0 1.0 0; 0 0 2.1]
    Ĥ₁ = ComplexF64[0 1 0; 1 0 √2; 0 √2 0]
    ϵ(t) = 0.3 * sin(π * t / 5)^2
    G = GradGenerator(hamiltonian(Ĥ₀, (Ĥ₁, ϵ)))
    tlist = collect(range(0, 5; length = 51))
    Ψ₀ = ComplexF64[1, 0, 0]

    Ψ_exact = propagate(GradVector(Ψ₀, 1), G, tlist; method = ExpProp)
    Ψ = propagate(GradVector(Ψ₀, 1), G, tlist; method = ExponentialUtilities, inplace)
    @test Ψ isa GradVector{1,Vector{ComplexF64}}
    @test norm(Ψ.state - Ψ_exact.state) < 1e-12
    @test norm(Ψ.grad_states[1] - Ψ_exact.grad_states[1]) < 1e-12

end
