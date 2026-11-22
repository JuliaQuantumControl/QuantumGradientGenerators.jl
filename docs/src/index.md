```@meta
# SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>
# SPDX-License-Identifier: MIT OR CC-BY-4.0
CurrentModule = QuantumGradientGenerators
```

# QuantumGradientGenerators

```@eval
using Markdown
using Pkg

VERSION = Pkg.dependencies()[Base.UUID("a563f35e-61db-434d-8c01-8b9e3ccdfd85")].version

github_badge = "[![Github](https://img.shields.io/badge/JuliaQuantumControl-QuantumGradientGenerators.jl-blue.svg?logo=github)](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl)"

version_badge = "![v$VERSION](https://img.shields.io/badge/version-v$VERSION-green.svg)"

Markdown.parse("$github_badge $version_badge")
```

Documentation for [QuantumGradientGenerators](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl).

## Overview

Consider a dynamical generator ``\hat{H}`` (a Hamiltonian or a Liouvillian) that depends on the control functions ``ϵ_1(t), \dots, ϵ_L(t)``, with the control derivatives ``\hat{μ}_l = \partial \hat{H} / \partial ϵ_l``. Gradient-based quantum control methods like [GRAPE](https://github.com/JuliaQuantumControl/GRAPE.jl) require the derivative ``\frac{\partial}{\partial ϵ_l} e^{-i \hat{H} dt} |Ψ⟩`` of a single time step of a piecewise-constant time propagation.

The gradient generator ``G = G[\hat{H}]`` is a block operator with ``\hat{H}`` on the diagonal and the ``\hat{μ}_l`` in the last block column, see [`GradGenerator`](@ref). It acts on gradient vectors

```math
Ψ = \begin{pmatrix}
|Ψ^\prime_1⟩ \\ \vdots \\ |Ψ^\prime_L⟩ \\ |Ψ⟩
\end{pmatrix}\,,
```

see [`GradVector`](@ref), that contain the gradient states ``|Ψ^\prime_l⟩`` in addition to the state ``|Ψ⟩``. The gradient vector ``Ψ`` is written without a ket, to distinguish it from the state ``|Ψ⟩``. Propagating a gradient vector with zero gradient states under ``G`` for a single time step yields the derivatives of the propagated state with respect to all controls,

```math
e^{-i G dt} \begin{pmatrix} 0 \\ \vdots \\ 0 \\ |Ψ⟩ \end{pmatrix}
= \begin{pmatrix}
    \frac{\partial}{\partial ϵ_1} e^{-i \hat{H} dt} |Ψ⟩ \\
    \vdots \\
    \frac{\partial}{\partial ϵ_L} e^{-i \hat{H} dt} |Ψ⟩ \\
    e^{-i \hat{H} dt} |Ψ⟩
\end{pmatrix}\,.
```

Any of the methods in [QuantumPropagators](https://github.com/JuliaQuantumControl/QuantumPropagators.jl) can evaluate this propagation. See the [Background](@ref QGG-Background) for a proof, the interpretation of gradient vectors as dual numbers in forward-mode automatic differentiation, and the generalization to time-dependent generators and to parameterized pulses. For a backward propagation, as in GRAPE, construct the gradient generator ``G[\hat{H}^\dagger]`` from the adjoint generator ``\hat{H}^\dagger``. Its last block column then contains ``\hat{μ}^\dagger_l``.

In the code, a gradient generator `G` holds ``\hat{H}`` in `G.H` and ``\hat{μ}_l`` in `G.control_derivs[l]`, and a gradient vector `Ψ` holds ``|Ψ⟩`` in `Ψ.state` and ``|Ψ^\prime_l⟩`` in `Ψ.grad_states[l]`. A time-dependent [`GradGenerator`](@ref) evaluates to a static [`GradgenOperator`](@ref) for specific values of the controls.

## Contents

```@contents
Pages = ["howto.md", "background.md", "api.md", "references.md"]
Depth = 2
```

## History

See the [Releases](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/releases) on Github.
