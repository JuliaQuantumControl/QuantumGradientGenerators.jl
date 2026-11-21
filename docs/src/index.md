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

Consider a dynamical generator ``\hat{H}`` (a Hamiltonian or a Liouvillian) that depends on the control functions ``\epsilon_1(t), \dots, \epsilon_L(t)``, with the control derivatives ``\hat{\mu}_l = \partial \hat{H} / \partial \epsilon_l``. Gradient-based quantum control methods like [GRAPE](https://github.com/JuliaQuantumControl/GRAPE.jl) require the derivative ``\frac{\partial}{\partial \epsilon_l} e^{-i \hat{H} dt} |\Psi⟩`` of a single time step of a piecewise-constant time propagation.

The gradient generator ``G = G[\hat{H}]`` is a block operator with ``\hat{H}`` on the diagonal and the ``\hat{\mu}_l`` in the last block column, see [`GradGenerator`](@ref). It acts on gradient vectors

```math
\Psi = \begin{pmatrix}
|\Psi^\prime_1⟩ \\ \vdots \\ |\Psi^\prime_L⟩ \\ |\Psi⟩
\end{pmatrix}\,,
```

see [`GradVector`](@ref), that contain the gradient states ``|\Psi^\prime_l⟩`` in addition to the state ``|\Psi⟩``. The gradient vector ``\Psi`` is written without a ket, to distinguish it from the state ``|\Psi⟩``. Propagating a gradient vector with zero gradient states under ``G`` for a single time step yields the derivatives of the propagated state with respect to all controls,

```math
e^{-i G dt} \begin{pmatrix} 0 \\ \vdots \\ 0 \\ |\Psi⟩ \end{pmatrix}
= \begin{pmatrix}
    \frac{\partial}{\partial \epsilon_1} e^{-i \hat{H} dt} |\Psi⟩ \\
    \vdots \\
    \frac{\partial}{\partial \epsilon_L} e^{-i \hat{H} dt} |\Psi⟩ \\
    e^{-i \hat{H} dt} |\Psi⟩
\end{pmatrix}\,.
```

Any of the methods in [QuantumPropagators](https://github.com/JuliaQuantumControl/QuantumPropagators.jl) can evaluate this propagation. For a backward propagation, as in GRAPE, construct the gradient generator ``G[\hat{H}^\dagger]`` from the adjoint generator ``\hat{H}^\dagger``. Its last block column then contains ``\hat{\mu}^\dagger_l``.

In the code, a gradient generator `G` holds ``\hat{H}`` in `G.H` and ``\hat{\mu}_l`` in `G.control_derivs[l]`, and a gradient vector `Ψ` holds ``|\Psi⟩`` in `Ψ.state` and ``|\Psi^\prime_l⟩`` in `Ψ.grad_states[l]`. A time-dependent [`GradGenerator`](@ref) evaluates to a static [`GradgenOperator`](@ref) for specific values of the controls.

## Contents

```@contents
Pages = ["api.md"]
Depth = 2
```

## History

See the [Releases](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/releases) on Github.
