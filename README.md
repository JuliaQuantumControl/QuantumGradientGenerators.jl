<!--
SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>

SPDX-License-Identifier: MIT OR CC-BY-4.0
-->

# QuantumGradientGenerators

[![Version](https://juliahub.com/docs/General/QuantumGradientGenerators/stable/version.svg)](https://juliahub.com/ui/Packages/General/QuantumGradientGenerators)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaQuantumControl.github.io/QuantumGradientGenerators.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaQuantumControl.github.io/QuantumGradientGenerators.jl/dev/)
[![Build Status](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![REUSE](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/actions/workflows/REUSE.yml/badge.svg)](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/actions/workflows/REUSE.yml)
[![Coverage](https://codecov.io/gh/JuliaQuantumControl/QuantumGradientGenerators.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaQuantumControl/QuantumGradientGenerators.jl)

Dynamic Gradients for Quantum Control.

For a [dynamical generator](https://juliaquantumcontrol.github.io/QuantumPropagators.jl/stable/generators/) $\hat{H}$ (a Hamiltonian or Liouvillian) that depends on control functions $\epsilon_1(t), \dots, \epsilon_L(t)$, this package defines the "gradient generator" $G = G[\hat{H}]$ and the "gradient vector" $\Psi = (|\Psi^\prime_1⟩, \dots, |\Psi^\prime_L⟩, |\Psi⟩)^T$, with the state $|\Psi⟩$ and the gradient states $|\Psi^\prime_l⟩$, so that

```math
e^{-i G dt} \begin{pmatrix} 0 \\ \vdots \\ 0 \\ |\Psi⟩ \end{pmatrix}
= \begin{pmatrix}
    \frac{\partial}{\partial \epsilon_1} e^{-i \hat{H} dt} |\Psi⟩ \\
    \vdots \\
    \frac{\partial}{\partial \epsilon_L} e^{-i \hat{H} dt} |\Psi⟩ \\
    e^{-i \hat{H} dt} |\Psi⟩
\end{pmatrix}\,.
```

This gradient generator is a core part of evaluating gradients in the [GRAPE][] method of [quantum control][QuantumControl].

To evaluate the above equation, the [QuantumPropagators][] package should be used.


## Documentation

The documentation of `QuantumGradientGenerators.jl` is available at <https://juliaquantumcontrol.github.io/QuantumGradientGenerators.jl>.


## Installation

The `QuantumGradientGenerators` package can be installed with [Pkg][] as

~~~
pkg> add QuantumGradientGenerators
~~~

Note that this package is not intended for direct use. Instead, it serves as a building block for implementing quantum control methods, e.g. in the [GRAPE][] package.

For development usage within the [JuliaQuantumControl][] organization, see the [organization development notes](https://github.com/JuliaQuantumControl#development).


## License

This project is [REUSE-compliant](https://reuse.software). The entire package can be distributed under the [MIT License](LICENSE): every file is available under MIT. Files that are not source code are additionally dual-licensed for reuse in other contexts — documentation (including the `README` and other prose) under [Creative Commons Attribution (`CC-BY-4.0`)](https://creativecommons.org/licenses/by/4.0/deed.en), and trivial configuration or tooling files as public domain ([`CC0-1.0`](https://creativecommons.org/publicdomain/zero/1.0/deed.en)). The authoritative per-file copyright and licensing information is provided by SPDX tags in each file (and `REUSE.toml` where a file cannot carry a header). It can be verified using the [`reuse` tool](https://github.com/fsfe/reuse-tool?tab=readme-ov-file#reuse), e.g., by running `reuse lint`.


[JuliaQuantumControl]: https://github.com/JuliaQuantumControl
[QuantumControl]: https://github.com/JuliaQuantumControl/QuantumControl.jl#readme
[GRAPE]: https://github.com/JuliaQuantumControl/GRAPE.jl#readme
[QuantumPropagators]: https://github.com/JuliaQuantumControl/QuantumPropagators.jl#readme
[Pkg]: https://pkgdocs.julialang.org/v1/
