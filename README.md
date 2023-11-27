# QuantumGradientGenerators

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaQuantumControl.github.io/QuantumGradientGenerators.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaQuantumControl.github.io/QuantumGradientGenerators.jl/dev/)
[![Build Status](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/JuliaQuantumControl/QuantumGradientGenerators.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaQuantumControl/QuantumGradientGenerators.jl)

Dynamic Gradients for Quantum Control.

For a [dynamical generator](https://juliaquantumcontrol.github.io/QuantumPropagators.jl/stable/generators/) `G` that depends on one more control function `ϵ₁(t)` … `ϵₗ(t)`, this package defines the "gradient generator" `G̃` and "gradient vector" `|Ψ̃⟩` derived from a quantum state `|Ψ⟩`, so that

```math
\exp(± i \tilde{G} t) |\tilde{Ψ}⟩ = \begin{pmatrix}
    \frac{∂}{∂ϵ_1} e^{± i G t} |Ψ⟩ \\
    \vdots \\
    \frac{∂}{∂ϵ_n} e^{± i G t} |Ψ⟩ \\
    e^{± i G dt} |Ψ⟩
\end{pmatrix}\,.
```

This gradient generator is a core part of evaluating gradients in the [GRAPE][] method of [quantum control][QuantumControl].

To evaluate the above equation, the [QuantumPropagators][] package should be used.


## Documentation

The documentation of `QuantumGradientGenerators.jl` is available at <https://juliaquantumcontrol.github.io/QuantumPropagators.jl>.


## Installation

The `QuantumGradientGenerators` package can be installed with [Pkg][] as

~~~
pkg> add QuantumGradientGenerators
~~~

Note that this package is not intended for direct use. Instead, it serves as a building block for implementing quantum control methods, e.g. in the [GRAPE][] package.

For development usage within the [JuliaQuantumControl][] organization, see the [organization development notes](https://github.com/JuliaQuantumControl#development).


[JuliaQuantumControl]: https://github.com/JuliaQuantumControl
[QuantumControl]: https://github.com/JuliaQuantumControl/QuantumControl.jl#readme
[GRAPE]: https://github.com/JuliaQuantumControl/GRAPE.jl#readme
[QuantumPropagators]: https://github.com/JuliaQuantumControl/QuantumPropagators.jl#readme
[Pkg]: https://pkgdocs.julialang.org/v1/
