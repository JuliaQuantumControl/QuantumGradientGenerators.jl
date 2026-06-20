<!--
SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>

SPDX-License-Identifier: MIT OR CC0-1.0
-->

# Release Notes

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Also see the [GitHub Releases](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/releases).

## [Unreleased]

## [v0.1.9] — 2026-06-20

* Added: `size` and `eltype` for gradient-generator operators [[#17], [#23]]
* Added: Matrix and vector interfaces for gradient-generator operators and vectors [[#18]]
* Added: An extension for [ArrayInterface.jl](https://github.com/JuliaArrays/ArrayInterface.jl) [[#19]]
* Added: A simplified constructor for `GradgenOperator` [[#20]]
* Added: Conversion between compatible `GradVector`s [[#21]]
* Added: Flat matrix-vector multiplication [[#22]]
* Added: Per-file copyright and licensing information following the [REUSE specification](https://reuse.software). Source code remains under the MIT License; documentation is additionally available under `CC-BY-4.0`, and trivial files under `CC0-1.0`.

## [v0.1.8] — 2024-09-04

* Removed: The dependency on `QuantumControlBase`; `QuantumGradientGenerators` is now self-contained

## [v0.1.7] — 2024-07-27

* Added: Support for the in-place interface

## [v0.1.6] — 2024-04-21

* Added: Support for `get_parameters`

## [v0.1.5] — 2024-01-22

* Changed: Compatibility with `QuantumControlBase` 0.9

## [v0.1.4] — 2024-01-08

* Added: An `rng` argument to `random_state`

## [v0.1.3] — 2023-10-06

* Changed: The minimum supported Julia version is now 1.9

## [v0.1.2] — 2023-05-15

* Fixed: The linalg interface

## [v0.1.1] — 2023-02-16

* Changed: `QuantumPropagators` is now an indirect dependency

## [v0.1.0] — 2023-01-28

Initial public release

[Unreleased]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/compare/v0.1.9..HEAD
[v0.1.9]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/releases/tag/v0.1.9
[v0.1.8]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/releases/tag/v0.1.8
[v0.1.7]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/releases/tag/v0.1.7
[v0.1.6]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/releases/tag/v0.1.6
[v0.1.5]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/releases/tag/v0.1.5
[v0.1.4]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/releases/tag/v0.1.4
[v0.1.3]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/releases/tag/v0.1.3
[v0.1.2]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/releases/tag/v0.1.2
[v0.1.1]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/releases/tag/v0.1.1
[v0.1.0]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/releases/tag/v0.1.0
[#17]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/pull/17
[#18]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/pull/18
[#19]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/pull/19
[#20]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/pull/20
[#21]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/pull/21
[#22]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/pull/22
[#23]: https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/pull/23
