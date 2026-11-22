# SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT OR CC0-1.0

using QuantumGradientGenerators
using Documenter
using DocumenterCitations
using DocumenterInterLinks
using Pkg


PROJECT_TOML = Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
VERSION = PROJECT_TOML["version"]
NAME = PROJECT_TOML["name"]
AUTHORS = join(PROJECT_TOML["authors"], ", ") * " and contributors"
GITHUB = "https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl"

DEV_OR_STABLE = "stable/"
if endswith(VERSION, "dev")
    DEV_OR_STABLE = "dev/"
end

links = InterLinks(
    "Julia" => "https://docs.julialang.org/en/v1/",
    "QuantumPropagators" => "https://juliaquantumcontrol.github.io/QuantumPropagators.jl/$DEV_OR_STABLE",
    "QuantumControl" => "https://juliaquantumcontrol.github.io/QuantumControl.jl/$DEV_OR_STABLE",
    "GRAPE" => "https://juliaquantumcontrol.github.io/GRAPE.jl/$DEV_OR_STABLE",
)

fallbacks = ExternalFallbacks(
    "QuantumPropagators.Interfaces.supports_inplace" => "@extref QuantumPropagators :jl:function:`QuantumPropagators.Interfaces.supports_inplace`",
    automatic = false,
)

println("Starting makedocs")

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)

PAGES = [
    "Home" => "index.md",
    "Background" => "background.md",
    "API" => "api.md",
    "References" => "references.md",
]

makedocs(;
    plugins = [bib, links, fallbacks],
    authors = AUTHORS,
    sitename = "QuantumGradientGenerators.jl",
    doctest = false,
    format = Documenter.HTML(;
        prettyurls = true,
        canonical = "https://juliaquantumcontrol.github.io/QuantumGradientGenerators.jl",
        assets = [
            asset(
                "https://juliaquantumcontrol.github.io/QuantumControl.jl/dev/assets/topbar/topbar.css"
            ),
            asset(
                "https://juliaquantumcontrol.github.io/QuantumControl.jl/dev/assets/topbar/topbar.js"
            ),
        ],
        mathengine = MathJax3(
            Dict(
                :loader => Dict("load" => ["[tex]/physics"]),
                :tex => Dict(
                    "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                    "tags" => "ams",
                    "packages" =>
                        ["base", "ams", "autoload", "configmacros", "physics"],
                    "macros" => Dict("ii" => "{\\mathrm{i}}"),
                ),
            )
        ),
        footer = "[$NAME.jl]($GITHUB) v$VERSION docs powered by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl)."
    ),
    pages = PAGES,
)

println("Finished makedocs")

# The documentation is deployed by `docs/deploy.jl`, in a separate CI job
