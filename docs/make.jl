using QuantumGradientGenerators
using Documenter
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
)

println("Starting makedocs")

makedocs(;
    plugins=[links],
    authors=AUTHORS,
    sitename="QuantumGradientGenerators.jl",
    doctest=false,
    format=Documenter.HTML(;
        prettyurls=true,
        canonical="https://juliaquantumcontrol.github.io/QuantumGradientGenerators.jl",
        assets=[
            asset(
                "https://juliaquantumcontrol.github.io/QuantumControl.jl/dev/assets/topbar/topbar.css"
            ),
            asset(
                "https://juliaquantumcontrol.github.io/QuantumControl.jl/dev/assets/topbar/topbar.js"
            ),
        ],
        footer="[$NAME.jl]($GITHUB) v$VERSION docs powered by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl)."
    ),
    pages=["Home" => "index.md", "API" => "api.md",]
)

println("Finished makedocs")

deploydocs(; repo="github.com/JuliaQuantumControl/QuantumGradientGenerators.jl",)
