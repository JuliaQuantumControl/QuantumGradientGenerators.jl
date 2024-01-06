using QuantumGradientGenerators
using Documenter
using DocumenterInterLinks
using Pkg

DocMeta.setdocmeta!(
    QuantumGradientGenerators,
    :DocTestSetup,
    :(using QuantumGradientGenerators);
    recursive=true
)

PROJECT_TOML = Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
VERSION = PROJECT_TOML["version"]
NAME = PROJECT_TOML["name"]
AUTHORS = join(PROJECT_TOML["authors"], ", ") * " and contributors"
GITHUB = "https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl"

println("Starting makedocs")

makedocs(;
    authors=AUTHORS,
    sitename="QuantumGradientGenerators.jl",
    modules=[QuantumGradientGenerators],
    repo="https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/blob/{commit}{path}#{line}",
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

deploydocs(;
    repo="github.com/JuliaQuantumControl/QuantumGradientGenerators.jl",
    devbranch="master"
)
