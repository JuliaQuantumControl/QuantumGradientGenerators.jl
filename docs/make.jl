using QuantumGradientGenerators
using Documenter

DocMeta.setdocmeta!(
    QuantumGradientGenerators,
    :DocTestSetup,
    :(using QuantumGradientGenerators);
    recursive=true
)

makedocs(;
    modules=[QuantumGradientGenerators],
    authors="Michael Goerz <mail@michaelgoerz.net",
    repo="https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl/blob/{commit}{path}#{line}",
    sitename="QuantumGradientGenerators.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaQuantumControl.github.io/QuantumGradientGenerators.jl",
        edit_link="master",
        assets=String[]
    ),
    pages=["Home" => "index.md",]
)

deploydocs(;
    repo="github.com/JuliaQuantumControl/QuantumGradientGenerators.jl",
    devbranch="master"
)
