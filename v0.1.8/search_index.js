var documenterSearchIndex = {"docs":
[{"location":"api/#QuantumGradientGenerators-API","page":"API","title":"QuantumGradientGenerators API","text":"","category":"section"},{"location":"api/#Index","page":"API","title":"Index","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"","category":"page"},{"location":"api/#Reference","page":"API","title":"Reference","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [QuantumGradientGenerators]","category":"page"},{"location":"api/#QuantumGradientGenerators.GradGenerator","page":"API","title":"QuantumGradientGenerators.GradGenerator","text":"Extended generator for the standard dynamic gradient.\n\nG̃ = GradGenerator(G)\n\ncontains the original time-dependent generator G (a Hamiltonian or Liouvillian) in G̃.G, a vector of control derivatives Gϵₗ(t) in G̃.control_derivs, and the controls in G̃.controls.\n\nFor a generator G = H(t) = H₀ + ϵ₁(t) H₁ +  +  ϵₙ(t) Hₙ, this extended generator encodes the block-matrix\n\nG = beginpmatrix\n         H(t)    0      dots     0       H₁     \n         0       H(t)   dots     0       H₂     \n    vdots              ddots            vdots \n         0       0      dots     H(t)    Hₙ     \n         0       0      dots     0       H(t)\nendpmatrix\n\nNote that the Gϵₗ(t) (Hₗ in the above example) may be time-dependent, to account for the possibility of non-linear control terms.\n\n\n\n\n\n","category":"type"},{"location":"api/#QuantumGradientGenerators.GradVector","page":"API","title":"QuantumGradientGenerators.GradVector","text":"Extended state-vector for the dynamic gradient.\n\nΨ̃ = GradVector(Ψ, num_controls)\n\nfor an initial state Ψ and num_controls control fields.\n\nThe GradVector conceptually corresponds to a direct-sum (block) column-vector Ψ = (Ψ₁ Ψ₂  Ψₙ Ψ)^T, where n is num_controls. With a matching G as in the documentation of GradGenerator, we have\n\nG Ψ = beginpmatrix\nH Ψ₁ + H₁Ψ \nvdots \nH Ψₙ + HₙΨ \nH Ψ\nendpmatrix\n\nand\n\ne^-i G dt beginpmatrix 0  vdots  0  Ψ endpmatrix\n= beginpmatrix\nfracϵ₁ e^-i H dt Ψ \nvdots \nfracϵₙ e^-i H dt Ψ \ne^-i H dt Ψ\nendpmatrix\n\nUpon initialization, Ψ₁Ψₙ are zero.\n\n\n\n\n\n","category":"type"},{"location":"api/#QuantumGradientGenerators.GradgenOperator","page":"API","title":"QuantumGradientGenerators.GradgenOperator","text":"Static generator for the dynamic gradient.\n\nusing QuantumPropagators.Controls: evaluate\n\nG::GradgenOperator = evaluate(gradgen::GradGenerator; vals_dict)\n\nis the result of plugging in specific values for all controls in a GradGenerator.\n\nThe resulting object can be multiplied directly with a GradVector, e.g., in the process of evaluating a piecewise-constant time propagation.\n\n\n\n\n\n","category":"type"},{"location":"api/#QuantumGradientGenerators.resetgradvec!-Tuple{GradVector}","page":"API","title":"QuantumGradientGenerators.resetgradvec!","text":"Reset the given gradient vector for a new gradient evaluation.\n\nresetgradvec!(Ψ̃::GradVector)\n\nzeroes out Ψ̃.grad_states but leaves Ψ̃.state unaffected. This is possible whether or not Ψ̃ supports in-place operations (QuantumPropagators.Interfaces.supports_inplace)\n\nresetgradvec!(Ψ̃::GradVector, Ψ)\n\nadditionally sets Ψ̃.state to Ψ, which requires that Ψ̃.state supports in-place operations.\n\nReturns Ψ̃.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = QuantumGradientGenerators","category":"page"},{"location":"#QuantumGradientGenerators","page":"Home","title":"QuantumGradientGenerators","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using Markdown\nusing Pkg\n\nVERSION = Pkg.dependencies()[Base.UUID(\"a563f35e-61db-434d-8c01-8b9e3ccdfd85\")].version\n\ngithub_badge = \"[![Github](https://img.shields.io/badge/JuliaQuantumControl-QuantumGradientGenerators.jl-blue.svg?logo=github)](https://github.com/JuliaQuantumControl/QuantumGradientGenerators.jl)\"\n\nversion_badge = \"![v$VERSION](https://img.shields.io/badge/version-v$VERSION-green.svg)\"\n\nMarkdown.parse(\"$github_badge $version_badge\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"Documentation for QuantumGradientGenerators.","category":"page"},{"location":"#Contents","page":"Home","title":"Contents","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"api.md\"]\nDepth = 2","category":"page"},{"location":"#History","page":"Home","title":"History","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"See the Releases on Github.","category":"page"}]
}