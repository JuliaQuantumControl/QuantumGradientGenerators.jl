import QuantumControlBase.QuantumPropagators.Controls: evaluate, evaluate!


function evaluate(O::GradgenOperator, args...; kwargs...)
    return O
end


function evaluate!(G::GradgenOperator, gradgen::GradGenerator, args...; vals_dict=IdDict())
    evaluate!(G.G, gradgen.G, args...; vals_dict)
    for (i, control) in enumerate(gradgen.controls)
        μ = gradgen.control_derivs[i]
        G.control_deriv_ops[i] = evaluate(μ, args...; vals_dict)
        # In most cases (for linear controls), evaluate(μ, ...) = μ
        # Hence, we're not using `copyto!`.
    end
    return G
end


function evaluate(gradgen::GradGenerator, args...; vals_dict=IdDict())
    G = evaluate(gradgen.G, args...; vals_dict)
    control_deriv_ops = [evaluate(μ, args...; vals_dict) for μ ∈ gradgen.control_derivs]
    num_controls = length(control_deriv_ops)
    GT = typeof(G)
    CGT = eltype(control_deriv_ops)
    GradgenOperator{num_controls,GT,CGT}(G, control_deriv_ops)
end
