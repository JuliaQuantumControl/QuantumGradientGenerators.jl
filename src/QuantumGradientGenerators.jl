# SPDX-FileCopyrightText: © 2022 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

module QuantumGradientGenerators

export GradGenerator, GradgenOperator, GradVector, resetgradvec!
export frozen_internalnorm, flat_internalnorm

include("grad_generator.jl")
include("gradgen_operator.jl")
include("grad_vector.jl")
include("evaluate.jl")
include("linalg.jl")
include("broadcast.jl")
include("spectral.jl")
include("internalnorm.jl")

end
