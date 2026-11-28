# SPDX-FileCopyrightText: © 2026 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

# ODE solvers (OrdinaryDiffEq) copy and fill their state with the "recursive"
# functions of RecursiveArrayTools

import RecursiveArrayTools: recursivecopy, recursivecopy!, recursivefill!

recursivecopy(Ψ::GradVector) = copy(Ψ)
recursivecopy!(dest::GradVector, src::GradVector) = copyto!(dest, src)
recursivefill!(Ψ::GradVector, v) = fill!(Ψ, v)
