#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module FMISensitivity

# load modules for reusability in other frameworks
import SciMLSensitivity
import SciMLSensitivity: ForwardDiff
import SciMLSensitivity: FiniteDiff
import SciMLSensitivity: ReverseDiff
import SciMLSensitivity: Zygote

import FMIBase.ChainRulesCore
using FMIBase.ChainRulesCore: ZeroTangent, NoTangent, @thunk

import ForwardDiffChainRules
using ForwardDiffChainRules: @ForwardDiff_frule
using SciMLSensitivity.ReverseDiff: @grad_from_chainrules

using SciMLSensitivity.LinearAlgebra
import SciMLSensitivity.SparseDiffTools

using FMIBase
using FMIBase.FMICore
using FMIBase: undual, unsense, untrack

include("utils.jl")
include("sense.jl")
include("hotfixes.jl")

end # module
