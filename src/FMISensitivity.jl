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
import FMICore.ChainRulesCore

import ForwardDiffChainRules: @ForwardDiff_frule
import SciMLSensitivity.ReverseDiff: @grad_from_chainrules
import FMICore.ChainRulesCore: ZeroTangent, NoTangent, @thunk
using FMICore: undual, unsense, untrack

using SciMLSensitivity.LinearAlgebra
import SciMLSensitivity.SparseDiffTools

import FMICore: invalidate!, check_invalidate!

using FMICore

function isZeroTangent(d)
    return false
end
function isZeroTangent(d::ZeroTangent)
    return true
end
function isZeroTangent(d::AbstractArray{<:ZeroTangent})
    return true
end

# additional dispatch for ReverseDiff.jl 
import SciMLSensitivity.ReverseDiff: increment_deriv!, ZeroTangent
function ReverseDiff.increment_deriv!(::ReverseDiff.TrackedReal, ::ZeroTangent)
    return nothing 
end

include("FMI2.jl")

end # module
