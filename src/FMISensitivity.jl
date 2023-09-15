#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module FMISensitivity

import SciMLSensitivity
import SciMLSensitivity.ForwardDiff
import SciMLSensitivity.ReverseDiff
using FMICore.ChainRulesCore
import ForwardDiffChainRules: @ForwardDiff_frule
import SciMLSensitivity.ReverseDiff: @grad_from_chainrules
import FMICore.ChainRulesCore: ZeroTangent, NoTangent, @thunk
import FMICore.ChainRulesCore
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

include("utils.jl")
include("FMI2.jl")

end # module
