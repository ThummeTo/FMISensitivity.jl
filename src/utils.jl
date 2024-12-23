#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

function isZeroTangent(d)
    return false
end
function isZeroTangent(d::ZeroTangent)
    return true
end
function isZeroTangent(d::AbstractArray{<:ZeroTangent})
    return true
end
