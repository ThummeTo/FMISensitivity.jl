#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# ccalls for inplace getters lead to wrong results when used with AD-primitives, 
# so this file contains special dispatches, that prevent allocations while also 
# enable for calling fmi2XXX functions with AD

import FMICore: fmi2GetEventIndicators!