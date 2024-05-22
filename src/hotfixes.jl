#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import SciMLSensitivity.Zygote: grad_mut, Context
import FMIBase: FMUEvaluationOutput
#grad_mut(av::AbstractVector) = invoke(grad_mut, Tuple{Any}, av)
grad_mut(av::FMUEvaluationOutput) = invoke(grad_mut, Tuple{Any}, av)
#grad_mut(c::Zygote.Context, av::AbstractVector) = invoke(grad_mut, Tuple{Zygote.Context, Any}, c, av)
grad_mut(c::Zygote.Context, av::FMUEvaluationOutput) = invoke(grad_mut, Tuple{Zygote.Context, Any}, c, av)
#grad_mut(av::AbstractVector) = []