#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMISensitivity.ForwardDiff
import FMISensitivity.Zygote
import FMISensitivity.ReverseDiff
import FMISensitivity.FiniteDiff

using FMISensitivity.FMIBase
using FMISensitivity.FMIBase.FMICore

using FMISensitivity.FMIBase:
    getContinuousStates, getReal, getRealType, getEventIndicators, getDirectionalDerivative

CHECK_ZYGOTE = false

# load demo FMU
c, fmu = getFMUStruct("SpringFrictionPendulumExtForce1D", :ME)

# enable time gradient evaluation (disabled by default for performance reasons)
EVAL_T_GRAD = true
fmu.executionConfig.eval_t_gradients = EVAL_T_GRAD

x_refs = fmu.modelDescription.stateValueReferences
x = getContinuousStates(c)
dx_refs = c.fmu.modelDescription.derivativeValueReferences
dx = getReal(c, dx_refs)
u_refs = fmu.modelDescription.inputValueReferences
u = ones(getRealType(fmu), length(u_refs))
y_refs = reverse(fmu.modelDescription.outputValueReferences) # to add a little spice!
y = getReal(c, y_refs)
p_refs = fmu.modelDescription.parameterValueReferences
p = getReal(c, p_refs)
e = getEventIndicators(c)
ec_idcs = collect(UInt32(i) for i = 1:fmu.modelDescription.numberOfEventIndicators)
t = 0.0

sampleJacobian(_f, x) = FiniteDiff.finite_difference_jacobian(
    _x -> [_f(_x)...],
    x,
    Val(:central);
    relstep = 0.0,
    absstep = 1e-4,
)

reset! = function (c::FMUInstance)

    # clean FMU call 
    fmu(; u = u .* 0.0, u_refs = u_refs)

    c.solution.evals_∂ẋ_∂x = 0
    c.solution.evals_∂ẋ_∂u = 0
    c.solution.evals_∂ẋ_∂p = 0
    c.solution.evals_∂ẋ_∂t = 0

    c.solution.evals_∂y_∂x = 0
    c.solution.evals_∂y_∂u = 0
    c.solution.evals_∂y_∂p = 0
    c.solution.evals_∂y_∂t = 0

    c.solution.evals_∂e_∂x = 0
    c.solution.evals_∂e_∂u = 0
    c.solution.evals_∂e_∂p = 0
    c.solution.evals_∂e_∂t = 0

    @test length(dx) == length(fmu.modelDescription.derivativeValueReferences)
    @test length(y) == length(fmu.modelDescription.outputValueReferences)
    @test length(e) == fmu.modelDescription.numberOfEventIndicators
end


# evaluation: set state
dx_y = fmu(; x = x)
@test length(dx_y) == 0

# evaluation: set state, get state derivative
dx_y = fmu(; x = x, dx_refs = :all)
@test isapprox(dx_y, [0.0, 5.25])

# evaluation: set state and inputs
dx_y = fmu(; x = x, u = u, u_refs = u_refs, dx_refs = :all)
@test isapprox(dx_y, [0.0, 6.25])

# evaluation: set state, get state derivative (after setting the input once)
dx_y = fmu(; x = x, u = u .* 0.0, u_refs = u_refs, dx_refs = :all)
@test isapprox(dx_y, [0.0, 5.25])

# evaluation: set state and inputs, get outputs (in-place)
dx_y = fmu(; x = x, u = u, u_refs = u_refs, y = y, y_refs = y_refs)
@test isapprox(dx_y, [6.25, 0.0])

# evaluation: set state and inputs, get state derivative and outputs (in-place)
dx_y = fmu(; dx_refs = :all, x = x, u = u, u_refs = u_refs, y = y, y_refs = y_refs)
@test isapprox(dx_y, [0.0, 6.25, 6.25, 0.0])

# evaluation: set state and inputs, parameters, get state derivative (in-place) and outputs (in-place)
dx_y = fmu(;
    x = x,
    u = u,
    u_refs = u_refs,
    y = y,
    y_refs = y_refs,
    dx = dx,
    dx_refs = :all,
    p = p,
    p_refs = p_refs,
)
@test isapprox(dx_y, [0.0, 6.25, 6.25, 0.0])

# known results
atol = 1e-3 # 1e-7
∂ẋ_∂x = [0.0 1.0; -10.0 -0.05]
∂ẋ_∂u = [0.0; 1.0]
∂ẋ_∂p = [
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0375 0.0 0.0 10.0 0.6 10.0 0.0 0.0 0.0 5.0 -5.25 0.0
]
∂y_∂x = [
    -10.0 -0.05;
    0.0 1.0
]
∂y_∂u = [
    1.0;
    0.0
]
∂y_∂p = [
    0.0375 0.0 0.0 10.0 0.6 10.0 0.0 0.0 0.0 5.0 -5.25 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
]
∂ẋ_∂t = [0.0, 0.0]
∂y_∂t = [0.0, 0.0]
∂e_∂x = [
    0.0 0.0
    0.0 0.0
    1.0 0.0
    1.0 0.0
    -10.0 -0.05
    -10.0 -0.05
    0.0 0.0
    0.0 0.0
    0.0 0.0
    0.0 0.0
    0.0 1.0
    0.0 1.0
    -10.0 -0.05
    -10.0 -0.05
    0.0 0.0
    0.0 0.0
    0.0 0.0
    0.0 0.0
    0.0 0.0
    0.0 0.0
    0.0 0.0
    0.0 0.0
    0.0 0.0
    0.0 0.0
    0.0 0.0
    0.0 0.0
    0.0 0.0
    0.0 0.0
    1.0 0.0
    1.0 0.0
    1.0 0.0
    1.0 0.0
]

# Test build-in directional derivatives (slow) only for jacobian A
fmu.executionConfig.JVPBuiltInDerivatives = true

_f = _x -> fmu(; x = _x, dx_refs = :all)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_rwd = ReverseDiff.jacobian(_f, x)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, x)[1] : nothing
# j_smp = sampleJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)
# j_get = getJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)

@test isapprox(j_fwd, ∂ẋ_∂x; atol = atol)
@test isapprox(j_rwd, ∂ẋ_∂x; atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂ẋ_∂x; atol = atol) : true
# @test isapprox(j_smp, ∂ẋ_∂x; atol=atol)
# @test isapprox(j_get, ∂ẋ_∂x; atol=atol)

# End: Test build-in derivatives (slow) only for jacobian A
fmu.executionConfig.JVPBuiltInDerivatives = false

# Test build-in adjoint derivatives (slow) only for jacobian A
fmu.executionConfig.VJPBuiltInDerivatives = true

_f = _x -> fmu(; x = _x, dx_refs = :all)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_rwd = ReverseDiff.jacobian(_f, x)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, x)[1] : nothing

@test isapprox(j_fwd, ∂ẋ_∂x; atol = atol)
@test isapprox(j_rwd, ∂ẋ_∂x; atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂ẋ_∂x; atol = atol) : true

# End: Test build-in derivatives (slow) only for jacobian A
fmu.executionConfig.VJPBuiltInDerivatives = false

@test c.solution.evals_∂ẋ_∂x == (CHECK_ZYGOTE ? 10 : 8)
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂x == 0
@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# Test custom finite differences sampling instead of directionalDerivative 
fmu.executionConfig.sensitivity_strategy = :FiniteDiff

_f = _x -> fmu(; x = _x, dx_refs = :all)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_rwd = ReverseDiff.jacobian(_f, x)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, x)[1] : nothing

@test isapprox(j_fwd, ∂ẋ_∂x; atol = atol)
@test isapprox(j_rwd, ∂ẋ_∂x; atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂ẋ_∂x; atol = atol) : true

# End: Test custom finite differences sampling instead of directionalDerivative 
fmu.executionConfig.sensitivity_strategy = :FMIDirectionalDerivative

@test c.solution.evals_∂ẋ_∂x == 4
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂x == 0
@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# Jacobian A=∂dx/∂x (out-of-plcae)
_f = _x -> fmu(; x = _x, dx_refs = :all)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_rwd = ReverseDiff.jacobian(_f, x)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, x)[1] : nothing
#j_smp = sampleJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)
#j_get = getJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)

@test isapprox(j_fwd, ∂ẋ_∂x; atol = atol)
@test isapprox(j_rwd, ∂ẋ_∂x; atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂ẋ_∂x; atol = atol) : true
#@test isapprox(j_smp, ∂ẋ_∂x; atol=atol)
#@test isapprox(j_get, ∂ẋ_∂x; atol=atol)

@test c.solution.evals_∂ẋ_∂x == (CHECK_ZYGOTE ? 6 : 4)
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂x == 0
@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# Jacobian A=∂dx/∂x (in-plcae)
_f = _x -> fmu(; x = _x, dx = dx, dx_refs = :all)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_rwd = ReverseDiff.jacobian(_f, x)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, x)[1] : nothing
#j_smp = sampleJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)
#j_get = getJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)

@test isapprox(j_fwd, ∂ẋ_∂x; atol = atol)
@test isapprox(j_rwd, ∂ẋ_∂x; atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂ẋ_∂x; atol = atol) : true
#@test isapprox(j_smp, ∂ẋ_∂x; atol=atol)
#@test isapprox(j_get, ∂ẋ_∂x; atol=atol)

@test c.solution.evals_∂ẋ_∂x == (CHECK_ZYGOTE ? 6 : 4)
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂x == 0
@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# Jacobian B=∂dx/∂u
_f = _u -> fmu(; u = _u, u_refs = u_refs, dx_refs = :all)
_f(u)
j_fwd = ForwardDiff.jacobian(_f, u)
j_rwd = ReverseDiff.jacobian(_f, u)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, u)[1] : nothing
#j_smp = fsampleJacobian(c, fmu.modelDescription.derivativeValueReferences, u_refs)
#j_get = getJacobian(c, fmu.modelDescription.derivativeValueReferences, u_refs)

@test isapprox(j_fwd, ∂ẋ_∂u; atol = atol)
@test isapprox(j_rwd, ∂ẋ_∂u; atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂ẋ_∂u; atol = atol) : true
#@test isapprox(j_smp, ∂ẋ_∂u; atol=atol)
#@test isapprox(j_get, ∂ẋ_∂u; atol=atol)

@test c.solution.evals_∂ẋ_∂x == 0
@test c.solution.evals_∂ẋ_∂u == (CHECK_ZYGOTE ? 5 : 3)
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂x == 0
@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# Jacobian C=∂y/∂x (in-place)
_f = _x -> fmu(; x = _x, y_refs = y_refs)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_rwd = ReverseDiff.jacobian(_f, x)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, x)[1] : nothing
j_smp = sampleJacobian(_f, x)
#j_smp = sampleJacobian(c, y_refs, fmu.modelDescription.stateValueReferences)
#j_get = getJacobian(c, y_refs, fmu.modelDescription.stateValueReferences)

@test isapprox(j_fwd, ∂y_∂x; atol = atol)
@test isapprox(j_rwd, ∂y_∂x; atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂y_∂x; atol = atol) : true
@test isapprox(j_smp, ∂y_∂x; atol = atol)
#@test isapprox(j_get, ∂y_∂x; atol=atol)

@test c.solution.evals_∂ẋ_∂x == 0
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂x == (CHECK_ZYGOTE ? 6 : 4)
@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# Jacobian AC=∂(dx y)/∂x (in-place)
_f = _x -> fmu(; x = _x, dx = dx, dx_refs = :all, y_refs = y_refs)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_rwd = ReverseDiff.jacobian(_f, x)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, x)[1] : nothing
#j_smp = sampleJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)
#j_get = getJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)

@test isapprox(j_fwd, vcat(∂ẋ_∂x, ∂y_∂x); atol = atol)
@test isapprox(j_rwd, vcat(∂ẋ_∂x, ∂y_∂x); atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, vcat(∂ẋ_∂x, ∂y_∂x); atol = atol) : true
#@test isapprox(j_smp, ∂ẋ_∂x; atol=atol)
#@test isapprox(j_get, ∂ẋ_∂x; atol=atol)

@test c.solution.evals_∂ẋ_∂x == (CHECK_ZYGOTE ? 10 : 6)
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂x == (CHECK_ZYGOTE ? 10 : 6)
@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# Jacobian ACE=∂(dx y ec)/∂x (out-of-place)
_f = _x -> fmu(; x = _x, dx_refs = :all, y_refs = y_refs, ec_idcs = ec_idcs).buffer
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_rwd = ReverseDiff.jacobian(_f, x)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, x)[1] : nothing
j_smp = sampleJacobian(_f, x)
#j_get = getJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)

@test isapprox(j_fwd, vcat(∂ẋ_∂x, ∂y_∂x, ∂e_∂x); atol = atol)
@test isapprox(j_rwd, vcat(∂ẋ_∂x, ∂y_∂x, ∂e_∂x); atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, vcat(∂ẋ_∂x, ∂y_∂x, ∂e_∂x); atol = atol) : true
@test isapprox(j_smp, vcat(∂ẋ_∂x, ∂y_∂x, ∂e_∂x); atol = atol)
#@test isapprox(j_get, ∂ẋ_∂x; atol=atol)

@test c.solution.evals_∂ẋ_∂x == (CHECK_ZYGOTE ? 1 : 38)
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂x == (CHECK_ZYGOTE ? 1 : 38)
@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 38
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# Jacobian D=∂y/∂u
_f = _u -> fmu(; u = _u, u_refs = u_refs, y_refs = y_refs)
_f(u)
j_fwd = ForwardDiff.jacobian(_f, u)
j_rwd = ReverseDiff.jacobian(_f, u)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, u)[1] : nothing
j_smp = sampleJacobian(_f, u)
#j_get = getJacobian(c, y_refs, u_refs)

@test isapprox(j_fwd, ∂y_∂u; atol = atol)
@test isapprox(j_rwd, ∂y_∂u; atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂y_∂u; atol = atol) : true
@test isapprox(j_smp, ∂y_∂u; atol = atol)
#@test isapprox(j_get, ∂y_∂u; atol=atol)

@test c.solution.evals_∂ẋ_∂x == 0
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂x == 0
@test c.solution.evals_∂y_∂u == (CHECK_ZYGOTE ? 5 : 3)
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# explicit time derivative ∂ẋ/∂t
_f = _t -> fmu(; t = _t, dx_refs = :all)
_f(t)
j_fwd = ForwardDiff.derivative(_f, t)

# ReverseDiff has no `derivative` function for scalars
_f = _t -> fmu(; t = _t[1], dx_refs = :all)

j_rwd = ReverseDiff.jacobian(_f, [t])
#@warn "ReverseDiff time gradient skipped."
#j_rwd = ∂ẋ_∂t
#c.solution.evals_∂ẋ_∂t += 2

j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, t)[1] : nothing

@test isapprox(j_fwd, ∂ẋ_∂t; atol = atol)
@test isapprox(j_rwd, ∂ẋ_∂t; atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂ẋ_∂t; atol = atol) : true

@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂x == 0
@test c.solution.evals_∂ẋ_∂t == (CHECK_ZYGOTE ? 5 : 3)

@test c.solution.evals_∂y_∂x == 0
@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# explicit time derivative ∂y/∂t 
_f = _t -> fmu(; y_refs = y_refs, t = _t)
_f(t)
j_fwd = ForwardDiff.derivative(_f, t)

# ReverseDiff has no `derivative` function for scalars
_f = _t -> fmu(; y_refs = y_refs, t = _t[1])

j_rwd = ReverseDiff.jacobian(_f, [t])
#@warn "ReverseDiff time gradient skipped."
#j_rwd = ∂y_∂t
#c.solution.evals_∂y_∂t += 2

j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, t)[1] : nothing

@test isapprox(j_fwd, ∂y_∂t; atol = atol)
@test isapprox(j_rwd, ∂y_∂t; atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂y_∂t; atol = atol) : true

@test c.solution.evals_∂ẋ_∂x == 0
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂x == 0
@test c.solution.evals_∂y_∂t == (CHECK_ZYGOTE ? 5 : 3)

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# Jacobian ∂ẋ/∂p
#if FMU_SUPPORTS_PARAMETER_SAMPLING
_f = _p -> fmu(; p = _p, p_refs = p_refs, dx_refs = :all)
_f(p)
j_fwd = ForwardDiff.jacobian(_f, p)
j_rwd = ReverseDiff.jacobian(_f, p)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, p)[1] : nothing

@test isapprox(j_fwd, ∂ẋ_∂p; atol = atol)
@test isapprox(j_rwd, ∂ẋ_∂p; atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂ẋ_∂p; atol = atol) : true

@test c.solution.evals_∂ẋ_∂x == 0
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == (CHECK_ZYGOTE ? 16 : 14)
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂x == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)
#end

# Jacobian ∂y/∂p
#if FMU_SUPPORTS_PARAMETER_SAMPLING
_f = _p -> fmu(; p = _p, p_refs = p_refs, y_refs = y_refs)
_f(p)
j_fwd = ForwardDiff.jacobian(_f, p)
j_rwd = ReverseDiff.jacobian(_f, p)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, p)[1] : nothing

@test isapprox(j_fwd, ∂y_∂p; atol = atol)
@test isapprox(j_rwd, ∂y_∂p; atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂y_∂p; atol = atol) : true

@test c.solution.evals_∂ẋ_∂x == 0
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == (CHECK_ZYGOTE ? 16 : 14)
@test c.solution.evals_∂y_∂x == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)
#end

# Jacobian ∂e/∂x
_f = function (_x)
    ec_idcs = collect(UInt32(i) for i = 1:fmu.modelDescription.numberOfEventIndicators)

    ret = fmu(; ec_idcs = ec_idcs, x = _x)

    return ret.ec
end
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_rwd = ReverseDiff.jacobian(_f, x)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, x)[1] : nothing
# j_fid = FiniteDiff.finite_difference_jacobian(_f, x)
# j_smp = sampleJacobian(c, :indicators, fmu.modelDescription.parameterValueReferences)
# no option to get sensitivitities directly in FMI2/FMI3... 

@test isapprox(j_fwd, ∂e_∂x; atol = atol)
@test isapprox(j_rwd, ∂e_∂x; atol = atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, j_rwd; atol = atol) : true

@test c.solution.evals_∂ẋ_∂x == 0
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂x == 0
@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == (CHECK_ZYGOTE ? 62 : 34)
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# clean up
unloadFMU(fmu)
