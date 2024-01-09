#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMISensitivity.ForwardDiff 
import FMISensitivity.Zygote
import FMISensitivity.ReverseDiff
# import FMISensitivity.FiniteDiff

CHECK_ZYGOTE = false

# load demo FMU
fmu = fmi2Load("SpringPendulumExtForce1D", EXPORTINGTOOL, EXPORTINGVERSION; type=:ME)

# enable time gradient evaluation (disabled by default for performance reasons)
fmu.executionConfig.eval_t_gradients = true

# prepare (allocate) an FMU instance
c, x0 = FMIImport.prepareSolveFMU(fmu, nothing, fmu.type, nothing, nothing, nothing, nothing, nothing, nothing, 0.0, 0.0, nothing)

x_refs = fmu.modelDescription.stateValueReferences
x = fmi2GetContinuousStates(c)
dx = fmi2GetReal(c, c.fmu.modelDescription.derivativeValueReferences)
u_refs = fmu.modelDescription.inputValueReferences
u = [0.0]
y_refs = fmu.modelDescription.outputValueReferences
y = fmi2GetReal(c, y_refs)
p_refs = fmu.modelDescription.parameterValueReferences
p = fmi2GetReal(c, p_refs)
e = fmi2GetEventIndicators(c)
t = 0.0

function reset!(c::FMIImport.FMU2Component)
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
ydx = fmu(;x=x)
@test length(ydx) == 0

# evaluation: set state, get state derivative
ydx = fmu(;x=x, dx_refs=:all)
@test length(ydx) == 2

# evaluation: set state and inputs
ydx = fmu(;x=x, u=u, u_refs=u_refs, dx_refs=:all)
@test length(ydx) == 2

# evaluation: set state and inputs, get outputs (in-place)
ydx = fmu(;x=x, u=u, u_refs=u_refs, y=y, y_refs=y_refs)
@test length(ydx) == 2

# evaluation: set state and inputs, get state derivative and outputs (in-place)
ydx = fmu(;dx_refs=:all, x=x, u=u, u_refs=u_refs, y=y, y_refs=y_refs)
@test length(ydx) == 4

# evaluation: set state and inputs, parameters, get state derivative (in-place) and outputs (in-place)
ydx = fmu(;x=x, u=u, u_refs=u_refs, y=y, y_refs=y_refs, dx=dx, dx_refs=:all, p=p, p_refs=p_refs)
@test length(ydx) == 4

# known results
atol= 1e-7
∂ẋ_∂x = [0.0 1.0; -10.0 0.0]
∂ẋ_∂u = [0.0; 1.0]
∂ẋ_∂p = [0.0   0.0  0.0   0.0   0.0  0.0;
         0.0  10.0  0.6  10.0  -6.0  5.0]
∂y_∂x = [0.0 1.0; -10.0 0.0]
∂y_∂u = [0.0; 1.0]
∂y_∂p = [0.0   0.0  0.0   0.0   0.0  0.0;
         0.0  10.0  0.6  10.0  -6.0  5.0]
∂ẋ_∂t = [0.0, 0.0]
∂y_∂t = [0.0, 0.0]
∂e_∂x = [0.0  0.0  0.0  0.0   0.0  0.0;
0.0  0.0  0.0  0.0   0.0  0.0;
0.0  0.0  0.0  0.0   0.0  0.0;
0.0  0.0  0.0  0.0   0.0  0.0;
0.0  0.0  0.0  0.0  -1.0  0.0;
0.0  0.0  0.0  0.0  -1.0  0.0;
0.0  0.0  0.0  0.0   0.0  0.0;
0.0  0.0  0.0  0.0   0.0  0.0;
0.0  0.0  0.0  0.0   0.0  0.0;
0.0  0.0  0.0  0.0   0.0  0.0;
0.0  0.0  0.0  0.0   0.0  0.0;
 0.0  0.0  0.0  0.0   0.0  0.0;
 0.0  0.0  0.0  0.0   0.0  0.0;
 0.0  0.0  0.0  0.0   0.0  0.0;
 0.0  0.0  0.0  0.0   0.0  0.0;
 0.0  0.0  0.0  0.0   0.0  0.0;
 0.0  0.0  0.0  0.0  -1.0  0.0;
 0.0  0.0  0.0  0.0  -1.0  0.0;
 0.0  0.0  0.0  0.0   0.0  0.0;
 0.0  0.0  0.0  0.0   0.0  0.0;
 0.0  0.0  0.0  0.0     0.0    0.0;
 0.0  0.0  0.0  0.0     0.0    0.0;
 0.0  0.0  0.0  0.0  -132.390625  0.0;
 0.0  0.0  0.0  0.0  -132.390625  0.0;
 0.0  0.0  0.0  0.0   132.389404296875  0.0;
 0.0  0.0  0.0  0.0   132.389404296875  0.0;
 0.0  0.0  0.0  1.0     0.0    0.0;
 0.0  0.0  0.0  1.0     0.0    0.0]

# Test build-in derivatives (slow) only for jacobian A
fmu.executionConfig.JVPBuiltInDerivatives = true

_f = _x -> fmu(;x=_x, dx_refs=:all)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_rwd = ReverseDiff.jacobian(_f, x)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, x)[1] : nothing
j_smp = fmi2SampleJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)
j_get = fmi2GetJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)

@test isapprox(j_fwd, ∂ẋ_∂x; atol=atol)
@test isapprox(j_rwd, ∂ẋ_∂x; atol=atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂ẋ_∂x; atol=atol) : true
@test isapprox(j_smp, ∂ẋ_∂x; atol=atol)
@test isapprox(j_get, ∂ẋ_∂x; atol=atol)

# End: Test build-in derivatives (slow) only for jacobian A
fmu.executionConfig.JVPBuiltInDerivatives = false

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

# Jacobian A=∂dx/∂x (out-of-plcae)
_f = _x -> fmu(;x=_x, dx_refs=:all)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_rwd = ReverseDiff.jacobian(_f, x)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, x)[1] : nothing
j_smp = fmi2SampleJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)
j_get = fmi2GetJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)

@test isapprox(j_fwd, ∂ẋ_∂x; atol=atol)
@test isapprox(j_rwd, ∂ẋ_∂x; atol=atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂ẋ_∂x; atol=atol) : true
@test isapprox(j_smp, ∂ẋ_∂x; atol=atol)
@test isapprox(j_get, ∂ẋ_∂x; atol=atol)

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
_f = _x -> fmu(;x=_x, dx=dx, dx_refs=:all)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_rwd = ReverseDiff.jacobian(_f, x)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, x)[1] : nothing
j_smp = fmi2SampleJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)
j_get = fmi2GetJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)

@test isapprox(j_fwd, ∂ẋ_∂x; atol=atol)
@test isapprox(j_rwd, ∂ẋ_∂x; atol=atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂ẋ_∂x; atol=atol) : true
@test isapprox(j_smp, ∂ẋ_∂x; atol=atol)
@test isapprox(j_get, ∂ẋ_∂x; atol=atol)

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
_f = _u -> fmu(; u=_u, u_refs=u_refs, dx_refs=:all)
_f(u)
j_fwd = ForwardDiff.jacobian(_f, u)
j_rwd = ReverseDiff.jacobian(_f, u)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, u)[1] : nothing
j_smp = fmi2SampleJacobian(c, fmu.modelDescription.derivativeValueReferences, u_refs)
j_get = fmi2GetJacobian(c, fmu.modelDescription.derivativeValueReferences, u_refs)

@test isapprox(j_fwd, ∂ẋ_∂u; atol=atol)
@test isapprox(j_rwd, ∂ẋ_∂u; atol=atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂ẋ_∂u; atol=atol) : true
@test isapprox(j_smp, ∂ẋ_∂u; atol=atol)
@test isapprox(j_get, ∂ẋ_∂u; atol=atol)

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
_f = _x -> fmu(;x=_x, y_refs=y_refs)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_rwd = ReverseDiff.jacobian(_f, x)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, x)[1] : nothing
j_smp = fmi2SampleJacobian(c, y_refs, fmu.modelDescription.stateValueReferences)
j_get = fmi2GetJacobian(c, y_refs, fmu.modelDescription.stateValueReferences)

@test isapprox(j_fwd, ∂y_∂x; atol=atol)
@test isapprox(j_rwd, ∂y_∂x; atol=atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂y_∂x; atol=atol) : true
@test isapprox(j_smp, ∂y_∂x; atol=atol)
@test isapprox(j_get, ∂y_∂x; atol=atol)

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

# Jacobian D=∂y/∂u
_f = _u -> fmu(; u=_u, u_refs=u_refs, y_refs=y_refs)
_f(u)
j_fwd = ForwardDiff.jacobian(_f, u)
j_rwd = ReverseDiff.jacobian(_f, u)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, u)[1] : nothing
j_smp = fmi2SampleJacobian(c, y_refs, u_refs)
j_get = fmi2GetJacobian(c, y_refs, u_refs)

@test isapprox(j_fwd, ∂y_∂u; atol=atol)
@test isapprox(j_rwd, ∂y_∂u; atol=atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂y_∂u; atol=atol) : true
@test isapprox(j_smp, ∂y_∂u; atol=atol)
@test isapprox(j_get, ∂y_∂u; atol=atol)

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
_f = _t -> fmu(; t=_t, dx_refs=:all)
_f(t)
j_fwd = ForwardDiff.derivative(_f, t)

# ReverseDiff has no `derivative` function for scalars
_f = _t -> fmu(; t=_t[1], dx_refs=:all)
j_rwd = ReverseDiff.jacobian(_f, [t])

j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, t)[1] : nothing

@test isapprox(j_fwd, ∂ẋ_∂t; atol=atol)
@test isapprox(j_rwd, ∂ẋ_∂t; atol=atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂ẋ_∂t; atol=atol) : true

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
_f = _t -> fmu(; y_refs=y_refs, t=_t)
_f(t)
j_fwd = ForwardDiff.derivative(_f, t)

# ReverseDiff has no `derivative` function for scalars
_f = _t -> fmu(; y_refs=y_refs, t=_t[1])
j_rwd = ReverseDiff.jacobian(_f, [t])

j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, t)[1] : nothing

@test isapprox(j_fwd, ∂y_∂t; atol=atol)
@test isapprox(j_rwd, ∂y_∂t; atol=atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂y_∂t; atol=atol) : true

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
_f = _p -> fmu(;p=_p, p_refs=p_refs, dx_refs=:all)
_f(p)
j_fwd = ForwardDiff.jacobian(_f, p)
j_rwd = ReverseDiff.jacobian(_f, p)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, p)[1] : nothing
j_smp = fmi2SampleJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.parameterValueReferences)
j_get = fmi2GetJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.parameterValueReferences)

@test isapprox(j_fwd, ∂ẋ_∂p; atol=atol)
@test isapprox(j_rwd, ∂ẋ_∂p; atol=atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂ẋ_∂p; atol=atol) : true
@test isapprox(j_smp, ∂ẋ_∂p; atol=atol)
@test isapprox(j_get, ∂ẋ_∂p; atol=atol)

@test c.solution.evals_∂ẋ_∂x == 0
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == (CHECK_ZYGOTE ? 10 : 8)
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

# Jacobian ∂y/∂p
_f = _p -> fmu(;p=_p, p_refs=p_refs, y_refs=y_refs)
_f(p)
j_fwd = ForwardDiff.jacobian(_f, p)
j_rwd = ReverseDiff.jacobian(_f, p)
j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, p)[1] : nothing
j_smp = fmi2SampleJacobian(c, fmu.modelDescription.outputValueReferences, fmu.modelDescription.parameterValueReferences)
j_get = fmi2GetJacobian(c, fmu.modelDescription.outputValueReferences, fmu.modelDescription.parameterValueReferences)

@test isapprox(j_fwd, ∂y_∂p; atol=atol)
@test isapprox(j_rwd, ∂y_∂p; atol=atol)
@test CHECK_ZYGOTE ? isapprox(j_zyg, ∂y_∂p; atol=atol) : true
@test isapprox(j_smp, ∂y_∂p; atol=atol)
@test isapprox(j_get, ∂y_∂p; atol=atol)

@test c.solution.evals_∂ẋ_∂x == 0
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == (CHECK_ZYGOTE ? 10 : 8)
@test c.solution.evals_∂y_∂x == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# clean up
fmi2Unload(fmu)

########## Event Indicators Check ###########

# [ToDo] Enable Test for Linux too (by providing a FMU)

if Sys.iswindows()
     # load demo FMU
     fmu = fmi2Load("VLDM", EXPORTINGTOOL, "2020x"; type=:ME)
     data = FMIZoo.VLDM(:train)

     # enable time gradient evaluation (disabled by default for performance reasons)
     fmu.executionConfig.eval_t_gradients = true

     # prepare (allocate) an FMU instance
     c, x0 = FMIImport.prepareSolveFMU(fmu, nothing, fmu.type, nothing, nothing, nothing, nothing, nothing, data.params, 0.0, 0.0, nothing)

     x_refs = fmu.modelDescription.stateValueReferences
     x = fmi2GetContinuousStates(c)
     dx = fmi2GetReal(c, c.fmu.modelDescription.derivativeValueReferences)
     u_refs = fmu.modelDescription.inputValueReferences
     u = zeros(fmi2Real, 0)
     y_refs = fmu.modelDescription.outputValueReferences
     y = zeros(fmi2Real, 0)
     p_refs = copy(fmu.modelDescription.parameterValueReferences)

     # remove some parameters
     deleteat!(p_refs, findall(x -> (x >= UInt32(134217728) && x <= UInt32(134217737)), p_refs))
     p = fmi2GetReal(c, p_refs)
     e = fmi2GetEventIndicators(c)
     t = 0.0

     # Jacobian ∂e/∂x
     _f = function(_x)
          ec_idcs = collect(UInt32(i) for i in 1:fmu.modelDescription.numberOfEventIndicators)
          
          ret = fmu(; ec_idcs=ec_idcs, x=_x)
          
          return ret.ec
     end
     _f(x)
     j_fwd = ForwardDiff.jacobian(_f, x)
     j_rwd = ReverseDiff.jacobian(_f, x)
     j_zyg = CHECK_ZYGOTE ? Zygote.jacobian(_f, x)[1] : nothing
     # j_fid = FiniteDiff.finite_difference_jacobian(_f, x)
     # j_smp = fmi2SampleJacobian(c, :indicators, fmu.modelDescription.parameterValueReferences)
     # no option to get sensitivitities directly in FMI2... 

     @test isapprox(j_fwd, ∂e_∂x; atol=atol)
     @test isapprox(j_rwd, ∂e_∂x; atol=atol)
     @test CHECK_ZYGOTE ? isapprox(j_zyg, j_rwd; atol=atol) : true

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
     fmi2Unload(fmu)
end