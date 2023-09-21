#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMISensitivity.ForwardDiff 
import FMISensitivity.Zygote
import FMISensitivity.ReverseDiff

# load demo FMU
fmu = fmi2Load("SpringPendulumExtForce1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"]; type=:ME)

# enable time gradient evaluation (disabled by default for performance reasons)
fmu.executionConfig.eval_t_gradients = true

# prepare (allocate) an FMU instance
c, x0 = FMIImport.prepareSolveFMU(fmu, nothing, fmu.type, nothing, nothing, nothing, nothing, nothing, nothing, 0.0, 0.0, nothing)

x = [1.0, 1.0]
x_refs = fmu.modelDescription.stateValueReferences
u = [2.0]
u_refs = fmu.modelDescription.inputValueReferences
y = [0.0, 0.0]
y_refs = fmu.modelDescription.outputValueReferences
p_refs = fmu.modelDescription.parameterValueReferences
p = zeros(length(p_refs))
dx = [0.0, 0.0]
e = zeros(fmi2Real, fmu.modelDescription.numberOfEventIndicators)
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

# evaluation: set state, get state derivative
ydx = fmu(;x=x, dx=dx)

# evaluation: set state and inputs
ydx = fmu(;x=x, u=u, u_refs=u_refs)

# evaluation: set state and inputs, get state derivative and outputs (in-place)
ydx = fmu(;dx=dx, x=x, u=u, u_refs=u_refs, y=y, y_refs=y_refs)

# evaluation: set state and inputs, parameters, get state derivative (in-place) and outputs (in-place)
ydx = fmu(;x=x, u=u, u_refs=u_refs, y=y, y_refs=y_refs, dx=dx, p=p, p_refs=p_refs)

# known results
atol= 1e-8
∂ẋ_∂x = [0.0 1.0; -10.0 0.0]
∂ẋ_∂u = [0.0; 1.0]
∂ẋ_∂p = [0.0   0.0  0.0   0.0   0.0  0.0;
     0.0  10.0  0.1  10.0  -3.0  5.0]
∂y_∂x = [0.0 1.0; -10.0 0.0]
∂y_∂u = [0.0; 1.0]
∂y_∂p = [0.0   0.0  0.0   0.0   0.0  0.0;
     0.0  10.0  0.1  10.0  -3.0  5.0]
∂x_∂t = [0.0, 0.0]
∂y_∂t = [0.0, 0.0]

# Test build-in derivatives (slow) only for jacobian A
fmu.executionConfig.JVPBuiltInDerivatives = true

_f = _x -> fmu(;x=_x, dx=dx)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_zyg = Zygote.jacobian(_f, x)[1]
j_rwd = ReverseDiff.jacobian(_f, x)
j_smp = fmi2SampleJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)
j_get = fmi2GetJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)

@test isapprox(j_fwd, ∂ẋ_∂x; atol=atol)
@test isapprox(j_zyg, ∂ẋ_∂x; atol=atol)
@test isapprox(j_rwd, ∂ẋ_∂x; atol=atol)
@test isapprox(j_smp, ∂ẋ_∂x; atol=atol)
@test isapprox(j_get, ∂ẋ_∂x; atol=atol)

# End: Test build-in derivatives (slow) only for jacobian A
fmu.executionConfig.JVPBuiltInDerivatives = false
reset!(c)

# Jacobian A=∂dx/∂x
_f = _x -> fmu(;x=_x, dx=dx)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_zyg = Zygote.jacobian(_f, x)[1]
j_rwd = ReverseDiff.jacobian(_f, x)
j_smp = fmi2SampleJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)
j_get = fmi2GetJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.stateValueReferences)

@test isapprox(j_fwd, ∂ẋ_∂x; atol=atol)
@test isapprox(j_zyg, ∂ẋ_∂x; atol=atol)
@test isapprox(j_rwd, ∂ẋ_∂x; atol=atol)
@test isapprox(j_smp, ∂ẋ_∂x; atol=atol)
@test isapprox(j_get, ∂ẋ_∂x; atol=atol)

@test c.solution.evals_∂ẋ_∂x == 2+2+2
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
_f = _u -> fmu(;x=x, u=_u, u_refs=u_refs, dx=dx)
_f(u)
j_fwd = ForwardDiff.jacobian(_f, u)
j_zyg = Zygote.jacobian(_f, u)[1]
j_rwd = ReverseDiff.jacobian(_f, u)
j_smp = fmi2SampleJacobian(c, fmu.modelDescription.derivativeValueReferences, u_refs)
j_get = fmi2GetJacobian(c, fmu.modelDescription.derivativeValueReferences, u_refs)

@test isapprox(j_fwd, ∂ẋ_∂u; atol=atol)
@test isapprox(j_zyg, ∂ẋ_∂u; atol=atol)
@test isapprox(j_rwd, ∂ẋ_∂u; atol=atol)
@test isapprox(j_smp, ∂ẋ_∂u; atol=atol)
@test isapprox(j_get, ∂ẋ_∂u; atol=atol)

@test c.solution.evals_∂ẋ_∂x == 5 
@test c.solution.evals_∂ẋ_∂u == 5
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
_f = _x -> fmu(;x=_x, y=y, y_refs=y_refs)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_zyg = Zygote.jacobian(_f, x)[1]
j_rwd = ReverseDiff.jacobian(_f, x)
j_smp = fmi2SampleJacobian(c, y_refs, fmu.modelDescription.stateValueReferences)
j_get = fmi2GetJacobian(c, y_refs, fmu.modelDescription.stateValueReferences)

@test isapprox(j_fwd, ∂y_∂x; atol=atol)
@test isapprox(j_zyg, ∂y_∂x; atol=atol)
@test isapprox(j_rwd, ∂y_∂x; atol=atol)
@test isapprox(j_smp, ∂y_∂x; atol=atol)
@test isapprox(j_get, ∂y_∂x; atol=atol)

@test c.solution.evals_∂ẋ_∂x == 0
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂x == 6
@test c.solution.evals_∂y_∂u == 0
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# Jacobian D=∂y/∂u
_f = _u -> fmu(;x=x, u=_u, u_refs=u_refs, y=y, y_refs=y_refs)
_f(u)
j_fwd = ForwardDiff.jacobian(_f, u)
j_zyg = Zygote.jacobian(_f, u)[1]
j_rwd = ReverseDiff.jacobian(_f, u)
j_smp = fmi2SampleJacobian(c, y_refs, u_refs)
j_get = fmi2GetJacobian(c, y_refs, u_refs)

@test isapprox(j_fwd, ∂y_∂u; atol=atol)
@test isapprox(j_zyg, ∂y_∂u; atol=atol)
@test isapprox(j_rwd, ∂y_∂u; atol=atol)
@test isapprox(j_smp, ∂y_∂u; atol=atol)
@test isapprox(j_get, ∂y_∂u; atol=atol)

@test c.solution.evals_∂ẋ_∂x == 0
@test c.solution.evals_∂ẋ_∂u == 0
@test c.solution.evals_∂ẋ_∂p == 0
@test c.solution.evals_∂ẋ_∂t == 0

@test c.solution.evals_∂y_∂x == 5
@test c.solution.evals_∂y_∂u == 5
@test c.solution.evals_∂y_∂p == 0
@test c.solution.evals_∂y_∂t == 0

@test c.solution.evals_∂e_∂x == 0
@test c.solution.evals_∂e_∂u == 0
@test c.solution.evals_∂e_∂p == 0
@test c.solution.evals_∂e_∂t == 0
reset!(c)

# explicit time derivative ∂dx/∂t
_f = _t -> fmu(;x=x, t=_t, dx=dx)
_f(t)
j_fwd = ForwardDiff.derivative(_f, t)
j_zyg = Zygote.jacobian(_f, t)[1]

# ReverseDiff has no `derivative` function for scalars
_f = _t -> fmu(;x=x, t=_t[1], dx=dx)
j_rwd = ReverseDiff.jacobian(_f, [t])

@test isapprox(j_fwd, ∂x_∂t; atol=atol)
@test isapprox(j_zyg, ∂x_∂t; atol=atol)
@test isapprox(j_rwd, ∂x_∂t; atol=atol)

@test c.solution.evals_∂ẋ_∂x == 0
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

# explicit time derivative ∂y/∂t 
_f = _t -> fmu(;x=x, y=y, y_refs=y_refs, t=_t)
_f(t)
j_fwd = ForwardDiff.derivative(_f, t)
j_zyg = Zygote.jacobian(_f, t)[1]

# ReverseDiff has no `derivative` function for scalars
_f = _t -> fmu(;x=x, y=y, y_refs=y_refs, t=_t[1])
j_rwd = ReverseDiff.jacobian(_f, [t])

@test isapprox(j_fwd, ∂y/∂t; atol=atol)
@test isapprox(j_zyg, ∂y/∂t; atol=atol)
@test isapprox(j_rwd, ∂y/∂t; atol=atol)

@test c.solution.evals_∂ẋ_∂x == 0
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

# Jacobian ∂ẋ/∂p
_f = _p -> fmu(;x=x, p=_p, p_refs=p_refs, dx=dx)
_f(p)
j_fwd = ForwardDiff.jacobian(_f, p)
j_zyg = Zygote.jacobian(_f, p)[1]
j_rwd = ReverseDiff.jacobian(_f, p)
j_smp = fmi2SampleJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.parameterValueReferences)
j_get = fmi2GetJacobian(c, fmu.modelDescription.derivativeValueReferences, fmu.modelDescription.parameterValueReferences)

@test isapprox(j_fwd, ∂ẋ/∂p; atol=atol)
@test isapprox(j_zyg, ∂ẋ/∂p; atol=atol)
@test isapprox(j_rwd, ∂ẋ/∂p; atol=atol)
@test isapprox(j_smp, ∂ẋ/∂p; atol=atol)
@test isapprox(j_get, ∂ẋ/∂p; atol=atol)

reset!(c)

# Jacobian ∂y/∂p
_f = _p -> fmu(;p=_p, p_refs=p_refs, y=y, y_refs=y_refs)
_f(p)
j_fwd = ForwardDiff.jacobian(_f, p)
j_zyg = Zygote.jacobian(_f, p)[1]
j_rwd = ReverseDiff.jacobian(_f, p)
j_smp = fmi2SampleJacobian(c, fmu.modelDescription.outputValueReferences, fmu.modelDescription.parameterValueReferences)
j_get = fmi2GetJacobian(c, fmu.modelDescription.outputValueReferences, fmu.modelDescription.parameterValueReferences)

@test isapprox(j_fwd, ∂y/∂p; atol=atol)
@test isapprox(j_zyg, ∂y/∂p; atol=atol)
@test isapprox(j_rwd, ∂y/∂p; atol=atol)
@test isapprox(j_smp, ∂y/∂p; atol=atol)
@test isapprox(j_get, ∂y/∂p; atol=atol)

reset!(c)

# clean up
fmi2Unload(fmu)

########## Event Indicators Check ###########

# load demo FMU
fmu = fmi2Load("VLDM", "Dymola", "2020x"; type=:ME)

# enable time gradient evaluation (disabled by default for performance reasons)
fmu.executionConfig.eval_t_gradients = true

# prepare (allocate) an FMU instance
c, x0 = FMIImport.prepareSolveFMU(fmu, nothing, fmu.type, nothing, nothing, nothing, nothing, nothing, nothing, 0.0, 0.0, nothing)

x_refs = fmu.modelDescription.stateValueReferences
x = zeros(length(x_refs))
u_refs = fmu.modelDescription.inputValueReferences
u = zeros(length(u_refs))
y_refs = fmu.modelDescription.outputValueReferences
y = zeros(length(y_refs))
p_refs = fmu.modelDescription.parameterValueReferences
p = zeros(length(p_refs))
dx = zeros(length(x_refs))
ec = zeros(fmi2Real, fmu.modelDescription.numberOfEventIndicators)
t = 0.0

# Jacobian ∂e/∂x
_f = _x -> fmu(;ec=ec, x=_x)
_f(x)
j_fwd = ForwardDiff.jacobian(_f, x)
j_zyg = Zygote.jacobian(_f, x)[1]
j_rwd = ReverseDiff.jacobian(_f, x)
j_smp = fmi2SampleJacobian(c, :indicators, fmu.modelDescription.parameterValueReferences)

# @test isapprox(j_fwd, F; atol=atol)
# @test isapprox(j_zyg, F; atol=atol)
# @test isapprox(j_rwd, F; atol=atol)
# @test isapprox(j_smp, F; atol=atol)
# @test isapprox(j_get, F; atol=atol)

@test c.solution.evals_∂ẋ_∂x == 0
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

# clean up
fmi2Unload(fmu)