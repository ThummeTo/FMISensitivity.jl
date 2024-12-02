#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# this is temporary until it's implemented in native Julia, see:
# https://discourse.julialang.org/t/debug-has-massive-performance-impact/103974/19
using Logging 
if Sys.iswindows()
    Logging.disable_logging(Logging.Debug)
end

using BenchmarkTools
using FMI.FMIImport.FMIBase: NO_fmi2Real

c, fmu = getFMUStruct("BouncingBall1D", :ME, "Dymola", "2022x")

function evalBenchmark(b)
    res = run(b)
    min_time = min(res.times...)
    memory = res.memory 
    allocs = res.allocs
    return min_time, memory, allocs 
end

########## f(x) evaluation / right-hand side ########## 

#c.solution = FMUSolution(c)
#fmi2Reset(c)
#fmi2EnterInitializationMode(c)
#fmi2ExitInitializationMode(c)

cRef = UInt64(pointer_from_objref(c))
dx = zeros(fmi2Real, 2)
dx_refs = c.fmu.modelDescription.derivativeValueReferences
y = zeros(fmi2Real, 0)
y_refs = zeros(fmi2ValueReference, 0)
x = zeros(fmi2Real, 2)
u = zeros(fmi2Real, 0)
u_refs = zeros(fmi2ValueReference, 0)
p = zeros(fmi2Real, 0)
p_refs = zeros(fmi2ValueReference, 0)
ec = zeros(fmi2Real, 0)
ec_idcs = zeros(fmi2ValueReference, 0)
t = NO_fmi2Real

b = @benchmarkable FMI.eval!($cRef, $dx, $dx_refs, $y, $y_refs, $x, $u, $u_refs, $p, $p_refs, $ec, $ec_idcs, $t)
min_time, memory, allocs = evalBenchmark(b)
@test allocs <= 0
@test memory <= 0

b = @benchmarkable $c(dx=$dx, y=$y, y_refs=$y_refs, x=$x, u=$u, u_refs=$u_refs, p=$p, p_refs=$p_refs, ec=$ec, ec_idcs=$ec_idcs, t=$t)
min_time, memory, allocs = evalBenchmark(b)
@test allocs <= 9   # `ignore_derivatives` causes an extra 3 allocations (48 bytes)
@test memory <= 224  # ToDo: What are the remaining allocations compared to `eval!`?

using FMISensitivity
import FMISensitivity.ForwardDiff
import FMISensitivity.ReverseDiff
function fun(_x)
    FMI.eval!(cRef, dx, dx_refs, y, y_refs, _x, u, u_refs, p, p_refs, ec, ec_idcs, t)
end
config = ForwardDiff.JacobianConfig(fun, x, ForwardDiff.Chunk{length(x)}())

b = @benchmarkable ForwardDiff.jacobian($fun, $x, $config)
min_time, memory, allocs = evalBenchmark(b)
# ToDo: This is too much!
@test allocs <= 250
@test memory <= 13600

b = @benchmarkable ReverseDiff.jacobian($fun, $x)
min_time, memory, allocs = evalBenchmark(b)
# ToDo: This is too much!
@test allocs <= 180
@test memory <= 11400