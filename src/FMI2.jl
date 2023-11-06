#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMICore: fmi2ValueReference, eval!

function ddSupported(c::FMU2Component)
    if !isnothing(c.fmu.modelDescription.modelExchange)
        if c.fmu.modelDescription.modelExchange.providesDirectionalDerivative == true
            return true 
        end
    elseif !isnothing(c.fmu.modelDescription.coSimulation)
        if c.fmu.modelDescription.coSimulation.providesDirectionalDerivative == true
            return true
        end
    end
    return false
end

# in FMI2 we can use fmi2GetDirectionalDerivative for JVP-computations
function fmi2JVP!(c::FMU2Component, mtxCache::Symbol, ∂f_refs, ∂x_refs, x, seed)

    jac = getfield(c, mtxCache)
    if isnothing(jac)
        jac = FMU2Jacobian{typeof(seed[1])}(c, ∂f_refs, ∂x_refs)
        setfield!(c, mtxCache, jac)
    end

    if c.fmu.executionConfig.JVPBuiltInDerivatives && ddSupported(c) && !isa(jac.f_refs, Tuple) && !isa(jac.x_refs, Symbol)
        fmi2GetDirectionalDerivative!(c, ∂f_refs, ∂x_refs, jac.vjp, seed)
        return jac.vjp
    else
        return jvp!(jac, x, seed)
    end
end

function fmi2GVP!(c::FMU2Component, mtxCache::Symbol, ∂f_refs, ∂x_refs, x, seed)

    grad = getfield(c, mtxCache)
    if isnothing(grad)
        grad = FMU2Gradient{typeof(seed[1])}(c, ∂f_refs, ∂x_refs)
        setfield!(c, mtxCache, grad)
    end

    if c.fmu.executionConfig.JVPBuiltInDerivatives && ddSupported(c) && !isa(grad.f_refs, Tuple) && !isa(grad.x_refs, Symbol)
        fmi2GetDirectionalDerivative!(c, ∂f_refs, ∂x_refs, grad.gvp, [seed])
        return grad.gvp
    else
        return gvp!(grad, x, seed)
    end
end

# in FMI2 there is no helper for VJP-computations (but in FMI3) ...
function fmi2VJP!(c::FMU2Component, mtxCache::Symbol, ∂f_refs, ∂x_refs, x, seed)

    jac = getfield(c, mtxCache)
    if isnothing(jac)
        jac = FMU2Jacobian{typeof(seed[1])}(c, ∂f_refs, ∂x_refs)
        setfield!(c, mtxCache, jac)
    end
    
    return vjp!(jac, x, seed)
end

function fmi2VGP!(c::FMU2Component, mtxCache::Symbol, ∂f_refs, ∂x_refs, x, seed)

    grad = getfield(c, mtxCache)
    if isnothing(grad)
        grad = FMU2Gradient{typeof(seed[1])}(c, ∂f_refs, ∂x_refs)
        setfield!(c, mtxCache, grad)
    end
    
    return vgp!(grad, x, seed)
end

function ChainRulesCore.frule(Δtuple, 
    ::typeof(eval!), 
    cRef, 
    dx,
    dx_refs,
    y,
    y_refs, 
    x,
    u,
    u_refs,
    p,
    p_refs, 
    ec, 
    ec_idcs,
    t)

    Δself, ΔcRef, Δdx, Δdx_refs, Δy, Δy_refs, Δx, Δu, Δu_refs, Δp, Δp_refs, Δec, Δec_idcs, Δt = undual(Δtuple)

    ### ToDo: Somehow, ForwardDiff enters with all types beeing Float64, this needs to be corrected.

    cRef = undual(cRef)
    if typeof(cRef) != UInt64
        cRef = UInt64(cRef)
    end
    
    # ToDo: is this necessary?
    t = undual(t)
    u = undual(u)
    x = undual(x)
    p = undual(p)

    dx_refs = undual(dx_refs)
    dx_refs = convert(Array{UInt32,1}, dx_refs)
    
    y_refs = undual(y_refs)
    y_refs = convert(Array{UInt32,1}, y_refs)
    
    u_refs = undual(u_refs)
    u_refs = convert(Array{UInt32,1}, u_refs)
    
    p_refs = undual(p_refs)
    p_refs = convert(Array{UInt32,1}, p_refs)

    ec_idcs = undual(ec_idcs)
    ec_idcs = convert(Array{UInt32,1}, ec_idcs) 
    
    ###

    c = unsafe_pointer_to_objref(Ptr{Nothing}(cRef))

    outputs = (length(y_refs) > 0)
    inputs = (length(u_refs) > 0)
    derivatives = (length(dx) > 0)
    states = (length(x) > 0)
    times = (t >= 0.0)
    parameters = (length(p_refs) > 0)
    eventIndicators = (length(ec) > 0)

    Ω = eval!(cRef, dx, dx_refs, y, y_refs, x, u, u_refs, p, p_refs, ec, ec_idcs, t)
    
    # time, states and inputs where already set in `eval!`, no need to repeat it here

    ∂y = ZeroTangent()
    ∂dx = ZeroTangent()
    ∂e = ZeroTangent()

    if length(dx_refs) == 0 # all derivatives, please!
        dx_refs = c.fmu.modelDescription.derivativeValueReferences
    end

    if Δx != NoTangent() && length(Δx) > 0

        if !isa(Δx, AbstractVector{fmi2Real})
            Δx = convert(Vector{fmi2Real}, Δx)
        end

        if states
            if derivatives
                ∂dx += fmi2JVP!(c, :∂ẋ_∂x, dx_refs, c.fmu.modelDescription.stateValueReferences, x, Δx)
                c.solution.evals_∂ẋ_∂x += 1
            end

            if outputs 
                ∂y += fmi2JVP!(c, :∂y_∂x, y_refs, c.fmu.modelDescription.stateValueReferences, x, Δx)
                c.solution.evals_∂y_∂x += 1
            end

            if eventIndicators
                ∂e += fmi2JVP!(c, :∂e_∂x, (:indicators, ec_idcs), c.fmu.modelDescription.stateValueReferences, x, Δx)
                c.solution.evals_∂e_∂x += 1
            end
        end
    end

    
    if Δu != NoTangent() && length(Δu) > 0

        if !isa(Δu, AbstractVector{fmi2Real})
            Δu = convert(Vector{fmi2Real}, Δu)
        end

        if inputs
            if derivatives
                ∂dx += fmi2JVP!(c, :∂ẋ_∂u, dx_refs, u_refs, u, Δu)
                c.solution.evals_∂ẋ_∂u += 1
            end

            if outputs
                ∂y += fmi2JVP!(c, :∂y_∂u, y_refs, u_refs, u, Δu)
                c.solution.evals_∂y_∂u += 1
            end

            if eventIndicators
                ∂e += fmi2JVP!(c, :∂e_∂u, (:indicators, ec_idcs), u_refs, u, Δu)
                c.solution.evals_∂e_∂u += 1
            end
        end
    end

    if Δp != NoTangent() && length(Δp) > 0

        if !isa(Δp, AbstractVector{fmi2Real})
            Δp = convert(Vector{fmi2Real}, Δp)
        end

        if parameters
            if derivatives
                ∂dx += fmi2JVP!(c, :∂ẋ_∂p, dx_refs, p_refs, p, Δp)
                c.solution.evals_∂ẋ_∂p += 1
            end

            if outputs
                ∂y += fmi2JVP!(c, :∂y_∂p, y_refs, p_refs, p, Δp)
                c.solution.evals_∂y_∂p += 1
            end

            if eventIndicators
                ∂e += fmi2JVP!(c, :∂e_∂p, (:indicators, ec_idcs), p_refs, p, Δp)
                c.solution.evals_∂e_∂p += 1
            end
        end
    end

    if Δt != NoTangent() && c.fmu.executionConfig.eval_t_gradients

        if times 
            if derivatives
                ∂dx += fmi2GVP!(c, :∂ẋ_∂t, dx_refs, :time, t, Δt)
                c.solution.evals_∂ẋ_∂t += 1
            end

            if outputs
                ∂y += fmi2GVP!(c, :∂y_∂t, y_refs, :time, t, Δt)
                c.solution.evals_∂y_∂t += 1
            end

            if eventIndicators
                ∂e += fmi2GVP!(c, :∂e_∂t, (:indicators, ec_idcs), :time, t, Δt)
                c.solution.evals_∂e_∂t += 1
            end
        end
    end

    @debug "frule:   ∂y=$(∂y)   ∂dx=$(∂dx)   ∂e=$(∂e)"

    ∂Ω = nothing
    if c.fmu.executionConfig.concat_eval
        ∂Ω = vcat(∂y, ∂dx, ∂e) # [∂y..., ∂dx..., ∂e...]
    else
        ∂Ω = (∂y, ∂dx, ∂e) 
    end

    return Ω, ∂Ω 
end

function ChainRulesCore.rrule(::typeof(eval!), 
    cRef, 
    dx,
    dx_refs, 
    y,
    y_refs, 
    x,
    u,
    u_refs,
    p,
    p_refs,
    ec, 
    ec_idcs, 
    t)

    @assert !isa(cRef, FMU2Component) "Wrong dispatched!"
      
    c = unsafe_pointer_to_objref(Ptr{Nothing}(cRef))
    
    outputs = (length(y_refs) > 0)
    inputs = (length(u_refs) > 0)
    derivatives = (length(dx) > 0)
    states = (length(x) > 0)
    times = (t >= 0.0)
    parameters = (length(p_refs) > 0)
    eventIndicators = (length(ec) > 0)

    Ω = eval!(cRef, dx, dx_refs, y, y_refs, x, u, u_refs, p, p_refs, ec, ec_idcs, t)

    ##############

    function eval_pullback(r̄)

        ȳ = nothing 
        d̄x = nothing
        ēc = nothing

        if c.fmu.executionConfig.concat_eval
            ylen = (isnothing(y_refs) ? 0 : length(y_refs))
            dxlen = length(dx)
            ȳ = r̄[1:ylen]
            d̄x = r̄[ylen+1:ylen+dxlen]
            ēc = r̄[ylen+dxlen+1:end]
        else
            ȳ, d̄x, ēc = r̄
        end

        outputs = outputs && !isZeroTangent(ȳ)
        derivatives = derivatives && !isZeroTangent(d̄x)
        eventIndicators = eventIndicators && !isZeroTangent(ēc)

        if !isa(ȳ, AbstractArray)
            ȳ = collect(ȳ) # [ȳ...]
        end

        if !isa(d̄x, AbstractArray)
            d̄x = collect(d̄x) # [d̄x...]
        end

        if !isa(ēc, AbstractArray)
            ēc = collect(ēc) # [ēc...]
        end

        # [ToDo] Is the following statement correct?
        #        Between building and using the pullback, maybe the time, state or inputs were changed, so we need to set them again.
       
        if states && c.x != x && !c.fmu.isZeroState
            fmi2SetContinuousStates(c, x)
        end

        if inputs ## && c.u != u
            fmi2SetReal(c, u_refs, u)
        end

        if parameters && c.fmu.executionConfig.set_p_every_step
            fmi2SetReal(c, p_refs, p)
        end

        if times && c.t != t
            fmi2SetTime(c, t)
        end

        n_dx_x = ZeroTangent()
        n_dx_u = ZeroTangent()
        n_dx_p = ZeroTangent()
        n_dx_t = ZeroTangent()

        n_y_x = ZeroTangent()
        n_y_u = ZeroTangent()
        n_y_p = ZeroTangent()
        n_y_t = ZeroTangent()

        n_ec_x = ZeroTangent()
        n_ec_u = ZeroTangent()
        n_ec_p = ZeroTangent()
        n_ec_t = ZeroTangent()

        @debug "rrule pullback d̄x, ȳ, ēc = $(d̄x), $(ȳ), $(ēc)"

        if length(dx_refs) == 0 # all derivatives, please!
            dx_refs = c.fmu.modelDescription.derivativeValueReferences
        end
        x_refs = c.fmu.modelDescription.stateValueReferences

        if derivatives 
            if states
                n_dx_x = fmi2VJP!(c, :∂ẋ_∂x, dx_refs, x_refs, x, d̄x) 
                c.solution.evals_∂ẋ_∂x += 1
            end

            if inputs
                n_dx_u = fmi2VJP!(c, :∂ẋ_∂u, dx_refs, u_refs, u, d̄x) 
                c.solution.evals_∂ẋ_∂u += 1
            end

            if parameters
                n_dx_p = fmi2VJP!(c, :∂ẋ_∂p, dx_refs, p_refs, p, d̄x) 
                c.solution.evals_∂ẋ_∂p += 1
            end

            if times && c.fmu.executionConfig.eval_t_gradients
                n_dx_t = fmi2VGP!(c, :∂ẋ_∂t, dx_refs, :time, t, d̄x) 
                c.solution.evals_∂ẋ_∂t += 1
            end
        end

        if outputs 
            if states
                n_y_x = fmi2VJP!(c, :∂y_∂x, y_refs, x_refs, x, ȳ) 
                c.solution.evals_∂y_∂x += 1
            end
        
            if inputs
                n_y_u = fmi2VJP!(c, :∂y_∂u, y_refs, u_refs, u, ȳ) 
                c.solution.evals_∂y_∂u += 1
            end

            if parameters
                n_y_p = fmi2VJP!(c, :∂y_∂p, y_refs, p_refs, p, ȳ) 
                c.solution.evals_∂y_∂p += 1
            end

            if times && c.fmu.executionConfig.eval_t_gradients
                n_y_t = fmi2VGP!(c, :∂y_∂t, y_refs, :time, t, ȳ)
                c.solution.evals_∂y_∂t += 1
            end
        end

        if eventIndicators
            if states
                n_ec_x = fmi2VJP!(c, :∂e_∂x, (:indicators, ec_idcs), x_refs, x, ēc) 
                c.solution.evals_∂e_∂x += 1
            end

            if inputs
                n_ec_u = fmi2VJP!(c, :∂e_∂u, (:indicators, ec_idcs), u_refs, u, ēc) 
                c.solution.evals_∂e_∂u += 1
            end

            if parameters
                n_ec_p = fmi2VJP!(c, :∂e_∂p, (:indicators, ec_idcs), p_refs, p, ēc) 
                c.solution.evals_∂e_∂p += 1
            end

            if times && c.fmu.executionConfig.eval_t_gradients
                n_ec_t = fmi2VGP!(c, :∂e_∂t, (:indicators, ec_idcs), :time, t, ēc) 
                c.solution.evals_∂e_∂t += 1
            end
        end

        # write back
        f̄ = NoTangent()
        c̄Ref = ZeroTangent()
        d̄x = ZeroTangent()
        d̄x_refs = ZeroTangent()
        ȳ = ZeroTangent()
        ȳ_refs = ZeroTangent()
        ēc = ZeroTangent()
        ēc_idcs = ZeroTangent()

        t̄ = n_y_t + n_dx_t + n_ec_t

        x̄ = n_y_x + n_dx_x + n_ec_x
        
        ū = n_y_u + n_dx_u + n_ec_u
        ū_refs = ZeroTangent()

        p̄ = n_y_p + n_dx_p + n_ec_p
        p̄_refs = ZeroTangent()
        
        @debug "rrule:   $((f̄, c̄Ref, d̄x, ȳ, ȳ_refs, x̄, ū, ū_refs, p̄, p̄_refs, ēc, ēc_idcs, t̄))"

        return (f̄, c̄Ref, d̄x, d̄x_refs, ȳ, ȳ_refs, x̄, ū, ū_refs, p̄, p̄_refs, ēc, ēc_idcs, t̄)
    end

    return (Ω, eval_pullback)
end

# dx, y, x, u, t
@ForwardDiff_frule eval!(cRef::UInt64, 
    dx    ::AbstractVector{<:ForwardDiff.Dual},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:ForwardDiff.Dual},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:ForwardDiff.Dual},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmi2ValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules eval!(cRef::UInt64, 
    dx    ::AbstractVector{<:ReverseDiff.TrackedReal},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:ReverseDiff.TrackedReal},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:ReverseDiff.TrackedReal},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::ReverseDiff.TrackedReal)

# x, p
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmi2ValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

# t
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmi2ValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::ReverseDiff.TrackedReal)

# x
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmi2ValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

# u
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:ForwardDiff.Dual},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmi2ValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:ReverseDiff.TrackedReal},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real}, 
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

# p
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmi2ValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

# ec
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmi2ValueReference},
    ec    ::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

# x, t
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmi2ValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::ReverseDiff.TrackedReal)

# x, ec, t
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmi2ValueReference},
    ec    ::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::ReverseDiff.TrackedReal)

# x, ec
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmi2ValueReference},
    ec    ::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

# x, p, t
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmi2ValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::ReverseDiff.TrackedReal)

# x, p, ec, t
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmi2ValueReference},
    ec    ::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::ReverseDiff.TrackedReal)

# x, p, ec
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmi2ValueReference},
    ec    ::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmi2ValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

# FiniteDiff Jacobians

abstract type FMU2Sensitivities end

mutable struct FMU2Jacobian{C, T, F} <: FMU2Sensitivities
    valid::Bool
    colored::Bool
    component::C

    mtx::Matrix{T}
    jvp::Vector{T}
    vjp::Vector{T}

    f_refs::Union{Vector{UInt32}, Tuple{Symbol, Vector{UInt32}}}
    x_refs::Union{Vector{UInt32}, Symbol}
    f_refs_set::Union{Set, Nothing}

    f::F 

    #cache::FiniteDiff.JacobianCache
    #colors::

    function FMU2Jacobian{T}(component::C, f_refs::Union{Vector{UInt32}, Tuple{Symbol, Vector{UInt32}}}, x_refs::Union{Vector{UInt32}, Symbol}) where {C, T}

        @assert !isa(f_refs, Tuple) || f_refs[1] == :indicators "`f_refs` is Tuple, it must be `:indicators`"
        @assert !isa(x_refs, Symbol) || x_refs == :time "`x_refs` is Symbol, it must be `:time`"

        f_len = 0
        x_len = 0
        f_refs_set = nothing
        f = nothing

        if isa(f_refs, Tuple)
            f_len = length(f_refs[2]) # number of event indicators to capture
            x_len = length(x_refs)
            f = f_∂e_∂v
        else
            f_len = length(f_refs)
            x_len = length(x_refs)
            f_refs_set = Set(f_refs)
            f = f_∂v_∂v
        end

        F = typeof(f)

        inst = new{C, T, F}()
        inst.f = f
        inst.component = component
        inst.f_refs = f_refs
        inst.f_refs_set = f_refs_set
        inst.x_refs = x_refs
        
        inst.mtx = zeros(T, f_len, x_len)
        inst.jvp = zeros(T, f_len)
        inst.vjp = zeros(T, x_len)

        inst.valid = false
        inst.colored = false
        
        return inst
    end

end 

mutable struct FMU2Gradient{C, T, F} <: FMU2Sensitivities
    valid::Bool
    colored::Bool
    component::C

    vec::Vector{T}
    gvp::Vector{T}
    vgp::Vector{T}

    f_refs::Union{Vector{UInt32}, Tuple{Symbol, Vector{UInt32}}}
    x_refs::Union{Vector{UInt32}, Symbol}
    f_refs_set::Union{Set, Nothing}

    f::F 

    #cache::FiniteDiff.GradientCache
    #colors::

    function FMU2Gradient{T}(component::C, f_refs::Union{Vector{UInt32}, Tuple{Symbol, Vector{UInt32}}}, x_refs::Union{UInt32, Symbol}) where {C, T}

        @assert !isa(f_refs, Tuple) || f_refs[1] == :indicators "`f_refs` is Tuple, it must be `:indicators`"
        @assert !isa(x_refs, Symbol) || x_refs == :time "`x_refs` is Symbol, it must be `:time`"

        f_len = 0
        x_len = 1
        f_refs_set = nothing
        f = nothing

        if isa(f_refs, Tuple)
            f_len = length(f_refs[2])
            f = f_∂e_∂t
        else
            f_len = length(f_refs)
            f_refs_set = Set(f_refs)
            f = f_∂v_∂t
        end

        F = typeof(f)

        inst = new{C, T, F}()
        inst.f = f
        inst.component = component
        inst.f_refs = f_refs
        inst.f_refs_set = f_refs_set
        inst.x_refs = x_refs
        
        inst.vec = zeros(T, f_len)
        inst.gvp = zeros(T, f_len)
        inst.vgp = zeros(T, x_len)

        inst.valid = false
        inst.colored = false
        
        return inst
    end

end 

function f_∂v_∂v(jac::FMU2Jacobian, dx, x)
    fmi2SetReal(jac.component, jac.x_refs, x; track=false)
    fmi2GetReal!(jac.component, jac.f_refs, dx)
    return dx
end

function f_∂e_∂v(jac::FMU2Jacobian, dx, x)
    fmi2SetReal(jac.component, jac.x_refs, x; track=false)
    fmi2GetEventIndicators!(jac.component, jac.component.eventIndicatorBuffer)
    dx[:] = jac.component.eventIndicatorBuffer[jac.f_refs[2]]
    return dx
end

function f_∂e_∂t(jac::FMU2Gradient, dx, x)
    fmi2SetTime(jac.component, x; track=false)
    fmi2GetEventIndicators!(jac.component, jac.component.eventIndicatorBuffer)
    dx[:] = jac.component.eventIndicatorBuffer[jac.f_refs[2]]
    return dx
end

function f_∂v_∂t(jac::FMU2Gradient, dx, x)
    fmi2SetTime(jac.component, x; track=false)
    fmi2GetReal!(jac.component, jac.f_refs, dx)
    return dx
end

function invalidate!(sens::FMU2Sensitivities)
    sens.valid = false 
    return nothing 
end

function check_invalidate!(vrs, sens::FMU2Sensitivities)
    if !sens.valid
        return 
    end

    if isnothing(sens.f_refs_set)
        return 
    end

    for vr ∈ vrs
        if vr ∈ sens.f_refs_set 
            invalidate!(sens)
        end
    end

    return nothing 
end

function uncolor!(jac::FMU2Sensitivities)
    jac.colored = false 
    return nothing 
end

function validate!(jac::FMU2Jacobian, x::AbstractVector)

    if jac.component.fmu.executionConfig.sensitivity_strategy == :FMIDirectionalDerivative && ddSupported(jac.component) && !isa(jac.f_refs, Tuple) && !isa(jac.x_refs, Symbol)
        # ToDo: use directional derivatives with sparsitiy information!
        for i in 1:length(jac.x_refs)
            fmi2GetDirectionalDerivative!(jac.component, jac.f_refs, [jac.x_refs[i]], view(jac.mtx, 1:length(jac.f_refs), i))
        end
    else #if jac.component.fmu.executionConfig.sensitivity_strategy == :FiniteDiff
        # cache = FiniteDiff.JacobianCache(x)
        FiniteDiff.finite_difference_jacobian!(jac.mtx, (_x, _dx) -> (jac.f(jac, _x, _dx)), x) # , cache)
    end

    jac.valid = true 
    return nothing
end

function validate!(grad::FMU2Gradient, x::Real)

    if grad.component.fmu.executionConfig.sensitivity_strategy == :FMIDirectionalDerivative && ddSupported(grad.component) && !isa(grad.f_refs, Tuple) && !isa(grad.x_refs, Symbol)
        # ToDo: use directional derivatives with sparsitiy information!
        fmi2GetDirectionalDerivative!(grad.component, grad.f_refs, grad.x_refs, grad.vec)
    else #if grad.component.fmu.executionConfig.sensitivity_strategy == :FiniteDiff
        # cache = FiniteDiff.GradientCache(x)
        FiniteDiff.finite_difference_gradient!(grad.vec, (_x, _dx) -> (grad.f(grad, _x, _dx)), x) # , cache)
    end

    grad.valid = true 
    return nothing
end
    
function color!(jac::FMU2Sensitivities)
    # ToDo
    # colors = SparseDiffTools.matrix_colors(sparsejac)

    jac.colored = true 
    return nothing
end

function update!(jac::FMU2Sensitivities, x)
    if !jac.valid
        validate!(jac, x)
    end

    if !jac.colored
        color!(jac)
    end
    return nothing
end

function jvp!(jac::FMU2Jacobian, x::AbstractVector, v::AbstractVector)
    update!(jac, x)
    return mul!(jac.jvp, jac.mtx, v)
end

function vjp!(jac::FMU2Jacobian, x::AbstractVector, v::AbstractVector)
    update!(jac, x)
    return mul!(jac.vjp, jac.mtx', v)
end

function gvp!(grad::FMU2Gradient, x, v)
    update!(grad, x)
    return mul!(grad.gvp, grad.vec, v)
end

function vgp!(grad::FMU2Gradient, x, v)
    update!(grad, x)
    mul!(grad.vgp, grad.vec', v) 
    return grad.vgp[1]
end

###

import SciMLSensitivity.Zygote: grad_mut, Context
import FMICore: FMU2EvaluationOutput
#grad_mut(av::AbstractVector) = invoke(grad_mut, Tuple{Any}, av)
grad_mut(av::FMU2EvaluationOutput) = invoke(grad_mut, Tuple{Any}, av)
#grad_mut(c::Zygote.Context, av::AbstractVector) = invoke(grad_mut, Tuple{Zygote.Context, Any}, c, av)
grad_mut(c::Zygote.Context, av::FMU2EvaluationOutput) = invoke(grad_mut, Tuple{Zygote.Context, Any}, c, av)

#grad_mut(av::AbstractVector) = []