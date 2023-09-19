#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMICore: fmi2ValueReference, eval!

# in FMI2 we can use fmi2GetDirectionalDerivative for JVP-computations
function fmi2JVP!(c::FMU2Component, mtxCache::Symbol, ∂f_refs, ∂x_refs, seed)

    if c.fmu.executionConfig.JVPBuiltInDerivatives # && fmi2ProvidesDirectionalDerivative(c.fmu.modelDescription)
        jac = getfield(c, mtxCache)
        if jac.b == nothing || size(jac.b) != (length(seed),)
            jac.b = zeros(length(seed))
        end 

        fmi2GetDirectionalDerivative!(c, ∂f_refs, ∂x_refs, jac.b, seed)
        return jac.b
    else
        jac = getfield(c, mtxCache)
        
        return FMICore.jvp!(jac, seed; ∂f_refs=∂f_refs, ∂x_refs=∂x_refs)
    end
end

# in FMI2 there is no helper for VJP-computations (but in FMI3) ...
function fmi2VJP!(c::FMU2Component, mtxCache::Symbol, ∂f_refs, ∂x_refs, seed)

    jac = getfield(c, mtxCache)  
    return FMICore.vjp!(jac, seed; ∂f_refs=∂f_refs, ∂x_refs=∂x_refs)
end

function ChainRulesCore.frule(Δtuple, 
    ::typeof(eval!), 
    cRef, 
    dx,
    y,
    y_refs, 
    x,
    u,
    u_refs,
    p,
    p_refs, 
    t)

    Δself, ΔcRef, Δdx, Δy, Δy_refs, Δx, Δu, Δu_refs, Δp, Δp_refs, Δt = undual(Δtuple)

    ### ToDo: Somehow, ForwardDiff enters with all types beeing Float64, this needs to be corrected.

    cRef = undual(cRef)
    if typeof(cRef) != UInt64
        cRef = UInt64(cRef)
    end
    
    t = undual(t)
    u = undual(u)
    x = undual(x)
    p = undual(p)
    
    y_refs = undual(y_refs)
    y_refs = convert(Array{UInt32,1}, y_refs)
    
    u_refs = undual(u_refs)
    u_refs = convert(Array{UInt32,1}, u_refs)
    
    p_refs = undual(p_refs)
    p_refs = convert(Array{UInt32,1}, p_refs)
    
    ###

    c = unsafe_pointer_to_objref(Ptr{Nothing}(cRef))

    outputs = (length(y_refs) > 0)
    inputs = (length(u_refs) > 0)
    derivatives = (length(dx) > 0)
    states = (length(x) > 0)
    times = (t >= 0.0)
    parameters = (length(p_refs) > 0)

    Ω = eval!(cRef, dx, y, y_refs, x, u, u_refs, p, p_refs, t)
    
    # time, states and inputs where already set in `eval!`, no need to repeat it here

    ∂y = ZeroTangent()
    ∂dx = ZeroTangent()

    if Δx != NoTangent() && length(Δx) > 0

        if !isa(Δx, AbstractVector{fmi2Real})
            Δx = convert(Vector{fmi2Real}, Δx)
        end

        if states
            if derivatives
                ∂dx += fmi2JVP!(c, :A, c.fmu.modelDescription.derivativeValueReferences, c.fmu.modelDescription.stateValueReferences, Δx)
                c.solution.evals_∂ẋ_∂x += 1
                #@info "$(Δx)"
            end

            if outputs 
                ∂y += fmi2JVP!(c, :C, y_refs, c.fmu.modelDescription.stateValueReferences, Δx)
                c.solution.evals_∂y_∂x += 1
            end
        end
    end

    
    if Δu != NoTangent() && length(Δu) > 0

        if !isa(Δu, AbstractVector{fmi2Real})
            Δu = convert(Vector{fmi2Real}, Δu)
        end

        if inputs
            if derivatives
                ∂dx += fmi2JVP!(c, :B, c.fmu.modelDescription.derivativeValueReferences, u_refs, Δu)
                c.solution.evals_∂ẋ_∂u += 1
            end

            if outputs
                ∂y += fmi2JVP!(c, :D, y_refs, u_refs, Δu)
                c.solution.evals_∂y_∂u += 1
            end
        end
    end

    if Δp != NoTangent() && length(Δp) > 0

        if !isa(Δp, AbstractVector{fmi2Real})
            Δp = convert(Vector{fmi2Real}, Δp)
        end

        if parameters
            if derivatives
                ∂dx += fmi2JVP!(c, :E, c.fmu.modelDescription.derivativeValueReferences, p_refs, Δp)
                c.solution.evals_∂ẋ_∂p += 1
            end

            if outputs
                ∂y += fmi2JVP!(c, :F, y_refs, p_refs, Δp)
                c.solution.evals_∂y_∂p += 1
            end
        end
    end

    if c.fmu.executionConfig.eval_t_gradients
        # partial time derivatives are not part of the FMI standard, so must be sampled in any case
        if Δt != NoTangent() && times && (derivatives || outputs)

            dt = 1e-6 # ToDo: Find a better value, e.g. based on the current solver step size

            dx1 = nothing
            dx2 = nothing
            y1 = nothing
            y2 = nothing 

            if derivatives
                dx1 = zeros(fmi2Real, length(c.fmu.modelDescription.derivativeValueReferences))
                dx2 = zeros(fmi2Real, length(c.fmu.modelDescription.derivativeValueReferences))
                fmi2GetDerivatives!(c, dx1)
            end

            if outputs
                y1 = zeros(fmi2Real, length(y))
                y2 = zeros(fmi2Real, length(y))
                fmi2GetReal!(c, y_refs, y1)
            end

            fmi2SetTime(c, t + dt; track=false)

            if derivatives
                fmi2GetDerivatives!(c, dx2)

                ∂dx_t = (dx2-dx1)/dt
                ∂dx += ∂dx_t * Δt

                c.solution.evals_∂ẋ_∂t += 1
            end

            if outputs
                fmi2GetReal!(c, y_refs, y2)

                ∂y_t = (y2-y1)/dt  
                ∂y += ∂y_t * Δt

                c.solution.evals_∂y_∂t += 1
            end

            fmi2SetTime(c, t; track=false)
        end
    end

    @debug "frule:   ∂y=$(∂y)   ∂dx=$(∂dx)"

    ∂Ω = nothing
    if c.fmu.executionConfig.concat_y_dx
        ∂Ω = vcat(∂y, ∂dx) # [∂y..., ∂dx...]
    else
        ∂Ω = (∂y, ∂dx) 
    end

    return Ω, ∂Ω 
end

function ChainRulesCore.rrule(::typeof(eval!), 
    cRef, 
    dx,
    y,
    y_refs, 
    x,
    u,
    u_refs,
    p,
    p_refs,
    t)

    @assert !isa(cRef, FMU2Component) "Wrong dispatched!"
      
    c = unsafe_pointer_to_objref(Ptr{Nothing}(cRef))
    
    outputs = (length(y_refs) > 0)
    inputs = (length(u_refs) > 0)
    derivatives = (length(dx) > 0)
    states = (length(x) > 0)
    times = (t >= 0.0)
    parameters = (length(p_refs) > 0)

    Ω = eval!(cRef, dx, y, y_refs, x, u, u_refs, p, p_refs, t)

    ##############

    function eval_pullback(r̄)

        ȳ = nothing 
        d̄x = nothing

        if c.fmu.executionConfig.concat_y_dx
            ylen = (isnothing(y_refs) ? 0 : length(y_refs))
            ȳ = r̄[1:ylen]
            d̄x = r̄[ylen+1:end]
        else
            ȳ, d̄x = r̄
        end

        outputs = outputs && !isZeroTangent(ȳ)
        derivatives = derivatives && !isZeroTangent(d̄x)

        if !isa(ȳ, AbstractArray)
            ȳ = collect(ȳ) # [ȳ...]
        end

        if !isa(d̄x, AbstractArray)
            d̄x = collect(d̄x) # [d̄x...]
        end

        # between building and using the pullback maybe the time, state or inputs were changed, so we need to re-set them
       
        if states && c.x != x
            fmi2SetContinuousStates(c, x)
        end

        if inputs ## && c.u != u
            fmi2SetReal(c, u_refs, u)
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

        @debug "rrule pullback ȳ, d̄x = $(ȳ), $(d̄x)"

        dx_refs = c.fmu.modelDescription.derivativeValueReferences
        x_refs = c.fmu.modelDescription.stateValueReferences

        if derivatives 
            if states
                n_dx_x = fmi2VJP!(c, :A, dx_refs, x_refs, d̄x) 
                c.solution.evals_∂ẋ_∂x += 1
            end

            if inputs
                n_dx_u = fmi2VJP!(c, :B, dx_refs, u_refs, d̄x) 
                c.solution.evals_∂ẋ_∂u += 1
            end

            if parameters
                n_dx_p = fmi2VJP!(c, :E, dx_refs, p_refs, d̄x) 
                c.solution.evals_∂ẋ_∂p += 1
            end
        end

        if outputs 
            if states
                n_y_x = fmi2VJP!(c, :C, y_refs, x_refs, ȳ) 
                c.solution.evals_∂y_∂x += 1
            end
        
            if inputs
                n_y_u = fmi2VJP!(c, :D, y_refs, u_refs, ȳ) 
                c.solution.evals_∂y_∂u += 1
            end

            if parameters
                n_y_p = fmi2VJP!(c, :F, y_refs, p_refs, ȳ) 
                c.solution.evals_∂y_∂p += 1
            end
        end

        if c.fmu.executionConfig.eval_t_gradients
            # sample time partials
            # in rrule this should be done even if no new time is actively set
            if (derivatives || outputs) # && times

                # if no time is actively set, use the component current time for sampling
                if !times 
                    t = c.t 
                end 

                dt = 1e-6 # ToDo: better value 

                dx1 = nothing
                dx2 = nothing
                y1 = nothing
                y2 = nothing 

                if derivatives
                    dx1 = zeros(fmi2Real, length(dx_refs))
                    dx2 = zeros(fmi2Real, length(dx_refs))
                    fmi2GetDerivatives!(c, dx1)
                end

                if outputs
                    y1 = zeros(fmi2Real, length(y))
                    y2 = zeros(fmi2Real, length(y))
                    fmi2GetReal!(c, y_refs, y1)
                end

                fmi2SetTime(c, t + dt; track=false)

                if derivatives
                    fmi2GetDerivatives!(c, dx2)

                    ∂dx_t = (dx2-dx1) / dt 
                    n_dx_t = ∂dx_t' * d̄x

                    c.solution.evals_∂ẋ_∂t += 1
                end

                if outputs 
                    fmi2GetReal!(c, y_refs, y2)

                    ∂y_t = (y2-y1) / dt 
                    n_y_t = ∂y_t' * ȳ 

                    c.solution.evals_∂y_∂t += 1
                end

                fmi2SetTime(c, t; track=false)
            end
        end

        # write back
        f̄ = NoTangent()
        c̄Ref = ZeroTangent()
        d̄x = ZeroTangent()
        ȳ = ZeroTangent()
        ȳ_refs = ZeroTangent()
        t̄ = n_y_t + n_dx_t

        x̄ = n_y_x + n_dx_x
        
        ū = n_y_u + n_dx_u
        ū_refs = ZeroTangent()

        p̄ = n_y_p + n_dx_p
        p̄_refs = ZeroTangent()
        
        @debug "rrule:   $((f̄, c̄Ref, d̄x, ȳ, ȳ_refs, x̄, ū, ū_refs, p̄, p̄_refs, t̄))"

        return (f̄, c̄Ref, d̄x, ȳ, ȳ_refs, x̄, ū, ū_refs, p̄, p̄_refs, t̄)
    end

    return (Ω, eval_pullback)
end

# dx, y, x, u, t
@ForwardDiff_frule eval!(cRef::UInt64, 
    dx    ::AbstractVector{<:ForwardDiff.Dual},
    y     ::AbstractVector{<:ForwardDiff.Dual},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:ForwardDiff.Dual},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmi2ValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules eval!(cRef::UInt64, 
    dx    ::AbstractVector{<:ReverseDiff.TrackedReal},
    y     ::AbstractVector{<:ReverseDiff.TrackedReal},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:ReverseDiff.TrackedReal},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    t     ::ReverseDiff.TrackedReal)

# x, p
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    t     ::Real)

# t
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmi2ValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    t     ::ReverseDiff.TrackedReal)

# x
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    t     ::Real)

# u
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:ForwardDiff.Dual},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:ReverseDiff.TrackedReal},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    t     ::Real)

# p
@ForwardDiff_frule eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmi2ValueReference},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmi2ValueReference},
    p     ::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmi2ValueReference},
    t     ::Real)

@grad_from_chainrules eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    t     ::Real)