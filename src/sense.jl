#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMIBase: eval!, invalidate!, check_invalidate!
using FMIBase: getDirectionalDerivative!, getAdjointDerivative!
using FMIBase: setContinuousStates, setInputs, setReal, setTime, setReal, getReal!, getEventIndicators!, getRealType

# in FMI2 and FMI3 we can use fmi2GetDirectionalDerivative for JVP-computations
function jvp!(c::FMUInstance, mtxCache::Symbol, ∂f_refs, ∂x_refs, x, seed)

    jac = getfield(c, mtxCache)
    if isnothing(jac)
        # [Note] type Real, so AD-primitves can be stored for AD over AD 
        # this is necessary for e.g. gradient over implicit solver solutions with autodiff=true
        T = typeof(seed[1])
        jac = FMUJacobian{T}(c, ∂f_refs, ∂x_refs)
        setfield!(c, mtxCache, jac)
    end

    jac.f_refs = ∂f_refs
    jac.x_refs = ∂x_refs

    if c.fmu.executionConfig.JVPBuiltInDerivatives && providesDirectionalDerivatives(c.fmu) && !isa(jac.f_refs, Tuple) && !isa(jac.x_refs, Symbol)
        getDirectionalDerivative!(c, ∂f_refs, ∂x_refs, seed, jac.jvp)
        return jac.jvp
    else
        return jvp!(jac, x, seed)
    end
end

function gvp!(c::FMUInstance, mtxCache::Symbol, ∂f_refs, ∂x_refs, x, seed)

    grad = getfield(c, mtxCache)
    if isnothing(grad)
        # [Note] type Real, so AD-primitves can be stored for AD over AD 
        # this is necessary for e.g. gradient over implicit solver solutions with autodiff=true
        T = typeof(seed[1])
        grad = FMUGradient{T}(c, ∂f_refs, ∂x_refs)
        setfield!(c, mtxCache, grad)
    end

    grad.f_refs = ∂f_refs
    grad.x_refs = ∂x_refs

    if c.fmu.executionConfig.JVPBuiltInDerivatives && providesDirectionalDerivatives(c.fmu) && !isa(grad.f_refs, Tuple) && !isa(grad.x_refs, Symbol)
        getDirectionalDerivative!(c, ∂f_refs, ∂x_refs, [seed], grad.gvp)
        return grad.gvp
    else
        return gvp!(grad, x, seed)
    end
end

# in FMI2 there is no helper for VJP-computations (but in FMI3) ...
function vjp!(c::FMUInstance, mtxCache::Symbol, ∂f_refs, ∂x_refs, x, seed)

    jac = getfield(c, mtxCache)
    if isnothing(jac)
        # [Note] type Real, so AD-primitves can be stored for AD over AD 
        # this is necessary for e.g. gradient over implicit solver solutions with autodiff=true
        T = typeof(seed[1])
        jac = FMUJacobian{T}(c, ∂f_refs, ∂x_refs)
        setfield!(c, mtxCache, jac)
    end

    jac.f_refs = ∂f_refs
    jac.x_refs = ∂x_refs
    
    if c.fmu.executionConfig.VJPBuiltInDerivatives && providesAdjointDerivatives(c.fmu) && !isa(jac.f_refs, Tuple) && !isa(jac.x_refs, Symbol)
        getAdjointDerivative!(c, ∂f_refs, ∂x_refs, seed, jac.vjp)
        return jac.vjp
    else
        return vjp!(jac, x, seed)
    end
end

function vgp!(c::FMUInstance, mtxCache::Symbol, ∂f_refs, ∂x_refs, x, seed)

    grad = getfield(c, mtxCache)
    if isnothing(grad)
        # [Note] type Real, so AD-primitves can be stored for AD over AD 
        # this is necessary for e.g. gradient over implicit solver solutions with autodiff=true
        T = typeof(seed[1])
        grad = FMUGradient{T}(c, ∂f_refs, ∂x_refs)
        setfield!(c, mtxCache, grad)
    end

    grad.f_refs = ∂f_refs
    grad.x_refs = ∂x_refs
    
    if c.fmu.executionConfig.VJPBuiltInDerivatives && providesAdjointDerivatives(c.fmu) && !isa(grad.f_refs, Tuple) && !isa(grad.x_refs, Symbol)
        getAdjointDerivative!(c, ∂f_refs, ∂x_refs, [seed], grad.vgp)
        return grad.vgp
    else
        return vgp!(grad, x, seed)
    end
end

function ChainRulesCore.frule(Δtuple, 
    ::typeof(FMIBase.eval!), 
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

    Δself, ΔcRef, Δdx, Δdx_refs, Δy, Δy_refs, Δx, Δu, Δu_refs, Δp, Δp_refs, Δec, Δec_idcs, Δt = Δtuple # undual ?

    @debug "frule start"

    ### ToDo: Somehow, ForwardDiff enters with all types beeing Float64, this needs to be corrected.

    cRef = unsense(cRef) # undual(cRef)
    if typeof(cRef) != UInt64
        cRef = UInt64(cRef)
    end
    c = unsafe_pointer_to_objref(Ptr{Nothing}(cRef))
    
    # ToDo: is this necessary?
    # t = undual(t)
    # u = undual(u)
    # x = undual(x)
    # p = undual(p)

    dx_refs = unsense(dx_refs)
    dx_refs = convert(Array{UInt32,1}, dx_refs)
    if length(dx_refs) == 0 && length(dx) == length(c.fmu.modelDescription.derivativeValueReferences) # all derivatives, please!
        dx_refs = c.fmu.modelDescription.derivativeValueReferences
    end
    
    # [Note] `unsense` is necessary for AD over AD
    y_refs = unsense(y_refs)
    u_refs = unsense(u_refs)
    p_refs = unsense(p_refs)
    ec_idcs = unsense(ec_idcs)

    y_refs = convert(Array{UInt32,1}, y_refs)
    u_refs = convert(Array{UInt32,1}, u_refs)
    p_refs = convert(Array{UInt32,1}, p_refs)
    ec_idcs = convert(Array{UInt32,1}, ec_idcs) 
    
    ###

    outputs = (length(y_refs) > 0)
    inputs = (length(u_refs) > 0)
    derivatives = (length(dx) > 0)
    states = (length(x) > 0)
    times = (t >= 0.0)
    parameters = (length(p_refs) > 0)
    eventIndicators = (length(ec_idcs) > 0)

    Ω = FMIBase.eval!(cRef, dx, dx_refs, y, y_refs, x, u, u_refs, p, p_refs, ec, ec_idcs, t)
    
    # time, states and inputs where already set in `eval!`, no need to repeat it here

    # if length(c.frule_output.y) != length(y)
    #     c.frule_output.y = zeros(length(y))
    # else
    #     c.frule_output.y .= 0.0
    # end

    # if length(c.frule_output.dx) != length(dx)
    #     c.frule_output.dx = zeros(length(dx))
    # else
    #     c.frule_output.dx .= 0.0
    # end

    # if length(c.frule_output.ec) != length(ec)
    #     c.frule_output.ec = zeros(length(ec))
    # else
    #     c.frule_output.ec .= 0.0
    # end

    # ∂y = c.frule_output.y 
    # ∂dx = c.frule_output.dx 
    # ∂e = c.frule_output.ec 

    ∂y = ZeroTangent()
    ∂dx = ZeroTangent()
    ∂e = ZeroTangent()

    if Δx != NoTangent() && length(Δx) > 0

        if states
            if derivatives
                ∂dx += jvp!(c, :∂ẋ_∂x, dx_refs, c.fmu.modelDescription.stateValueReferences, x, Δx)
                c.solution.evals_∂ẋ_∂x += 1
            end

            if outputs 
                ∂y += jvp!(c, :∂y_∂x, y_refs, c.fmu.modelDescription.stateValueReferences, x, Δx)
                c.solution.evals_∂y_∂x += 1
            end

            if eventIndicators
                ∂e += jvp!(c, :∂e_∂x, (:indicators, ec_idcs), c.fmu.modelDescription.stateValueReferences, x, Δx)
                c.solution.evals_∂e_∂x += 1
            end
        end
    end

    
    if Δu != NoTangent() && length(Δu) > 0

        if inputs
            if derivatives
                ∂dx += jvp!(c, :∂ẋ_∂u, dx_refs, u_refs, u, Δu)
                c.solution.evals_∂ẋ_∂u += 1
            end

            if outputs
                ∂y += jvp!(c, :∂y_∂u, y_refs, u_refs, u, Δu)
                c.solution.evals_∂y_∂u += 1
            end

            if eventIndicators
                ∂e += jvp!(c, :∂e_∂u, (:indicators, ec_idcs), u_refs, u, Δu)
                c.solution.evals_∂e_∂u += 1
            end
        end
    end

    if Δp != NoTangent() && length(Δp) > 0

        if parameters
            if derivatives
                ∂dx += jvp!(c, :∂ẋ_∂p, dx_refs, p_refs, p, Δp)
                c.solution.evals_∂ẋ_∂p += 1
            end

            if outputs
                ∂y += jvp!(c, :∂y_∂p, y_refs, p_refs, p, Δp)
                c.solution.evals_∂y_∂p += 1
            end

            if eventIndicators
                ∂e += jvp!(c, :∂e_∂p, (:indicators, ec_idcs), p_refs, p, Δp)
                c.solution.evals_∂e_∂p += 1
            end
        end
    end

    if Δt != NoTangent() && c.fmu.executionConfig.eval_t_gradients

        if times 
            if derivatives
                ∂dx += gvp!(c, :∂ẋ_∂t, dx_refs, :time, t, Δt)
                c.solution.evals_∂ẋ_∂t += 1
            end

            if outputs
                ∂y += gvp!(c, :∂y_∂t, y_refs, :time, t, Δt)
                c.solution.evals_∂y_∂t += 1
            end

            if eventIndicators
                ∂e += gvp!(c, :∂e_∂t, (:indicators, ec_idcs), :time, t, Δt)
                c.solution.evals_∂e_∂t += 1
            end
        end
    end

    @debug "frule end:   ∂y=$(∂y)   ∂dx=$(∂dx)   ∂e=$(∂e)"

    # ∂Ω = nothing
    # if c.fmu.executionConfig.concat_eval
    #     ∂Ω = vcat(∂y, ∂dx, ∂e) # [∂y..., ∂dx..., ∂e...]
    # else
    #     ∂Ω = (∂y, ∂dx, ∂e) 
    # end
    # ∂Ω = vcat(∂y, ∂dx, ∂e)

    # [Note] Type Real is required for AD over AD
    ∂Ω = FMUEvaluationOutput{Real}() # Float64
    ∂Ω.y  = ∂y
    ∂Ω.dx = ∂dx
    ∂Ω.ec = ∂e

    # c.frule_output.y  = ∂y
    # c.frule_output.dx = ∂dx
    # c.frule_output.ec = ∂e
    # ∂Ω = c.frule_output

    return Ω, ∂Ω 
end

function ChainRulesCore.rrule(::typeof(FMIBase.eval!), 
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

    @assert !isa(cRef, FMUInstance) "Wrong dispatched!"

    @debug "rrule start: $((cRef, dx, dx_refs, y, y_refs, x, u, u_refs, p, p_refs, ec, ec_idcs, t))"
      
    c = unsafe_pointer_to_objref(Ptr{Nothing}(cRef))
    
    outputs = (length(y_refs) > 0)
    inputs = (length(u_refs) > 0)
    derivatives = (length(dx) > 0)
    states = (length(x) > 0)
    times = (t >= 0.0)
    parameters = (length(p_refs) > 0)
    eventIndicators = (length(ec) > 0)

    # [ToDo] remove!
    # x = unsense(x)

    # [Note] it is mandatory to set the (unknown) discrete state of the FMU by 
    #        setting the corresponding snapshot (that holds all related quantitites, including the discrete state)
    #        from the snapshot cache. This needs to be done for Ω, as well as for the pullback seperately,
    #        because they are evaluated at different points in time during ODE solving.
    if length(c.solution.snapshots) > 0 
        sn = getSnapshot(c.solution, t)
        apply!(c, sn)
    end

    Ω = FMIBase.eval!(cRef, dx, dx_refs, y, y_refs, x, u, u_refs, p, p_refs, ec, ec_idcs, t)

    # [ToDo] remove this copy
    Ω = copy(Ω)

    # if t < 1.0
    #     @assert dx[2] <= 0.0 "$(dx[2]) for t=$(t)"
    # end 
    # if t > 1.0 && t < 2.0
    #     @assert dx[2] >= 0.0 "$(dx[2]) for t=$(t)"
    # end

    ##############

    # [ToDo] maybe the arrays change between pullback creation and use! check this!
    x = copy(x)
    p = copy(p)
    u = copy(u)
    dx = copy(dx)
    y = copy(y)
    ec = copy(ec)

    function eval_pullback(r̄)

        #println("$(t),")

              # ȳ = nothing 
        # d̄x = nothing
        # ēc = nothing

        # if c.fmu.executionConfig.concat_eval
        #     ylen = (isnothing(y_refs) ? 0 : length(y_refs))
        #     dxlen = length(dx)
        #     ȳ = r̄[1:ylen]
        #     d̄x = r̄[ylen+1:ylen+dxlen]
        #     ēc = r̄[ylen+dxlen+1:end]
        # else
        #     ȳ, d̄x, ēc = r̄
        # end

        # [ToDo] This is not a good workaround for ReverseDiff!
        # for i in 1:length(r̄)
        #     if abs(r̄[i]) > 1e64 
        #         r̄[i] = 0.0 
        #     end
        # end

        ylen = (isnothing(y_refs) ? 0 : length(y_refs))
        dxlen = (isnothing(dx) ? 0 : length(dx))
        
        d̄x = r̄[1:dxlen] # @view(r̄[ylen+1:ylen+dxlen])
        ȳ  = r̄[dxlen+1:dxlen+ylen] # @view(r̄[1:ylen])
        ēc = r̄[ylen+dxlen+1:end] # @view(r̄[ylen+dxlen+1:end])

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

        # [NOTE] for construction of the gradient/jacobian over an ODE solution, many different pullbacks are requested 
        #        and chained together. At the time of creation of the pullback, it is not known which jacobians are needed.
        #        Therefore for correct sensitivities, the FMU state must be captured during simulation and 
        #        set during pullback evaluation. (discrete FMU state might change during simulation)
        if length(c.solution.snapshots) > 0 # c.t != t 
            sn = getSnapshot(c.solution, t)
            apply!(c, sn)
        end

        # [ToDo] Not everything is still needed (from the setters)
        #        These lines could be replaced by an `eval!` call?
        if states && !c.fmu.isZeroState # && c.x != x 
            setContinuousStates(c, x)
        end

        if inputs ## && c.u != u
            setInputs(c, u_refs, u)
        end

        if parameters && c.fmu.executionConfig.set_p_every_step
            setReal(c, p_refs, p)
        end

        if times # && c.t != t
            setTime(c, t)
        end

        x̄ = zeros(length(x)) #ZeroTangent()
        t̄ = 0.0 #ZeroTangent()
        ū = zeros(length(u)) #ZeroTangent()
        p̄ = zeros(length(p)) #eroTangent()

        if length(dx) > 0 && length(dx_refs) == 0 # all derivatives, please!
            dx_refs = c.fmu.modelDescription.derivativeValueReferences
        end
        x_refs = c.fmu.modelDescription.stateValueReferences

        if derivatives 
            if states
                # [ToDo] everywhere here, `+=` allocates, better `.+=` ?
                x̄ += vjp!(c, :∂ẋ_∂x, dx_refs, x_refs, x, d̄x) 
                c.solution.evals_∂ẋ_∂x += 1
            end

            if inputs
                ū += vjp!(c, :∂ẋ_∂u, dx_refs, u_refs, u, d̄x) 
                c.solution.evals_∂ẋ_∂u += 1
            end

            if parameters
                p̄ += vjp!(c, :∂ẋ_∂p, dx_refs, p_refs, p, d̄x) 
                c.solution.evals_∂ẋ_∂p += 1
            end

            if times && c.fmu.executionConfig.eval_t_gradients
                t̄ += vgp!(c, :∂ẋ_∂t, dx_refs, :time, t, d̄x) 
                c.solution.evals_∂ẋ_∂t += 1
            end
        end

        if outputs 
            if states
                x̄ += vjp!(c, :∂y_∂x, y_refs, x_refs, x, ȳ) 
                c.solution.evals_∂y_∂x += 1
            end
        
            if inputs
                ū += vjp!(c, :∂y_∂u, y_refs, u_refs, u, ȳ) 
                c.solution.evals_∂y_∂u += 1
            end

            if parameters
                p̄ += vjp!(c, :∂y_∂p, y_refs, p_refs, p, ȳ) 
                c.solution.evals_∂y_∂p += 1
            end

            if times && c.fmu.executionConfig.eval_t_gradients
                t̄ += vgp!(c, :∂y_∂t, y_refs, :time, t, ȳ)
                c.solution.evals_∂y_∂t += 1
            end
        end

        if eventIndicators
            if states
                x̄ += vjp!(c, :∂e_∂x, (:indicators, ec_idcs), x_refs, x, ēc) 
                c.solution.evals_∂e_∂x += 1
            end

            if inputs
                ū += vjp!(c, :∂e_∂u, (:indicators, ec_idcs), u_refs, u, ēc) 
                c.solution.evals_∂e_∂u += 1
            end

            if parameters
                p̄ += vjp!(c, :∂e_∂p, (:indicators, ec_idcs), p_refs, p, ēc) 
                c.solution.evals_∂e_∂p += 1
            end

            if times && c.fmu.executionConfig.eval_t_gradients
                t̄ += vgp!(c, :∂e_∂t, (:indicators, ec_idcs), :time, t, ēc) 
                c.solution.evals_∂e_∂t += 1
            end
        end

        # write back
        f̄ = [] # NoTangent()
        c̄Ref = [] # ZeroTangent()
        d̄x_refs = [] # ZeroTangent()
        ȳ_refs = [] # ZeroTangent()
        ēc_idcs = [] # ZeroTangent()
        ū_refs = [] # ZeroTangent()
        p̄_refs = [] # ZeroTangent()

        d̄x = zeros(length(dx)) # ZeroTangent()
        ȳ = zeros(length(y)) # ZeroTangent()
        ēc = zeros(length(ec)) # ZeroTangent() # copy(ec) # 

        @debug "pullback on d̄x, ȳ, ēc = $(d̄x), $(ȳ), $(ēc)\nt= $(t)s\nx=$(x)\ndx=$(dx)\n$((x̄, ū, p̄, t̄))"
        
        # [ToDo] This needs to be a tuple... but this prevents pre-allocation...
        return (f̄, c̄Ref, d̄x, d̄x_refs, ȳ, ȳ_refs, x̄, ū, ū_refs, p̄, p̄_refs, ēc, ēc_idcs, t̄)
    end

    @debug "rrule end: $((Ω, eval_pullback))"

    return (Ω, eval_pullback)
end

# dx, y, x, u, p, ec, t
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64, 
    dx    ::AbstractVector{<:ForwardDiff.Dual},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:ForwardDiff.Dual},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:ForwardDiff.Dual},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64, 
    dx    ::AbstractVector{<:ReverseDiff.TrackedReal},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:ReverseDiff.TrackedReal},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:ReverseDiff.TrackedReal},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ReverseDiff.TrackedReal)

# dx, y, x, u, t
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64, 
    dx    ::AbstractVector{<:ForwardDiff.Dual},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:ForwardDiff.Dual},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:ForwardDiff.Dual},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64, 
    dx    ::AbstractVector{<:ReverseDiff.TrackedReal},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:ReverseDiff.TrackedReal},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:ReverseDiff.TrackedReal},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ReverseDiff.TrackedReal)

# x, u
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64, 
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:ForwardDiff.Dual},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64, 
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:ReverseDiff.TrackedReal},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

# x, p
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

# t
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ReverseDiff.TrackedReal)

# x
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

# u
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:ForwardDiff.Dual},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:ReverseDiff.TrackedReal},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real}, 
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

# p
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

# ec
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

# x, t
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ReverseDiff.TrackedReal)

# x, ec, t
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ReverseDiff.TrackedReal)

# ec, t
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:Real}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ReverseDiff.TrackedReal)

# x, ec
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

# x, p, t
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ReverseDiff.TrackedReal)

# x, p, ec, t
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ForwardDiff.Dual)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::ReverseDiff.TrackedReal)

# x, p, ec
@ForwardDiff_frule FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x     ::AbstractVector{<:ForwardDiff.Dual}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p     ::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmiValueReference},
    ec    ::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

@grad_from_chainrules FMIBase.eval!(cRef::UInt64,  
    dx    ::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y     ::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x     ::AbstractVector{<:ReverseDiff.TrackedReal}, 
    u     ::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p     ::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec    ::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t     ::Real)

# FiniteDiff Jacobians

abstract type FMUSensitivities end

mutable struct FMUJacobian{C, T, F} <: FMUSensitivities
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

    validations::Int
    colorings::Int

    function FMUJacobian{T}(component::C, f_refs::Union{Vector{UInt32}, Tuple{Symbol, Vector{UInt32}}}, x_refs::Union{Vector{UInt32}, Symbol}) where {C, T}

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
        inst.validations = 0
        inst.colored = false
        inst.colorings = 0
        
        return inst
    end

end 

mutable struct FMUGradient{C, T, F} <: FMUSensitivities
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

    validations::Int
    colorings::Int

    function FMUGradient{T}(component::C, f_refs::Union{Vector{UInt32}, Tuple{Symbol, Vector{UInt32}}}, x_refs::Union{UInt32, Symbol}) where {C, T}

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
        inst.validations = 0
        inst.colored = false
        inst.colorings = 0
        
        return inst
    end

end 

function f_∂v_∂v(jac::FMUJacobian, dx, x)
    setReal(jac.component, jac.x_refs, x; track=false)
    getReal!(jac.component, jac.f_refs, dx)
    return dx
end

function f_∂e_∂v(jac::FMUJacobian, dx, x)
    setReal(jac.component, jac.x_refs, x; track=false)
    getEventIndicators!(jac.component, dx, jac.f_refs[2])
    return dx
end

function f_∂e_∂t(jac::FMUGradient, dx, x)
    setTime(jac.component, x; track=false)
    getEventIndicators!(jac.component, dx, jac.f_refs[2])
    return dx
end

function f_∂v_∂t(jac::FMUGradient, dx, x)
    setTime(jac.component, x; track=false)
    getReal!(jac.component, jac.f_refs, dx)
    return dx
end

function FMIBase.invalidate!(sens::FMUSensitivities)
    sens.valid = false 
    return nothing 
end

function FMIBase.check_invalidate!(vrs, sens::FMUSensitivities)
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

function uncolor!(jac::FMUSensitivities)
    jac.colored = false 
    return nothing 
end

function onehot(c::FMUInstance, len::Integer, i::Integer) # [ToDo] this could be solved without allocations
    ret = zeros(getRealType(c), len)
    ret[i] = 1.0
    return ret 
end

function validate!(jac::FMUJacobian, x::AbstractVector)

    rows = length(jac.f_refs)
    cols = length(jac.x_refs)

    if jac.component.fmu.executionConfig.sensitivity_strategy == :FMIDirectionalDerivative && providesDirectionalDerivatives(jac.component.fmu) && !isa(jac.f_refs, Tuple) && !isa(jac.x_refs, Symbol)
        # ToDo: use directional derivatives with sparsitiy information!
        # ToDo: Optimize allocation (onehot)
        # [Note] Jacobian is sampled column by column
        for i in 1:cols
            getDirectionalDerivative!(jac.component, jac.f_refs, jac.x_refs, onehot(jac.component, cols, i), view(jac.mtx, 1:rows, i))
        end
    elseif jac.component.fmu.executionConfig.sensitivity_strategy == :FMIAdjointDerivative && providesAdjointDerivatives(jac.component.fmu) && !isa(jac.f_refs, Tuple) && !isa(jac.x_refs, Symbol)
        # ToDo: use directional derivatives with sparsitiy information!
        # ToDo: Optimize allocation (onehot)
        # [Note] Jacobian is sampled row by row
        for i in 1:rows
            getAdjointDerivative!(jac.component, jac.f_refs, jac.x_refs, onehot(jac.component, rows, i), view(jac.mtx, 1:cols, i))
        end
    else #if jac.component.fmu.executionConfig.sensitivity_strategy == :FiniteDiff
        # cache = FiniteDiff.JacobianCache(x)
        FiniteDiff.finite_difference_jacobian!(jac.mtx, (_x, _dx) -> (jac.f(jac, _x, _dx)), x) # , cache)
    # else
    #     @assert false "Unknown sensitivity strategy `$(jac.component.fmu.executionConfig.sensitivity_strategy)`."
    end

    jac.validations += 1
    jac.valid = true 
    return nothing
end

function validate!(grad::FMUGradient, x::Real)

    if grad.component.fmu.executionConfig.sensitivity_strategy == :FMIDirectionalDerivative && providesDirectionalDerivatives(grad.component.fmu) && !isa(grad.f_refs, Tuple) && !isa(grad.x_refs, Symbol)
        # ToDo: use directional derivatives with sparsitiy information!
        getDirectionalDerivative!(grad.component, grad.f_refs, grad.x_refs, ones(length(jac.f_refs)), grad.vec)
    else #if grad.component.fmu.executionConfig.sensitivity_strategy == :FiniteDiff
        # cache = FiniteDiff.GradientCache(x)
        FiniteDiff.finite_difference_gradient!(grad.vec, (_x, _dx) -> (grad.f(grad, _x, _dx)), x) # , cache)
    end

    grad.validations += 1
    grad.valid = true 
    return nothing
end
    
function color!(sens::FMUSensitivities)
    # ToDo
    # colors = SparseDiffTools.matrix_colors(sparsejac)

    sens.colorings += 1
    sens.colored = true 
    return nothing
end

function ref_length(ref::AbstractArray)
    return length(ref)
end

function ref_length(ref::Symbol)
    if ref == :time 
        return 1 
    else
        @assert false "unknwon ref symbol: $(ref)"
    end
end

function ref_length(ref::Tuple)
    @assert length(ref) == 2 "tuple ref length is $(length(ref)) != 2"
    if ref[1] == :indicators 
        return length(ref[2])
    else
        @assert false "unknwon tuple ref $(ref)"
    end
end

function update!(jac::FMUJacobian, x)

    if size(jac.mtx) != (ref_length(jac.f_refs), ref_length(jac.x_refs))
        jac.mtx = similar(jac.mtx, ref_length(jac.f_refs), ref_length(jac.x_refs))
        jac.jvp = similar(jac.jvp, ref_length(jac.f_refs))
        jac.vjp = similar(jac.vjp, ref_length(jac.x_refs))
        
        jac.valid = false
    end

    if !jac.valid
        validate!(jac, x)
    end

    if !jac.colored
        color!(jac)
    end
    return nothing
end

function update!(gra::FMUGradient, x)

    if length(gra.vec) != ref_length(gra.f_refs)
        gra.vec = similar(gra.vec, ref_length(jac.f_refs))
        gra.gvp = similar(gra.gvp, ref_length(jac.f_refs))
        gra.vgp = similar(gra.vgp, ref_length(jac.x_refs))
        
        gra.valid = false
    end

    if !gra.valid
        validate!(gra, x)
    end

    if !gra.colored
        color!(gra)
    end
    return nothing
end

function jvp!(jac::FMUJacobian, x::AbstractVector, v::AbstractVector)
    FMISensitivity.update!(jac, x)
    #return jac.mtx * v
    return mul!(jac.jvp, jac.mtx, v)
end

function vjp!(jac::FMUJacobian, x::AbstractVector, v::AbstractVector)
    FMISensitivity.update!(jac, x)
    #return jac.mtx' * v 
    return mul!(jac.vjp, jac.mtx', v)
end

function gvp!(grad::FMUGradient, x, v)
    FMISensitivity.update!(grad, x)
    #return grad.vec * v 
    return mul!(grad.gvp, grad.vec, v)
end

function vgp!(grad::FMUGradient, x, v)
    FMISensitivity.update!(grad, x)
    mul!(grad.vgp, grad.vec', v) 
    return grad.vgp[1]
end
