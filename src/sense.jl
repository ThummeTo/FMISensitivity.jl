#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import FMIBase: eval!, invalidate!, check_invalidate!
using FMIBase:
    getDirectionalDerivative!, getAdjointDerivative!, sampleDirectionalDerivative!
using FMIBase:
    setContinuousStates,
    setInputs,
    setReal,
    setTime,
    setReal,
    getReal!,
    getEventIndicators!,
    getRealType,
    startSampling,
    stopSampling,
    issense

# in FMI2 and FMI3 we can use fmi2GetDirectionalDerivative for JVP-computations
function jvp!(c::FMUInstance, mtxCache::Symbol, ∂f_refs, ∂x_refs, x, seed; accu = nothing)

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

    if c.fmu.executionConfig.JVPBuiltInDerivatives &&
       providesDirectionalDerivatives(c.fmu) &&
       !isa(jac.f_refs, Tuple) &&
       !isa(jac.x_refs, Symbol)
        getDirectionalDerivative!(c, ∂f_refs, ∂x_refs, seed, jac.jvp)
    else
        jvp!(jac, x, seed)
    end

    accu .+= jac.jvp

    return nothing
end

function gvp!(c::FMUInstance, mtxCache::Symbol, ∂f_refs, ∂x_refs, x, seed; accu = nothing)

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

    if c.fmu.executionConfig.JVPBuiltInDerivatives &&
       providesDirectionalDerivatives(c.fmu) &&
       !isa(grad.f_refs, Tuple) &&
       !isa(grad.x_refs, Symbol)
        getDirectionalDerivative!(c, ∂f_refs, ∂x_refs, [seed], grad.gvp)
    else
        gvp!(grad, x, seed)
    end

    accu .+= grad.gvp

    return nothing
end

# in FMI2 there is no helper for VJP-computations (but in FMI3) ...
function vjp!(c::FMUInstance, mtxCache::Symbol, ∂f_refs, ∂x_refs, x, seed; accu = nothing)

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

    if c.fmu.executionConfig.VJPBuiltInDerivatives &&
       providesAdjointDerivatives(c.fmu) &&
       !isa(jac.f_refs, Tuple) &&
       !isa(jac.x_refs, Symbol)
        getAdjointDerivative!(c, ∂f_refs, ∂x_refs, seed, jac.vjp)
    else
        vjp!(jac, x, seed)
    end

    accu .+= jac.vjp

    return nothing
end

function vgp!(c::FMUInstance, mtxCache::Symbol, ∂f_refs, ∂x_refs, x, seed; accu = nothing)

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

    if c.fmu.executionConfig.VJPBuiltInDerivatives &&
       providesAdjointDerivatives(c.fmu) &&
       !isa(grad.f_refs, Tuple) &&
       !isa(grad.x_refs, Symbol)
        getAdjointDerivative!(c, ∂f_refs, ∂x_refs, [seed], grad.vgp)
    else
        vgp!(grad, x, seed)
    end

    accu .+= grad.vgp

    return nothing
end

function ChainRulesCore.frule(
    Δtuple,
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
    t,
    x_d,
)

    Δself,
    ΔcRef,
    Δdx,
    Δdx_refs,
    Δy,
    Δy_refs,
    Δx,
    Δu,
    Δu_refs,
    Δp,
    Δp_refs,
    Δec,
    Δec_idcs,
    Δt,
    Δx_d = Δtuple # undual ?

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
    if length(dx_refs) == 0 &&
       length(dx) == length(c.fmu.modelDescription.derivativeValueReferences) # all derivatives, please!
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
    times = FMIBase.isSetReal(c.fmu, t)
    states = (length(x) > 0)
    parameters = (length(p_refs) > 0)
    eventIndicators = (length(ec_idcs) > 0)

    Ω = FMIBase.eval!(
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
        t,
        x_d,
    )

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

    ∂y = zeros(length(y))
    ∂dx = zeros(length(dx))
    ∂e = zeros(length(ec))

    if Δx != NoTangent() && length(Δx) > 0

        if states
            if derivatives
                jvp!(
                    c,
                    :∂ẋ_∂x,
                    dx_refs,
                    c.fmu.modelDescription.stateValueReferences,
                    x,
                    Δx;
                    accu = ∂dx,
                )
                c.solution.evals_∂ẋ_∂x += 1
            end

            if outputs
                jvp!(
                    c,
                    :∂y_∂x,
                    y_refs,
                    c.fmu.modelDescription.stateValueReferences,
                    x,
                    Δx;
                    accu = ∂y,
                )
                c.solution.evals_∂y_∂x += 1
            end

            if eventIndicators
                jvp!(
                    c,
                    :∂e_∂x,
                    (:indicators, ec_idcs),
                    c.fmu.modelDescription.stateValueReferences,
                    x,
                    Δx;
                    accu = ∂e,
                )
                c.solution.evals_∂e_∂x += 1
            end
        end
    end

    if Δu != NoTangent() && length(Δu) > 0

        if inputs
            if derivatives
                jvp!(c, :∂ẋ_∂u, dx_refs, u_refs, u, Δu; accu = ∂dx)
                c.solution.evals_∂ẋ_∂u += 1
            end

            if outputs
                jvp!(c, :∂y_∂u, y_refs, u_refs, u, Δu; accu = ∂y)
                c.solution.evals_∂y_∂u += 1
            end

            if eventIndicators
                jvp!(c, :∂e_∂u, (:indicators, ec_idcs), u_refs, u, Δu; accu = ∂e)
                c.solution.evals_∂e_∂u += 1
            end
        end
    end

    if Δp != NoTangent() && length(Δp) > 0

        if parameters
            if derivatives
                jvp!(c, :∂ẋ_∂p, dx_refs, p_refs, p, Δp; accu = ∂dx)
                c.solution.evals_∂ẋ_∂p += 1
            end

            if outputs
                jvp!(c, :∂y_∂p, y_refs, p_refs, p, Δp; accu = ∂y)
                c.solution.evals_∂y_∂p += 1
            end

            if eventIndicators
                jvp!(c, :∂e_∂p, (:indicators, ec_idcs), p_refs, p, Δp; accu = ∂e)
                c.solution.evals_∂e_∂p += 1
            end
        end
    end

    if Δt != NoTangent() && c.fmu.executionConfig.eval_t_gradients

        if times
            if derivatives
                gvp!(c, :∂ẋ_∂t, dx_refs, :time, t, Δt; accu = ∂dx)
                c.solution.evals_∂ẋ_∂t += 1
            end

            if outputs
                gvp!(c, :∂y_∂t, y_refs, :time, t, Δt; accu = ∂y)
                c.solution.evals_∂y_∂t += 1
            end

            if eventIndicators
                gvp!(c, :∂e_∂t, (:indicators, ec_idcs), :time, t, Δt; accu = ∂e)
                c.solution.evals_∂e_∂t += 1
            end
        end
    end

    @debug "frule end:   ∂y=$(∂y)   ∂dx=$(∂dx)   ∂e=$(∂e)"

    # [Note] Type Real is required for AD over AD
    ∂Ω = FMUEvaluationOutput{Real}() # Float64
    ∂Ω.dx = ∂dx
    ∂Ω.y = ∂y
    ∂Ω.ec = ∂e

    return Ω, ∂Ω
end

function ChainRulesCore.rrule(
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
    t,
    x_d,
)

    @assert !isa(cRef, FMUInstance) "Wrong dispatched!"

    @debug "rrule start: $((cRef, dx, dx_refs, y, y_refs, x, u, u_refs, p, p_refs, ec, ec_idcs, t, x_d))"

    c = unsafe_pointer_to_objref(Ptr{Nothing}(cRef))

    y_len = (isnothing(y_refs) ? 0 : length(y_refs))
    dx_len = (isnothing(dx) ? 0 : length(dx))

    _outputs = (length(y_refs) > 0)
    _derivatives = (length(dx) > 0)
    _eventIndicators = (length(ec) > 0)
    states = (length(x) > 0)
    inputs = (length(u_refs) > 0)
    times = FMIBase.isSetReal(c.fmu, t)
    parameters = (length(p_refs) > 0)

    @assert !issense(x_d) "discrete state sensitive!"

    # x_d = []
    # # because of single discrete state
    # if c.fmu.isDummyDiscrete
    #     x_d = unsense(x[end:end])
    # end

    # [ToDo] remove!
    # x = unsense(x)

    # two strategies for `snapshotEveryStep`: 
    # (false) use the closest snapshot, change values to the current state etc. -> might be difficult with nasty algebraic loops!
    # (true) make snapshots for every time step (more secure, more memory)
    pullback_snapshot = nothing
    Ω = nothing

    if c.fmu.executionConfig.snapshot_every_step
        # capture state
        # startSampling(c)
        # tmp_snapshot = snapshot!(c)

        Ω = FMIBase.eval!(
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
            t,
            x_d,
        )

        # [Todo] this is wrong, discrete state may not match, bc rrule could be called after event handling for 
        # before a state before the event!
        pullback_snapshot = snapshot!(c)

        # re-set original state to persue simulation
        # stopSampling(c)
        # apply!(c, tmp_snapshot)
        # freeSnapshot!(tmp_snapshot)
    else
        # [Note] it is mandatory to set the (unknown) discrete state of the FMU by 
        #        setting the corresponding snapshot (that holds all related quantities, including the discrete state)
        #        from the snapshot cache. This needs to be done for Ω, as well as for the pullback separately,
        #        because they are (or might be) evaluated at different points in time during ODE solving.
        # if length(c.solution.snapshots) > 0
        #     pullback_snapshot = getSnapshotOrPrevious(c.solution, t)

        #     # for discontinuous systems, this happens for ReverseDiff - whyever
        #     if isnan(t)
        #         pullback_snapshot = c.solution.snapshots[end]
        #         @warn "rrule is called for t=NaN, fallback to last snapshot at t=$(pullback_snapshot.t)."
        #     end

        #     @assert !isnothing(pullback_snapshot) "rrule failed to find snapshot for t=$(t), only available snapshots are:\n$(collect(s.t for s in c.solution.snapshots))."

        #     apply!(c, pullback_snapshot)
        # else
        #     # if no snapshots available, nothing to set here :-)
        # end

        Ω = FMIBase.eval!(
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
            t,
            x_d,
        )

        #pullback_snapshot = getSnapshot(c.solution, t)

        # [ToDo] maybe the arrays change between pullback creation and use! check this!
        t = copy(t) # is scalar, but could be AD-primitive.
        x = copy(x)
        x_d = copy(x_d)
        p = copy(p)
        u = copy(u)
        # dx = copy(dx)
        # y = copy(y)
        # ec = copy(ec)
    end

    # [ToDo] remove this copy
    # Ω = copy(Ω)

    # if t < 1.0
    #     @assert dx[2] <= 0.0 "$(dx[2]) for t=$(t)"
    # end 
    # if t > 1.0 && t < 2.0
    #     @assert dx[2] >= 0.0 "$(dx[2]) for t=$(t)"
    # end

    ##############

    if dx_len > 0 && length(dx_refs) == 0 # all derivatives, please!
        dx_refs = c.fmu.modelDescription.derivativeValueReferences
    end
    x_refs = c.fmu.modelDescription.stateValueReferences

    function eval_pullback(r̄)

        @debug "eval pullback start"

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

        d̄x = @view(r̄[1:dx_len])  # r̄[1:dx_len] 
        ȳ = @view(r̄[(dx_len+1):(dx_len+y_len)]) # r̄[dx_len+1:dx_len+y_len] 
        ēc = @view(r̄[(dx_len+y_len+1):end]) # r̄[y_len+dx_len+1:end] 

        outputs = _outputs && !isZeroTangent(ȳ)
        derivatives = _derivatives && !isZeroTangent(d̄x)
        eventIndicators = _eventIndicators && !isZeroTangent(ēc)

        if !isa(ȳ, AbstractArray)
            ȳ = collect(ȳ) # [ȳ...]
        end

        if !isa(d̄x, AbstractArray)
            d̄x = collect(d̄x) # [d̄x...]
        end

        if !isa(ēc, AbstractArray)
            ēc = collect(ēc) # [ēc...]
        end

        # ToDo: Is sampling actually required here?
        #startSampling(c)   
        # if !isnothing(pullback_snapshot)
        #     apply!(c, pullback_snapshot)
        # end

        # here, we need to set the state/time/etc. to fit the instance the pullback was created!
        if c.fmu.executionConfig.snapshot_every_step
            # [NOTE] for construction of the gradient/jacobian over an ODE solution, many different pullbacks are requested 
            #        and chained together. At the time of creation of the pullback, it is not known which jacobians are needed.
            #        Therefore for correct sensitivities, the FMU state must be captured during simulation and 
            #        set during pullback evaluation. (discrete FMU state might change during simulation)

            #startSampling(c)   
            apply!(c, pullback_snapshot)
        end

        # light weight call to eval!
        FMIBase.eval_set!(c, x, u, u_refs, p, p_refs, t, x_d)

        x̄ = zeros(length(x)) #ZeroTangent()
        t̄ = zeros(1) #ZeroTangent()
        ū = zeros(length(u)) #ZeroTangent()
        p̄ = zeros(length(p)) #ZeroTangent()
        x̄_d = zeros(length(x_d)) # ZeroTangent()

        if derivatives
            if states
                vjp!(c, :∂ẋ_∂x, dx_refs, x_refs, x, d̄x; accu = x̄)
                c.solution.evals_∂ẋ_∂x += 1
                #@info "t=$(t)\n\tx=$(x)\n\tx̄=$(x̄)\n\tmtx=$(c.∂ẋ_∂x.mtx)\n\tvjp=$(c.∂ẋ_∂x.vjp)\n\td̄x=$(d̄x)"
            end

            if inputs
                vjp!(c, :∂ẋ_∂u, dx_refs, u_refs, u, d̄x; accu = ū)
                c.solution.evals_∂ẋ_∂u += 1
            end

            if parameters
                vjp!(c, :∂ẋ_∂p, dx_refs, p_refs, p, d̄x; accu = p̄)
                c.solution.evals_∂ẋ_∂p += 1
            end

            if times && c.fmu.executionConfig.eval_t_gradients
                vgp!(c, :∂ẋ_∂t, dx_refs, :time, t, d̄x; accu = t̄)
                c.solution.evals_∂ẋ_∂t += 1
            end
        end

        if outputs
            if states
                vjp!(c, :∂y_∂x, y_refs, x_refs, x, ȳ; accu = x̄)
                c.solution.evals_∂y_∂x += 1
            end

            if inputs
                vjp!(c, :∂y_∂u, y_refs, u_refs, u, ȳ; accu = ū)
                c.solution.evals_∂y_∂u += 1
            end

            if parameters
                vjp!(c, :∂y_∂p, y_refs, p_refs, p, ȳ; accu = p̄)
                c.solution.evals_∂y_∂p += 1
            end

            if times && c.fmu.executionConfig.eval_t_gradients
                vgp!(c, :∂y_∂t, y_refs, :time, t, ȳ; accu = t̄)
                c.solution.evals_∂y_∂t += 1
            end
        end

        # if sum(abs.(ēc)) > 0
        #     @info "t=$(t)\nēc=$(ēc)"
        # end

        # if sum(abs.(d̄x)) > 0
        #     @info "t=$(t)\nd̄x=$(d̄x)"
        # end

        #@info "$(_eventIndicators) | $(states)"

        if _eventIndicators # ToDo: This should be `eventIndicators` but we get it for every ēc bc. of workaround in `condition!`
            if states
                vjp!(c, :∂e_∂x, (:indicators, ec_idcs), x_refs, x, ēc; accu = x̄)
                c.solution.evals_∂e_∂x += 1

                #@info "t=$(t)\nēc=$(ēc)"
            end

            if inputs
                vjp!(c, :∂e_∂u, (:indicators, ec_idcs), u_refs, u, ēc; accu = ū)
                c.solution.evals_∂e_∂u += 1
            end

            if parameters
                vjp!(c, :∂e_∂p, (:indicators, ec_idcs), p_refs, p, ēc; accu = p̄)
                c.solution.evals_∂e_∂p += 1

                #@info "t=$(t)\nēc=$(ēc)"
            end

            if times && c.fmu.executionConfig.eval_t_gradients
                vgp!(c, :∂e_∂t, (:indicators, ec_idcs), :time, t, ēc; accu = t̄)
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

        t̄ = t̄[1]

        @debug "pullback on d̄x, ȳ, ēc = $(d̄x), $(ȳ), $(ēc)\nt= $(t)s\nx=$(x)\nx_d=$(x_d)\ndx=$(dx)\n(x̄=$(x̄), x̄_d=$(x̄_d), ū=$(ū), p̄=$(p̄), t̄=$(t̄))"

        if c.fmu.executionConfig.snapshot_every_step
            #stopSampling(c)
            #apply!(c, tmp_snapshot_inner)
            freeSnapshot!(pullback_snapshot)
        end

        d̄x = zeros(length(dx)) # ZeroTangent()
        ȳ = zeros(length(y)) # ZeroTangent()
        ēc = zeros(length(ec)) # ZeroTangent() # copy(ec) # 

        # [ToDo] This needs to be a tuple... but this prevents pre-allocation...
        return (
            f̄,
            c̄Ref,
            d̄x,
            d̄x_refs,
            ȳ,
            ȳ_refs,
            x̄,
            ū,
            ū_refs,
            p̄,
            p̄_refs,
            ēc,
            ēc_idcs,
            t̄,
            x̄_d,
        )
    end

    @debug "rrule end: $((Ω, eval_pullback))"

    return (Ω, eval_pullback)
end

# dx, y, x, u, p, ec, t
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:ForwardDiff.Dual},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:ForwardDiff.Dual},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:ForwardDiff.Dual},
    u::AbstractVector{<:ForwardDiff.Dual},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ForwardDiff.Dual,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:ReverseDiff.TrackedReal},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:ReverseDiff.TrackedReal},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:ReverseDiff.TrackedReal},
    u::AbstractVector{<:ReverseDiff.TrackedReal},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ReverseDiff.TrackedReal,
    x_d::AbstractVector{<:Real},
)

# dx, y, x, u, t
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:ForwardDiff.Dual},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:ForwardDiff.Dual},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:ForwardDiff.Dual},
    u::AbstractVector{<:ForwardDiff.Dual},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ForwardDiff.Dual,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:ReverseDiff.TrackedReal},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:ReverseDiff.TrackedReal},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:ReverseDiff.TrackedReal},
    u::AbstractVector{<:ReverseDiff.TrackedReal},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ReverseDiff.TrackedReal,
    x_d::AbstractVector{<:Real},
)

# x, u
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:ForwardDiff.Dual},
    u::AbstractVector{<:ForwardDiff.Dual},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:ReverseDiff.TrackedReal},
    u::AbstractVector{<:ReverseDiff.TrackedReal},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

# x, u, t
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:ForwardDiff.Dual},
    u::AbstractVector{<:ForwardDiff.Dual},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ForwardDiff.Dual,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:ReverseDiff.TrackedReal},
    u::AbstractVector{<:ReverseDiff.TrackedReal},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ReverseDiff.TrackedReal,
    x_d::AbstractVector{<:Real},
)

# x, p
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:ForwardDiff.Dual},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:ReverseDiff.TrackedReal},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

# t
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ForwardDiff.Dual,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ReverseDiff.TrackedReal,
    x_d::AbstractVector{<:Real},
)

# x
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:ForwardDiff.Dual},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:ReverseDiff.TrackedReal},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

# u
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:Real},
    u::AbstractVector{<:ForwardDiff.Dual},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:Real},
    u::AbstractVector{<:ReverseDiff.TrackedReal},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

# p
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

# ec
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

# x, t
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:ForwardDiff.Dual},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ForwardDiff.Dual,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:ReverseDiff.TrackedReal},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ReverseDiff.TrackedReal,
    x_d::AbstractVector{<:Real},
)

# x, ec, t
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:ForwardDiff.Dual},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ForwardDiff.Dual,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:ReverseDiff.TrackedReal},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ReverseDiff.TrackedReal,
    x_d::AbstractVector{<:Real},
)

# ec, t
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ForwardDiff.Dual,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ReverseDiff.TrackedReal,
    x_d::AbstractVector{<:Real},
)

# x, ec
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:ForwardDiff.Dual},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:ReverseDiff.TrackedReal},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:Real},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

# x, p, t
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:ForwardDiff.Dual},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ForwardDiff.Dual,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:ReverseDiff.TrackedReal},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:Real},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ReverseDiff.TrackedReal,
    x_d::AbstractVector{<:Real},
)

# x, p, ec, t
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:ForwardDiff.Dual},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ForwardDiff.Dual,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:ReverseDiff.TrackedReal},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::ReverseDiff.TrackedReal,
    x_d::AbstractVector{<:Real},
)

# x, p, ec
@ForwardDiff_frule FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:fmiValueReference},
    x::AbstractVector{<:ForwardDiff.Dual},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:fmiValueReference},
    p::AbstractVector{<:ForwardDiff.Dual},
    p_refs::AbstractVector{<:fmiValueReference},
    ec::AbstractVector{<:ForwardDiff.Dual},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

@grad_from_chainrules FMIBase.eval!(
    cRef::UInt64,
    dx::AbstractVector{<:Real},
    dx_refs::AbstractVector{<:fmiValueReference},
    y::AbstractVector{<:Real},
    y_refs::AbstractVector{<:UInt32},
    x::AbstractVector{<:ReverseDiff.TrackedReal},
    u::AbstractVector{<:Real},
    u_refs::AbstractVector{<:UInt32},
    p::AbstractVector{<:ReverseDiff.TrackedReal},
    p_refs::AbstractVector{<:UInt32},
    ec::AbstractVector{<:ReverseDiff.TrackedReal},
    ec_idcs::AbstractVector{<:fmiValueReference},
    t::Real,
    x_d::AbstractVector{<:Real},
)

# FiniteDiff Jacobians

abstract type FMUSensitivities end

mutable struct FMUJacobian{C,T,F} <: FMUSensitivities
    valid::Bool
    colored::Bool
    instance::C

    mtx::Matrix{T}
    jvp::Vector{T}
    vjp::Vector{T}

    f_refs::Union{Vector{UInt32},Tuple{Symbol,Vector{UInt32}}}
    x_refs::Union{Vector{UInt32},Symbol}
    f_refs_set::Union{Set,Nothing}

    f::F

    #cache::FiniteDiff.JacobianCache
    #colors::

    validations::Int
    colorings::Int

    function FMUJacobian{T}(
        instance::C,
        f_refs::Union{Vector{UInt32},Tuple{Symbol,Vector{UInt32}}},
        x_refs::Union{Vector{UInt32},Symbol},
    ) where {C,T}

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

        inst = new{C,T,F}()
        inst.f = f
        inst.instance = instance
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

mutable struct FMUGradient{C,T,F} <: FMUSensitivities
    valid::Bool
    colored::Bool
    instance::C

    vec::Vector{T}
    gvp::Vector{T}
    vgp::Vector{T}

    f_refs::Union{Vector{UInt32},Tuple{Symbol,Vector{UInt32}}}
    x_refs::Union{Vector{UInt32},Symbol}
    f_refs_set::Union{Set,Nothing}

    f::F

    #cache::FiniteDiff.GradientCache
    #colors::

    validations::Int
    colorings::Int

    function FMUGradient{T}(
        instance::C,
        f_refs::Union{Vector{UInt32},Tuple{Symbol,Vector{UInt32}}},
        x_refs::Union{UInt32,Symbol},
    ) where {C,T}

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

        inst = new{C,T,F}()
        inst.f = f
        inst.instance = instance
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

function f_∂v_∂v(jac::FMUJacobian, f, x)
    setReal(jac.instance, jac.x_refs, x; track = false)
    getReal!(jac.instance, jac.f_refs, f)
    return f
end

function f_∂e_∂v(jac::FMUJacobian, f, x)
    symbol, f_refs = jac.f_refs
    @assert symbol == :indicators "Called `f_∂e_∂v` but f_refs is not in event indicator shape."
    setReal(jac.instance, jac.x_refs, x; track = false)
    getEventIndicators!(jac.instance, f, f_refs)
    return f
end

function f_∂e_∂t(jac::FMUGradient, f, x)
    setTime(jac.instance, x; track = false)
    getEventIndicators!(jac.instance, f, jac.f_refs[2])
    return f
end

function f_∂v_∂t(jac::FMUGradient, f, x)
    setTime(jac.instance, x; track = false)
    getReal!(jac.instance, jac.f_refs, f)
    return f
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

function onehot!(seed, i::Integer) # [ToDo] this could be solved without allocations
    seed .= 0.0
    seed[i] = 1.0
    return seed
end

function validate!(jac::FMUJacobian, x::AbstractVector)

    rows = length(jac.f_refs)
    cols = length(jac.x_refs)

    # only VR to VR value references can be sampled using built-in functions in FMI
    if !isa(jac.f_refs, Tuple) && !isa(jac.x_refs, Symbol)
        if jac.instance.fmu.executionConfig.sensitivity_strategy ==
           :FMIDirectionalDerivative && providesDirectionalDerivatives(jac.instance.fmu)

            # ToDo: use directional derivatives with sparsitiy information!
            # ToDo: Optimize allocation (onehot)
            # [Note] Jacobian is sampled column by column

            seed = zeros(getRealType(jac.instance), cols)

            for i = 1:cols
                status = getDirectionalDerivative!(
                    jac.instance,
                    jac.f_refs,
                    jac.x_refs,
                    onehot!(seed, i),
                    view(jac.mtx, 1:rows, i),
                )
            end
        elseif jac.instance.fmu.executionConfig.sensitivity_strategy ==
               :FMIAdjointDerivative && providesAdjointDerivatives(jac.instance.fmu)

            # ToDo: use directional derivatives with sparsitiy information!
            # ToDo: Optimize allocation (onehot)
            # [Note] Jacobian is sampled row by row

            seed = zeros(getRealType(jac.instance), rows)

            for i = 1:rows
                getAdjointDerivative!(
                    jac.instance,
                    jac.f_refs,
                    jac.x_refs,
                    onehot!(seed, i),
                    view(jac.mtx, 1:cols, i),
                )
            end
        elseif jac.instance.fmu.executionConfig.sensitivity_strategy == :FiniteDiff

            seed = zeros(getRealType(jac.instance), cols)

            # ToDo: also use FiniteDiff here!
            #finite_diff_jacobian!(jac, x)

            for i = 1:cols
                sampleDirectionalDerivative!(
                    jac.instance,
                    jac.f_refs,
                    jac.x_refs,
                    onehot!(seed, i),
                    view(jac.mtx, 1:rows, i);
                    Δx = jac.instance.fmu.executionConfig.finitediff_absstep,
                )
            end
        else
            @assert false "Unknown sensitivity strategy `$(jac.instance.fmu.executionConfig.sensitivity_strategy)`."
        end
    else
        finite_diff_jacobian!(jac, x)
    end

    jac.validations += 1
    jac.valid = true
    return nothing
end

function finite_diff_jacobian!(jac, x)

    # FMUs remember their state, therefore me need to check the state before sampling ...
    if !isa(jac.x_refs, Symbol)
        x_old = FMIBase.getReal(jac.instance, jac.x_refs)
    end

    # cache = FiniteDiff.JacobianCache(x)
    fdtype = jac.instance.fmu.executionConfig.finitediff_fdtype

    # this is FiniteDiff default behaviour
    relstep = FiniteDiff.default_relstep(fdtype, eltype(x))
    absstep = relstep

    if jac.instance.fmu.executionConfig.finitediff_relstep >= 0.0
        relstep = jac.instance.fmu.executionConfig.finitediff_relstep
    end

    if jac.instance.fmu.executionConfig.finitediff_absstep >= 0.0
        absstep = jac.instance.fmu.executionConfig.finitediff_absstep
    end

    #@info "x: $(x)"
    #@info "size(jac.mtx): $(size(jac.mtx))"

    #jac.mtx = transpose(jac.mtx)

    # ToDo: for setting `fdtype`, a weird error message is generated (looks like a different sampling pattern)
    FiniteDiff.finite_difference_jacobian!(
        jac.mtx,
        (_dx, _x) -> jac.f(jac, _dx, _x),
        x,
        fdtype;
        relstep = relstep,
        absstep = absstep, # 
    ) # , cache)

    #jac.mtx = transpose(jac.mtx)

    # ... and set it afterwards
    if !isa(jac.x_refs, Symbol)
        FMIBase.setReal(jac.instance, jac.x_refs, x_old)
    end
    return nothing
end

function finite_diff_gradient!(grad, x)

    # FMUs remember their state, therefore me need to check the state before sampling ...
    if !isa(grad.x_refs, Symbol)
        x_old = FMIBase.getReal(grad.instance, grad.x_refs)
    end

    # cache = FiniteDiff.JacobianCache(x)
    fdtype = grad.instance.fmu.executionConfig.finitediff_fdtype

    # this is FiniteDiff default behaviour
    relstep = FiniteDiff.default_relstep(fdtype, eltype(x))
    absstep = relstep

    if grad.instance.fmu.executionConfig.finitediff_relstep >= 0.0
        relstep = grad.instance.fmu.executionConfig.finitediff_relstep
    end

    if grad.instance.fmu.executionConfig.finitediff_absstep >= 0.0
        absstep = grad.instance.fmu.executionConfig.finitediff_absstep
    end

    # cache = FiniteDiff.GradientCache(x)
    FiniteDiff.finite_difference_gradient!(
        grad.vec,
        (_dx, _x) -> (grad.f(grad, _dx, _x)),
        x,
        fdtype;
        relstep = relstep,
        absstep = absstep,
    ) # , cache)

    # ... and set it afterwards
    if !isa(grad.x_refs, Symbol)
        FMIBase.setReal(grad.instance, grad.x_refs, x_old)
    end
    return nothing
end

function validate!(grad::FMUGradient, x::Real)

    if !isa(grad.f_refs, Tuple) && !isa(grad.x_refs, Symbol)

        if grad.instance.fmu.executionConfig.sensitivity_strategy ==
           :FMIDirectionalDerivative && providesDirectionalDerivatives(grad.instance.fmu)

            # ToDo: use directional derivatives with sparsitiy information!
            getDirectionalDerivative!(
                grad.instance,
                grad.f_refs,
                grad.x_refs,
                ones(length(jac.f_refs)),
                grad.vec,
            )
        elseif grad.instance.fmu.executionConfig.sensitivity_strategy == :FiniteDiff
            finite_diff_gradient!(grad, x)
        else
            @assert false "Unknown sensitivity strategy `$(grad.instance.fmu.executionConfig.sensitivity_strategy)`."
        end
    else
        finite_diff_gradient!(grad, x)
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
        #if length(jac.mtx) != ref_length(jac.f_refs) * ref_length(jac.x_refs) # this is cheaper
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
        gra.vec = similar(gra.vec, ref_length(gra.f_refs))
        gra.gvp = similar(gra.gvp, ref_length(gra.f_refs))
        gra.vgp = similar(gra.vgp, ref_length(gra.x_refs))

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

function jvp!(jac::FMUJacobian, x::AbstractVector, v::AbstractVector; jvp = jac.jvp)
    FMISensitivity.update!(jac, x)
    #return jac.mtx * v
    mul!(jvp, jac.mtx, v)
    return nothing
end

function vjp!(jac::FMUJacobian, x::AbstractVector, v::AbstractVector; vjp = jac.vjp)
    FMISensitivity.update!(jac, x)
    #return jac.mtx' * v 
    mul!(vjp, jac.mtx', v)
    return nothing
end

function gvp!(grad::FMUGradient, x, v; gvp = grad.gvp)
    FMISensitivity.update!(grad, x)
    #return grad.vec * v 
    mul!(gvp, grad.vec, v)
    return nothing
end

function vgp!(grad::FMUGradient, x, v, vgp = grad.vgp)
    FMISensitivity.update!(grad, x)
    mul!(vgp, grad.vec', v)
    return nothing
end
