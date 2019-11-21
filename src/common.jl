struct KB{algType} <: OrdinaryDiffEq.OrdinaryDiffEqAlgorithm end
export KB

alg_type(::KB{algType}) where {algType} = algType
# OrdinaryDiffEq.isadaptive(::KB{algType}) where {algType} = OrdinaryDiffEq.isadaptive(algType())
# OrdinaryDiffEq.alg_order(::KB{algType}) where {algType} = OrdinaryDiffEq.alg_order(algType())
# OrdinaryDiffEq.isfsal(::KB) = true

"""
Caches hold previous values needed by the timesteppers
"""
# OrdinaryDiffEq.@cache
mutable struct KBCaches{algTypeCache} <: OrdinaryDiffEq.OrdinaryDiffEqConstantCache
  line::Vector{algTypeCache} # stepping through vertical or horizontal lines in the R^2 t-plane
  diagonal::algTypeCache # stepping diagonally through the R^2 t-plane
end

"""
Recycle OrdinaryDiffEq's Adams-Bashfourth-Moulton caches
"""
function OrdinaryDiffEq.alg_cache(::KB{algType}, N, args...) where algType
  cache = OrdinaryDiffEq.alg_cache(algType(), args...)
  return KBCaches{typeof(cache)}([recursivecopy(cache) for i=1:N], cache) # repeat cache
end

"""
Code much like OrdinaryDiffEq.jl, which is licensed under the MIT "Expat" License
"""
function DiffEqBase.__init(prob::ODEProblem,
                           alg::KB{algType},
                           dt=zero(eltype(prob.tspan)),
                           abstol=nothing,
                           reltol=nothing,
                           adaptive=OrdinaryDiffEq.isadaptive(algType()),
                           dtmax = eltype(prob.tspan)(prob.tspan[end]-prob.tspan[1]),
                           maxiters = adaptive ? 1000000 : typemax(Int),
                           gamma = OrdinaryDiffEq.gamma_default(algType()),
                           qmin = OrdinaryDiffEq.qmin_default(algType()),
                           qmax = OrdinaryDiffEq.qmax_default(algType()),
                           qsteady_min = OrdinaryDiffEq.qsteady_min_default(algType()),
                           qsteady_max = OrdinaryDiffEq.qsteady_max_default(algType()),
                           qoldinit = 1//10^4,
                           callback=nothing) where {algType}

  # Internals
  saveat = eltype(prob.tspan)[]
  tstops = eltype(prob.tspan)[]
  d_discontinuities= eltype(prob.tspan)[]

  tType = eltype(prob.tspan)
  tspan = prob.tspan
  tdir = sign(tspan[end]-tspan[1])

  t = tspan[1]

  if (!adaptive && dt == tType(0))
    error("Fixed timestep methods require a choice of dt")
  end

  f = prob.f
  p = prob.p

  @assert typeof(prob.u0) <: ArrayPartition "Functions need to be inside an ArrayPartition"
  @assert eltype(prob.u0.x) <: GreenFunction "Can only timestep Green functions!"

  u = ArrayPartition(map(x->x[1,1], prob.u0.x)...) # it's also ks
  uType = typeof(u)

  uBottomEltype = recursive_bottom_eltype(u)
  uBottomEltypeNoUnits = recursive_unitless_bottom_eltype(u)
  uEltypeNoUnits = recursive_unitless_eltype(u)
  tTypeNoUnits   = typeof(one(tType))

  if abstol === nothing
    if uBottomEltypeNoUnits == uBottomEltype
      abstol_internal = real(convert(uBottomEltype,oneunit(uBottomEltype)*1//10^6))
    else
      abstol_internal = real.(oneunit.(u).*1//10^6)
    end
  else
    abstol_internal = real.(abstol)
  end

  if reltol === nothing
    if uBottomEltypeNoUnits == uBottomEltype
      reltol_internal = real(convert(uBottomEltype,oneunit(uBottomEltype)*1//10^3))
    else
      reltol_internal = real.(oneunit.(u).*1//10^3)
    end
  else
    reltol_internal = real.(reltol)
  end

  dtmax > zero(dtmax) && tdir < 0 && (dtmax *= tdir) # Allow positive dtmax, but auto-convert

  if uBottomEltypeNoUnits == uBottomEltype
    rate_prototype = u
  else
    rate_prototype = u/oneunit(tType)
  end
  rateType = typeof(rate_prototype)

  tstops_internal, saveat_internal, d_discontinuities_internal = OrdinaryDiffEq.tstop_saveat_disc_handling(
    tstops, saveat, d_discontinuities, tspan)

  callbacks_internal = OrdinaryDiffEq.CallbackSet(callback)

  max_len_cb = DiffEqBase.max_vector_callback_length(callbacks_internal)
  if max_len_cb isa VectorContinuousCallback
    callback_cache = DiffEqBase.CallbackCache(max_len_cb.len,uBottomEltype,uBottomEltype)
  else
    callback_cache = nothing
  end

  if !adaptive
    steps = Int(ceil(Int,(tspan[2]-tspan[1])/dt) + 1)

    ts = map(x -> (x .* (dt, dt) .+ t), Iterators.product(1:steps, 1:steps))

    timeseries = recursivecopy(prob.u0)
    timeseries = ArrayPartition(map(g->resize(g, (steps,steps)), timeseries.x)...)
    caches = alg_cache(alg,steps,0,recursivecopy(u),0,0,0,0,0,0,0,0,0,0,0,Val(false))
  else
    @assert false "Not yet supported"
  end

  QT = tTypeNoUnits <: Integer ? typeof(qmin) : tTypeNoUnits

  id = OrdinaryDiffEq.InterpolationData(f,timeseries,ts,[],true,caches)
  beta2=OrdinaryDiffEq.beta2_default(algType())
  beta1=OrdinaryDiffEq.beta1_default(algType(),beta2)

  opts = OrdinaryDiffEq.DEOptions{
                   typeof(abstol_internal),
                   typeof(reltol_internal),QT,tType,
                   typeof(DiffEqBase.ODE_DEFAULT_NORM),
                   typeof(LinearAlgebra.opnorm),
                   typeof(callbacks_internal),
                   typeof(DiffEqBase.ODE_DEFAULT_ISOUTOFDOMAIN),
                   typeof(DiffEqBase.ODE_DEFAULT_PROG_MESSAGE),
                   typeof(DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK),
                   typeof(tstops_internal),
                   typeof(d_discontinuities_internal),
                   typeof(nothing),
                   typeof(nothing),
                   typeof(maxiters),
                   typeof(tstops),
                   typeof(saveat),
                   typeof(d_discontinuities)}(
                       maxiters,true,adaptive,abstol_internal,
                       reltol_internal,QT(gamma),QT(qmax),
                       QT(qmin),QT(qsteady_max),
                       QT(qsteady_min),QT(2),tType(dtmax),
                       eps(tType),DiffEqBase.ODE_DEFAULT_NORM,LinearAlgebra.opnorm,nothing,tstops_internal,saveat_internal,
                       d_discontinuities_internal,
                       tstops,saveat,d_discontinuities,
                       nothing,false,1000,
                       "ODE",DiffEqBase.ODE_DEFAULT_PROG_MESSAGE,true,false,
                       QT(beta1),QT(beta2),QT(qoldinit),true,
                       true,true,true,callbacks_internal,DiffEqBase.ODE_DEFAULT_ISOUTOFDOMAIN,
                       DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK,true,
                       true,false,false,false)

  destats = DiffEqBase.DEStats(0)

  sol = DiffEqBase.build_solution(prob,alg,ts,timeseries,
                    dense=true,interp=id,
                    calculate_error = false, destats=destats)

  integrator = KBIntegrator{alg_type(alg),
                            typeof(sol),
                            typeof(timeseries),
                            tType,
                            typeof(p), 
                            typeof(f),
                            typeof(caches),
                            typeof(opts)}(
                              sol, timeseries, f, p, dt, caches, opts, t, tdir)

  # # initialize_callbacks!(integrator, initialize_save)
  initialize!(integrator,integrator.caches)
  OrdinaryDiffEq.handle_dt!(integrator)
  integrator
end

mutable struct KBIntegrator{algType, solType, uType, tType, pType, fctType, cacheType, optsType}
  sol::solType
  u::uType
  f::fctType
  p::pType
  t_idxs::Tuple{Int64,Int64}
  dt_idxs::Tuple{Int64,Int64}
  dt::tType
  caches::cacheType
  opts::optsType

  t::tType
  tdir::tType
  # iter::Int
  # alg::algType
  # accept_step::Bool
  # force_stepfail::Bool

  function KBIntegrator{algType, solType, uType, tType, pType, fctType, cacheType, optsType}(sol, u, f, p, dt, caches, opts, t, tdir) where {algType, solType, uType, tType, pType, fctType, cacheType, optsType}
    new{algType, solType, uType, tType, pType, fctType, cacheType, optsType}(sol, u, f, p, (1,1), (0,1), dt, caches, opts, t, tdir)
  end
end

function DiffEqBase.solve!(integrator::KBIntegrator)
  # @inbounds while !isempty(integrator.opts.tstops)
    while first(integrator.t_idxs) < length(integrator.caches.line)
    # while integrator.tdir * integrator.t < top(integrator.opts.tstops)
      loopheader!(integrator)
#       # if check_error!(integrator) != :Success
# #     #     return integrator.sol
# #     #   end
      perform_step!(integrator,integrator.caches)
      loopfooter!(integrator)
#       if isempty(integrator.opts.tstops)
#         break
#       end
    end
# #     # handle_tstop!(integrator) # NOTE: passed
  # end
#   # postamble!(integrator) # Note: passed

#   # f = integrator.sol.prob.f

#   # if DiffEqBase.has_analytic(f)
#   #   DiffEqBase.calculate_solution_errors!(integrator.sol;timeseries_errors=integrator.opts.timeseries_errors,dense_errors=integrator.opts.dense_errors)
#   # end
#   # if integrator.sol.retcode != :Default
#   #   return integrator.sol
#   # end
#   # integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol,:Success) # NOTE: Not tested
end

function loopheader!(integrator::KBIntegrator)
  # Apply right after iterators / callbacks

  # Accept or reject the step
  # if integrator.iter > 0
    # if ((integrator.opts.adaptive && integrator.accept_step) || !integrator.opts.adaptive) && !integrator.force_stepfail
      # integrator.success_iter += 1
      # apply_step!(integrator)
    # elseif integrator.opts.adaptive && !integrator.accept_step
    #   if integrator.isout
    #     integrator.dt = integrator.dt*integrator.opts.qmin
    #   elseif !integrator.force_stepfail
    #     step_reject_controller!(integrator,integrator.alg)
    #   end
    # end
  # end

  # integrator.iter += 1
  # fix_dt_at_bounds!(integrator)
  # modify_dt_for_tstops!(integrator)
  # integrator.force_stepfail = false
  nothing
end

function loopfooter!(integrator::KBIntegrator)

  # Carry-over from callback
  # This is set to true if u_modified requires callback FSAL reset
  # But not set to false when reset so algorithms can check if reset occurred

  # integrator.reeval_fsal = false
  # integrator.u_modified = false
  ttmp = integrator.t + integrator.dt
  # if integrator.force_stepfail
  #     if integrator.opts.adaptive
  #       integrator.dt = integrator.dt/integrator.opts.failfactor
  #     elseif integrator.last_stepfail
  #       return
  #     end
  #     integrator.last_stepfail = true
  #     integrator.accept_step = false
  # elseif integrator.opts.adaptive
  #   q = stepsize_controller!(integrator,integrator.alg)
  #   integrator.isout = integrator.opts.isoutofdomain(integrator.u,integrator.p,ttmp)
  #   integrator.accept_step = (!integrator.isout && integrator.EEst <= 1.0) || (integrator.opts.force_dtmin && abs(integrator.dt) <= abs(integrator.opts.dtmin))
  #   if integrator.accept_step # Accept
  #     integrator.destats.naccept += 1
  #     integrator.last_stepfail = false
  #     dtnew = step_accept_controller!(integrator,integrator.alg,q)
  #     integrator.tprev = integrator.t
  #     # integrator.EEst has unitless type of integrator.t
  #     if typeof(integrator.EEst)<: AbstractFloat && !isempty(integrator.opts.tstops)
  #       tstop = integrator.tdir * top(integrator.opts.tstops)
  #       abs(ttmp - tstop) < 10eps(max(integrator.t,tstop)/oneunit(integrator.t))*oneunit(integrator.t) ?
  #                                 (integrator.t = tstop) : (integrator.t = ttmp)
  #     else
  #       integrator.t = ttmp
  #     end
  #     calc_dt_propose!(integrator,dtnew)
  #     handle_callbacks!(integrator)
  #   else # Reject
  #     integrator.destats.nreject += 1
    # end
  if !integrator.opts.adaptive #Not adaptive
  #   integrator.destats.naccept += 1
  #   integrator.tprev = integrator.t
  #   # integrator.EEst has unitless type of integrator.t
  #   if typeof(integrator.EEst)<: AbstractFloat && !isempty(integrator.opts.tstops)
  #     tstop = integrator.tdir * top(integrator.opts.tstops)
  #     abs(ttmp - tstop) < 10eps(integrator.t/oneunit(integrator.t))*oneunit(integrator.t) ?
  #                                 (integrator.t = tstop) : (integrator.t = ttmp)
    # else
      integrator.t = ttmp
      integrator.t_idxs = integrator.t_idxs .+ (1,1)
  #   end
  #   integrator.last_stepfail = false
  #   integrator.accept_step = true
  #   integrator.dtpropose = integrator.dt
  #   handle_callbacks!(integrator)
  end
  # if integrator.opts.progress && integrator.iter%integrator.opts.progress_steps==0
  #   @logmsg(-1,
  #   integrator.opts.progress_name,
  #   _id = :OrdinaryDiffEq,
  #   message=integrator.opts.progress_message(integrator.dt,integrator.u,integrator.p,integrator.t),
  #   progress=integrator.t/integrator.sol.prob.tspan[2])
  # end
  nothing
end
