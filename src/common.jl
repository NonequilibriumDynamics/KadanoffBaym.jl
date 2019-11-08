struct KB{algType} <: DiffEqBase.AbstractODEAlgorithm end
export KB

alg_type(::KB{algType}) where {algType} = algType()
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
function OrdinaryDiffEq.alg_cache(::KB{algType}, args...) where algType
  cache = OrdinaryDiffEq.alg_cache(algType(), args...)
  return KBCaches{typeof(cache)}([recursivecopy(cache)], cache) # repeat cache
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
  t′ = tspan[1]

  if (!adaptive && dt == tType(0))
    error("Fixed timestep methods require a choice of dt")
  end

  f = prob.f
  p = prob.p

  u = recursivecopy(prob.u0) # it's also ks

  uType = typeof(u)
  @assert uType <: ArrayPartition "Functions need to be inside an ArrayPartition"
  @assert eltype(u.x) <: GreenFunction "Can only timestep Green functions!"

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
    cache = alg_cache(alg,0,recursivecopy(u),0,0,0,0,0,0,0,0,0,0,0,Val(false))

    steps = Int(ceil(Int,(tspan[2]-tspan[1])/dt) + 1)

    ts = Matrix{Tuple{tType,tType}}(undef,steps,steps)
    ts[1,1] = (t, t′)

    timeseries = recursivecopy(u)
    timeseries = uType(map(g->resize(g, (11,11)), timeseries.x))
  else
    @assert false "Not yet supported"
  end

  QT = tTypeNoUnits <: Integer ? typeof(qmin) : tTypeNoUnits

  id = OrdinaryDiffEq.InterpolationData(f,timeseries,ts,[],true,cache)
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
end

@with_kw mutable struct KBIntegrator
  t_idxs
  dt_idxs
  dt
  u
  f
  fsalfirst
  fsallast
  k
  p = nothing
  u_modified::Bool = false
end

function DiffEqBase.solve!(integrator::KBIntegrator)
  @inbounds while !isempty(integrator.opts.tstops)
    while integrator.tdir * integrator.t < top(integrator.opts.tstops)
    #   loopheader!(integrator)
    #   if check_error!(integrator) != :Success
    #     return integrator.sol
    #   end
      perform_step!(integrator,integrator.cache)
    #   loopfooter!(integrator)
      if isempty(integrator.opts.tstops)
        break
      end
    end
    handle_tstop!(integrator) # NOTE: Not tested
  end
  # postamble!(integrator)

  # f = integrator.sol.prob.f

  # if DiffEqBase.has_analytic(f)
  #   DiffEqBase.calculate_solution_errors!(integrator.sol;timeseries_errors=integrator.opts.timeseries_errors,dense_errors=integrator.opts.dense_errors)
  # end
  # if integrator.sol.retcode != :Default
  #   return integrator.sol
  # end
  integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol,:Success) # NOTE: Not tested
end
