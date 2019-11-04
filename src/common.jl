struct KB{algType} <: DiffEqBase.AbstractODEAlgorithm end
export KB

OrdinaryDiffEq.isadaptive(::KB{algType}) where {algType} = OrdinaryDiffEq.isadaptive(algType())
OrdinaryDiffEq.alg_order(::KB{algType}) where {algType} = OrdinaryDiffEq.alg_order(algType())
OrdinaryDiffEq.isfsal(::KB) = true

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
  return KBCaches{typeof(cache)}([copy(cache)], cache) # repeat cache
end

"""
Code much like OrdinaryDiffEq.jl, which is licensed under the MIT "Expat" License
"""
function DiffEqBase.__init(prob::ODEProblem,
                           alg::KB,
                           dt::Float64,
                           abstol=nothing,
                           reltol=nothing,
                           dtmax = eltype(prob.tspan)((prob.tspan[end]-prob.tspan[1])),
                           adaptive=OrdinaryDiffEq.isadaptive(alg),
                           callback=nothing)

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
  
  callbacks_internal = OrdinaryDiffEq.CallbackSet(callback,prob.callback)

  max_len_cb = DiffEqBase.max_vector_callback_length(callbacks_internal)
  if max_len_cb isa VectorContinuousCallback
    callback_cache = DiffEqBase.CallbackCache(max_len_cb.len,uBottomEltype,uBottomEltype)
  else
    callback_cache = nothing
  end
    
  if !adaptive
    steps = ceil(Int,(tspan[2]-tspan[1])/dt) + 1
    ts = Matrix{Tuple{tType,tType}}(undef, steps, steps)
    timeseries = Matrix{typeof(prob.u0)}(undef,0,0)
  else
    ts = Matrix{tType}(undef,2,2)
    @assert false "Not yet supported"
  end
  
  ts[1,1] = (t, t′)
  timeseries[1,1] = u
    
  qmin = OrdinaryDiffEq.qmin_default(alg)
  QT = tTypeNoUnits <: Integer ? typeof(qmin) : tTypeNoUnits
    
  k = rateType[]
    
  calck = (callback !== nothing && callback != CallbackSet()) || # Empty callback
                                   (prob.callback !== nothing && prob.callback != CallbackSet()) || # Empty prob.callback
                                   (!isempty(setdiff(saveat,tstops)) || true), # and no dense output

  if OrdinaryDiffEq.uses_uprev(alg, adptive) || calck
    uprev = recursivecopy(u)
  else
    uprev = u
  end

  if !OrdinaryDiffEq.alg_extrapolates(alg)
    uprev2 = uprev
  end
    
  cache = OrdinaryDiffEq.alg_cache(alg,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol_internal,p,calck,Val{isinplace(prob)})

  id = OrdinaryDiffEq.InterpolationData(f,timeseries,ts,ks,true,cache)
  beta2 = OrdinaryDiffEq.beta2_default(alg)
  beta1 = OrdinaryDiffEq.beta1_default(alg,beta2)
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
