struct KB{algType} <: OrdinaryDiffEq.OrdinaryDiffEqAlgorithm end
export KB

OrdinaryDiffEq.alg_order(::KB{algType}) where {algType} = OrdinaryDiffEq.alg_order(algType())
OrdinaryDiffEq.isfsal(::KB) = false

"""
Caches hold previous values needed by the timesteppers
"""
# OrdinaryDiffEq.@cache
mutable struct KBCaches{algCacheType} <: OrdinaryDiffEq.OrdinaryDiffEqConstantCache
  caches::Array{algCacheType, 1}
end

"""
Recycle OrdinaryDiffEq's Adams-Bashfourth-Moulton caches
"""
function OrdinaryDiffEq.alg_cache(::KB{algType}, args...) where algType
  cache = OrdinaryDiffEq.alg_cache(algType(), args...)
  return KBCaches{typeof(cache)}([cache])
end

"""
Code much like OrdinaryDiffEq.jl, which is licensed under the MIT "Expat" License
"""
function DiffEqBase.__init(prob::ODEProblem, alg::KB, dt::Float64,
                           abstol=nothing, reltol=nothing,
  dtmax = eltype(prob.tspan)((prob.tspan[end]-prob.tspan[1])),
  callback=nothing)

  tType = eltype(prob.tspan)
  tspan = prob.tspan
  tdir = sign(tspan[end]-tspan[1])

  t = tspan[1]
  t′ = tspan[1]

  f = prob.f
  p = prob.p

  u = recursivecopy(prob.u0) # it's also ks

  uType = typeof(u)
  @assert uType <: Vector
  @assert eltype(uType) <: GreenFunction

  uBottomEltype = recursive_bottom_eltype(u)
  uBottomEltypeNoUnits = recursive_unitless_bottom_eltype(u)

  # ks = Vector{uType}(undef, 0)

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
  else # has units!
    rate_prototype = u/oneunit(tType)
  end
  
  rateType = typeof(rate_prototype)
  
  saveat = eltype(prob.tspan)[]
  tstops = eltype(prob.tspan)[]
  d_discontinuities= eltype(prob.tspan)[]
  
  tstops_internal, saveat_internal, d_discontinuities_internal =
  OrdinaryDiffEq.tstop_saveat_disc_handling(tstops, saveat, d_discontinuities, tspan)
  
  callbacks_internal = OrdinaryDiffEq.CallbackSet(callback,prob.callback)

  max_len_cb = DiffEqBase.max_vector_callback_length(callbacks_internal)
  if max_len_cb isa VectorContinuousCallback
    callback_cache = DiffEqBase.CallbackCache(max_len_cb.len,uBottomEltype,uBottomEltype)
  else
    callback_cache = nothing
  end
  
  ksEltype = Vector{rateType}
  timeseries_init = typeof(prob.u0)[]
  timeseries = convert(Vector{uType},timeseries_init)

  ts_init = eltype(prob.tspan)[]
  ts = convert(Vector{tType},ts_init)
  
  ks_init = []
  ks = convert(Vector{ksEltype},ks_init)
  alg_choice = Int[]
    
  adptive = OrdinaryDiffEq.isadaptive(alg)
  if !adptive
    steps = ceil(Int,(tspan[2]-tspan[1])/dt)
    sizehint!(timeseries, steps+1)
    sizehint!(ts,steps+1)
    sizehint!(ks,steps+1)
  else
    @assert false
  end
  
  copyat_or_push!(ts,1,t)
  copyat_or_push!(timeseries,1,u)
  copyat_or_push!(ks,1,[rate_prototype])
    
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

function OrdinaryDiffEq.initialize!(integrator,cache::KBCache)
  assert false
end

function OrdinaryDiffEq.perform_step!(integrator,cache::KBCache,repeat_step=false)
  @unpack t, dt = integrator
end

"""
Euler-Heun's method

y_{n+1} = y_{n} + Δt/2 (f(t, y_{n}) + f(t+Δt, ̃y_{n+1}))
̃y_{n+1} = y_{n} + Δt f(t, y_{n})
"""
function EulerHeun!(integrator,cache,repeat_step=false)
  @unpack t_idxs, dt_idxs, dt, u, f, p = integrator
  @unpack f⁽¹⁾ = cache

  u_cached = map(y -> y[t_idxs...], u)
    
  foreach(zip(u, f⁽¹⁾)) do (yᵢ, f⁽¹⁾ᵢ)
    yᵢ[t_idxs...] += dt * f⁽¹⁾ᵢ # predictor
  end

  f⁽²⁾ = f((t_idxs .+ dt_idxs)..., u, p)
#   integrator.destats.nf += 1

  foreach(zip(u, u_cached, f⁽¹⁾, f⁽²⁾)) do (yᵢ, y′ᵢ,f⁽¹⁾ᵢ,f⁽²⁾ᵢ)
    yᵢ[t_idxs...] = y′ᵢ # return for cached value
    yᵢ[(t_idxs .+ dt_idxs)...] = y′ᵢ + dt/2 * (f⁽¹⁾ᵢ + f⁽²⁾ᵢ) # corrector
  end
  u
end
