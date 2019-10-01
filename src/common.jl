struct KB{algType} <: OrdinaryDiffEq.OrdinaryDiffEqAlgorithm end
export KB

OrdinaryDiffEq.alg_order(::KB{algType}) where {algType} = OrdinaryDiffEq.alg_order(algType())
OrdinaryDiffEq.isfsal(::KB) = false

cachify(::KB{algType}) where algType = eval(Symbol(string(algType) * "ConstantCache"))

# OrdinaryDiffEq.@cache
mutable struct KBCache{algCacheType} <: OrdinaryDiffEq.OrdinaryDiffEqConstantCache
  caches::Array{algCacheType, 1}
end

function OrdinaryDiffEq.alg_cache(::KB{algType},u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol_internal,p,calck,::Type{Val{false}}) where algType
  cache = OrdinaryDiffEq.alg_cache(algType(),u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol_internal,p,calck,Val{false})
  return KBCacheCache{cachify(algType)}()
end

# Code much like OrdinaryDiffEq.jl, which is licensed under the MIT "Expat" License
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
end
