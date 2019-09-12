using Parameters
using MuladdMacro

using Reexport
@reexport using OrdinaryDiffEq

struct KadanoffBaym <: OrdinaryDiffEq.OrdinaryDiffEqAlgorithm end
export KadanoffBaym

OrdinaryDiffEq.alg_order(alg::KadanoffBaym) = 4

# OrdinaryDiffEq.@cache
mutable struct KadanoffBaym4Cache{rateType} <: OrdinaryDiffEq.OrdinaryDiffEqConstantCache
  k2::rateType
  k3::rateType
  k4::rateType
  step::Int
end

# OrdinaryDiffEq.@cache
mutable struct KadanoffBaym43Cache{rateType} <: OrdinaryDiffEq.OrdinaryDiffEqConstantCache
  k2::rateType
  k3::rateType
  k4::rateType
  step::Int
end

function OrdinaryDiffEqAlgorithm.alg_cache(::KadanoffBaym,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Type{Val{false}})
  k2 = rate_prototype
  k3 = rate_prototype
  k4 = rate_prototype
  KadanoffBaym43Cache(k2,k3,k4,1)
end

function OrdinaryDiffEq.initialize!(integrator,cache::Union{KadanoffBaym43Cache,
                                                            KadanoffBaym4Cache})
  integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
  integrator.destats.nf += 1
  integrator.kshortsize = 2
  integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

  # Avoid undefined entries if k is an array of arrays
  integrator.fsallast = zero.(integrator.fsalfirst)
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
end

# Predictor - Corrector
@muladd function OrdinaryDiffEq.perform_step!(integrator,cache::KadanoffBaym43Cache,repeat_step=false)
  @unpack t,dt,uprev,u,f,p = integrator
  @unpack k2,k3,k4 = cache
  k1 = integrator.fsalfirst

  if integrator.u_modified
    cache.step = 1
  end
  cnt = cache.step

  if cache.step <= 2 # Euler-Heun
    cache.step += 1

    OrdinaryDiffEq.perform_step!(integrator, OrdinaryDiffEq.HeunConstantCache())

    # Update cache for ADM
    if cnt == 1
      cache.k3 = k1
    else
      cache.k2 = k1
    end

  else # Adams-Bashfourth-Moulton
    OrdinaryDiffEq.perform_step!(integrator, KadanoffBaym4Cache(k2,k3,k4,cnt)) # Predictor
    k = integrator.fsallast
    u = uprev + (dt/24)*(9*k + 19*k1 - 5*k2 + k3) # Corrector
    cache.k4 = k3
    cache.k3 = k2
    cache.k2 = k1

    # Update integrator
    integrator.fsallast = f(u, p, t+dt)
    integrator.destats.nf += 1
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.u = u
  end
end

# Corrector
@muladd function OrdinaryDiffEq.perform_step!(integrator,cache::KadanoffBaym4Cache,repeat_step=false)
  @unpack t,dt,uprev,u,f,p = integrator
  @unpack k2,k3,k4 = cache
  k1 = integrator.fsalfirst
  
  u  = uprev + (dt/24)*(55*k1 - 59*k2 + 37*k3 - 9*k4)
  cache.k4 = k3
  cache.k3 = k2
  cache.k2 = k1

  integrator.fsallast = f(u, p, t+dt)
  integrator.destats.nf += 1
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
  integrator.u = u
end