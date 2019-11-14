"""
Initializes
"""
function initialize!(integrator::KBIntegrator,caches::KBCaches)
  nothing
end

"""
Performs a Kadanoff-Baym time step
"""
function perform_step!(integrator::KBIntegrator,caches::KBCaches,repeat_step=false)
  @unpack t_idxs, dt_idxs = integrator

  # @assert !integrator.u_modified
  @assert dt_idxs == (0, 1)

  _, t′ = t_idxs

  for t in 1:t′
    integrator.t_idxs = (t, t′)
    cache = caches.line[t]
    abm43!(integrator, cache)
  end

  # Step the diagonal!
  integrator.t_idxs = t, t′ = (t′,t′) .+ reverse(dt_idxs)
  cache = caches.line[t]
  abm43!(integrator, cache)
end

"""
Adams-Bashfourth-Moulton 43 predictor corrector method
y_{n+1} = y_{n} + Δt/24 [9 f(̃y_{n+1}) + 19 f(y_{n}) - 5 f(y_{n-1}) + f(y_{n-2})]
̃y_{n+1} = y_{n} + Δt/24 [55 f(y_{n}) - 59 f(y_{n-1}) + 37 f(y_{n-2}) - 9 f(y_{n-3})]

Euler-Heun's method
y_{n+1} = y_{n} + Δt/2 [f(t+Δt, ̃y_{n+1}) + f(t, y_{n})]
̃y_{n+1} = y_{n} + Δt f(t, y_{n})
"""
function abm43!(integrator,cache::OrdinaryDiffEq.ABM43ConstantCache,repeat_step=false)
  @unpack t_idxs, dt_idxs, dt, u, f, p = integrator
  @unpack k2,k3,k4 = cache
  # @assert !integrator.u_modified

  k1 = f(u, p, t_idxs...)
  # integrator.destats.nf += 1

  u′ = ArrayPartition(map(uᵢ -> uᵢ[t_idxs...], u.x)...) # caches u

  if cache.step <= 3 # Euler-Heun
    foreach(zip(u.x, k1.x)) do (uᵢ, k1ᵢ)
      print(typeof(dt * k1ᵢ))
      uᵢ[t_idxs...] += dt * k1ᵢ # Predictor
    end

    k = f(u, p, (t_idxs .+ dt_idxs)...)
    # integrator.destats.nf += 1

    foreach(zip(u.x,u′.x,k.x,k1.x)) do (uᵢ,u′ᵢ,kᵢ,k1ᵢ)
      uᵢ[t_idxs...] = u′ᵢ # reset to cached value
      uᵢ[(t_idxs .+ dt_idxs)...] = u′ᵢ + (dt/2) * (kᵢ + k1ᵢ) # Corrector
    end 

    # Update ABM's cache
    if cache.step == 1
      cache.k4 = k1
    elseif cache.step == 2
      cache.k3 = k1
    else
      cache.k2 = k1
    end
    cache.step += 1
  else # Adams-Bashfourth-Moulton 4-3
    foreach(zip(u.x,k1.x,k2.x,k3.x,k4.x)) do (uᵢ,k1ᵢ,k2ᵢ,k3ᵢ,k4ᵢ)
      uᵢ[(t_idxs .+ dt_idxs)...] = uᵢ[t_idxs...] + (dt/24) * (55*k1ᵢ - 59*k2ᵢ + 37*k3ᵢ - 9*k4ᵢ) # Predictor
    end
    
    k = f(u, p, (t_idxs .+ dt_idxs)...)
    # integrator.destats.nf += 1

    foreach(zip(u.x,u′.x,k.x,k1.x,k2.x,k3.x)) do (uᵢ,u′ᵢ,kᵢ,k1ᵢ,k2ᵢ,k3ᵢ)
      uᵢ[t_idxs...] = u′ᵢ # reset to cached value
      uᵢ[(t_idxs .+ dt_idxs)...] = u′ᵢ + (dt/24) * (9*kᵢ + 19*k1ᵢ - 5*k2ᵢ + k3ᵢ) # Corrector
    end

    # Update ABM's cache
    cache.k4 = k3
    cache.k3 = k2
    cache.k2 = k1
  end
end
