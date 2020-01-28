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
  @assert dt_idxs == (1, 0)
  T, T = t_idxs

  # Step all previous times
  for t in 1:T
    integrator.t_idxs = (T, t)
    abm43!(integrator, caches.line[t], integrator.f)
  end

  # Sets up memory rhs of the next cache
  fill_cache!(integrator, caches.line[T+1])

  # Step the diagonal through reflections
  if integrator.f_diag === nothing
    integrator.t_idxs = (T, T+1)
    abm43!(integrator, caches.line[T+1], integrator.f)
    integrator.t_idxs = (T, T)

    # Walking through the diagonal requires a backwards reflection
    foreach(integrator.u.x) do uᵢ
      uᵢ[(T+1, T+1)...] = -adjoint(uᵢ[(T+1, T+1)...])
    end

  # Step the diagonal with the diagonal function
  else
    integrator.dt_idxs = (1, 1)
    abm43!(integrator, caches.diagonal, integrator.f_diag)
    integrator.dt_idxs = (1, 0)
  end

  @assert dt_idxs == (1, 0)
  @assert integrator.t_idxs == (T, T)
end

"""
Adams-Bashfourth-Moulton 43 predictor corrector method
y_{n+1} = y_{n} + Δt/24 [9 f(̃y_{n+1}) + 19 f(y_{n}) - 5 f(y_{n-1}) + f(y_{n-2})]
̃y_{n+1} = y_{n} + Δt/24 [55 f(y_{n}) - 59 f(y_{n-1}) + 37 f(y_{n-2}) - 9 f(y_{n-3})]
"""
function abm43!(integrator,cache::OrdinaryDiffEq.ABM43ConstantCache,f)
  @unpack t_idxs, dt_idxs, dt, u, p = integrator
  @unpack k2,k3,k4 = cache
  # @assert !integrator.u_modified

  k1 = f(u, p, t_idxs...)
  # integrator.destats.nf += 1

  if cache.step <= 2 # Euler-Heun 
    cache.step += 1
    eulerHeun!(integrator, k1, cache, f)
    
  else # Adams-Bashfourth-Moulton 4-3
    if cache.step <= 3
      cache.step += 1
      eulerHeun!(integrator, k1, cache, f) # Predictor
    else
      foreach(zip(u.x,k1.x,k2.x,k3.x,k4.x)) do (uᵢ,k1ᵢ,k2ᵢ,k3ᵢ,k4ᵢ)
        uᵢ[(t_idxs .+ dt_idxs)...] = uᵢ[t_idxs...] + (dt/24) * (55*k1ᵢ - 59*k2ᵢ + 37*k3ᵢ - 9*k4ᵢ) # Predictor
      end
    end
    
    k = f(u, p, (t_idxs .+ dt_idxs)...)
    # integrator.destats.nf += 1

    foreach(zip(u.x,k.x,k1.x,k2.x,k3.x)) do (uᵢ,kᵢ,k1ᵢ,k2ᵢ,k3ᵢ)
      uᵢ[(t_idxs .+ dt_idxs)...] = uᵢ[t_idxs...] + (dt/24) * (9*kᵢ + 19*k1ᵢ - 5*k2ᵢ + k3ᵢ) # Corrector
    end
  end

  # Update ABM's cache
  cache.k4 = k3
  cache.k3 = k2
  cache.k2 = k1
end

"""
Euler-Heun's method
y_{n+1} = y_{n} + Δt/2 [f(t+Δt, ̃y_{n+1}) + f(t, y_{n})]
̃y_{n+1} = y_{n} + Δt f(t, y_{n})
"""
function eulerHeun!(integrator,k1,cache::OrdinaryDiffEq.ABM43ConstantCache,f)
  @unpack t_idxs, dt_idxs, dt, u, f, p = integrator
  @unpack k2,k3,k4 = cache
  # @assert !integrator.u_modified

  foreach(zip(u.x, k1.x)) do (uᵢ, k1ᵢ)
    uᵢ[(t_idxs .+ dt_idxs)...] = uᵢ[t_idxs...] + dt * k1ᵢ # Predictor
  end

  k = f(u, p, (t_idxs .+ dt_idxs)...)
  # integrator.destats.nf += 1

  foreach(zip(u.x,k.x,k1.x)) do (uᵢ,kᵢ,k1ᵢ)
    uᵢ[(t_idxs .+ dt_idxs)...] = uᵢ[t_idxs...] + (dt/2) * (kᵢ + k1ᵢ) # Corrector
  end
end

"""
Reflections allow us to fill up the cache
"""
function fill_cache!(integrator, cache::OrdinaryDiffEq.ABM43ConstantCache)
  @unpack t_idxs, u, p, f = integrator

  t, t′ = t_idxs
  ts = t:-1:max(t-3, 1)

  for (t, k) in zip(ts, (:k2, :k3, :k4))
    setfield!(cache, k, f(u, p, t, t′+1))
    cache.step += 1
  end
end
