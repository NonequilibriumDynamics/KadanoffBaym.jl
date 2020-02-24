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
  # @assert !integrator.u_modified
  @assert integrator.dt_idxs == (1, 0)
  @assert ==(integrator.t_idxs...)
  
  T = first(integrator.t_idxs)

  # Step all previous times
  for t in 1:T
    integrator.t_idxs = (T, t)
    abm43!(integrator, caches.line[t], integrator.f)
  end

  # Step the diagonal through reflections
  if integrator.f_diag === nothing
    integrator.t_idxs = (T, T+1)
    abm43!(integrator, caches.line[T+1], integrator.f)
    integrator.t_idxs = (T, T)

    # Walking through the diagonal requires a backwards reflection
    foreach(integrator.u) do uᵢ
      uᵢ[(T+1, T+1)...] = -adjoint(uᵢ[(T+1, T+1)...])
    end

  # Step the diagonal with the diagonal function
  else
    integrator.dt_idxs = (1, 1)
    abm43!(integrator, caches.diagonal, integrator.f_diag)
    integrator.dt_idxs = (1, 0)
  end

  # Sets up rhs memory of the next cache
  fill_cache!(integrator, caches.line[T+1])

  @assert integrator.dt_idxs == (1, 0)
  @assert integrator.t_idxs == (T, T)
end

"""
Adams-Bashfourth-Moulton 43 predictor corrector method
y_{n+1} = y_{n} + Δt/24 [9 f(̃y_{n+1}) + 19 f(y_{n}) - 5 f(y_{n-1}) + f(y_{n-2})]
̃y_{n+1} = y_{n} + Δt/24 [55 f(y_{n}) - 59 f(y_{n-1}) + 37 f(y_{n-2}) - 9 f(y_{n-3})]
"""
@muladd function abm43!(integrator,cache::OrdinaryDiffEq.ABM43ConstantCache,f)
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
      @. u[(t_idxs .+ dt_idxs)..., ..] = u[t_idxs..., ..] + (dt/24) * (55*k1 - 59*k2 + 37*k3 - 9*k4) # Predictor
    end
    
    k = f(u, p, (t_idxs .+ dt_idxs)...)
    # integrator.destats.nf += 1

    @. u[(t_idxs .+ dt_idxs)..., ..] = u[t_idxs..., ..] + (dt/24) * (9*k + 19*k1 - 5*k2 + k3) # Corrector
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
@muladd function eulerHeun!(integrator,k1,cache::OrdinaryDiffEq.ABM43ConstantCache,f)
  @unpack t_idxs, dt_idxs, dt, u, p = integrator
  @unpack k2,k3,k4 = cache
  # @assert !integrator.u_modified

  @. u[(t_idxs .+ dt_idxs)..., ..] = u[t_idxs..., ..] + dt * k1 # Predictor

  k = f(u, p, (t_idxs .+ dt_idxs)...)
  # integrator.destats.nf += 1

  @. u[(t_idxs .+ dt_idxs)..., ..] = u[t_idxs..., ..] + (dt/2) * (k + k1) # Corrector
end

"""
Reflections allow us to fill up the cache
"""
function fill_cache!(integrator, cache::OrdinaryDiffEq.ABM43ConstantCache)
  @unpack t_idxs, u, p, f = integrator

  t, t′ = t_idxs
  ts = t:-1:max(t-3, 1)

  cache.step = 1
  for (t, k) in zip(ts, (:k2, :k3, :k4))
    setfield!(cache, k, f(u, p, t, t′+1))
    cache.step += 1
  end
end
