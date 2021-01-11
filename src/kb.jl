"""
  Kadanoff-Baym adaptive timestepper

Solves the 2-time (Voltera integral) differential equations
  du/dt1 = F[u](t1,t2)
  du/dt2 = G[u](t1,t2)

The signature of `f_vert` and `f_diag` must be of the form
  f_vert(u, t_grid, t1, t2)
  f_diag(u, t_grid, t1)
And if 1-time functions `v` are present
  f_vert(u, v, t_grid, t1, t2)
  f_diag(u, v, t_grid, t1)
  f_line(u, v, t_grid, t1)

## Parameters
  - `RHS` of the differential equation du/dt1 (`f_vert`)
  - `RHS` of the differential equation (d/dt1 + d/dt2)u (`f_diag`)
  - Initial value for the 2-point functions (`u0`)

## Optional keyword parameters
  - For repeated operations, the user can call

  If 1-time functions are present:
  - Initial value for the 1-point functions (`l0`)
  - 
  - 

  If higher precision Volterra integrals are needed
  - Initial value for the 2-time volterra integrals (`v0`)
  - The kernel of the `RHS` of du/dt1 (`kernel_vert`)
  - The kernel of the `RHS` of (d/dt1 + d/dt2)u (`kenerl_diag`)

## Notes
  - It is required that both `u`s and can `v`s can be **indexed** by 2-time 
    and 1-time arguments, respectively, and can be `resize!`d at will
    uᵢⱼ = resize!(u, new_size)
    vᵢ = resize!(v, new_size)
  - Unlike standard ODE solvers, `kbsolve` is designed to mutate the initial 
    condition `u`
  - The Kadanoff-Baym timestepper is a 2-time generalization of the VCABM stepper 
    presented in Ernst Hairer, Gerhard Wanner, and Syvert P Norsett
    Solving Ordinary Differential Equations I: Nonstiff Problems
"""
function kbsolve(f_vert, f_diag, u0, (t0, tmax);
  update_time=(x...)->nothing,
  l0=nothing, f_line=nothing, update_line=(x...)->nothing,
  v0=nothing, kernel_vert=nothing, kernel_diag=nothing,
  kwargs...)
  
  opts = VCABMOptions(; kwargs...)  

  # Support for initial time-grid
  if isempty(size(t0))
    t0 = [t0]
  end

  @assert last(t0) < tmax "Only t0 < tmax supported"

  # Initialize state and caches
  state, caches = begin
    t = length(t0)

    if isnothing(f_line)
      u = (u0,) # Internal representation of `u` is (`u`, `v`)
      cache_line = nothing
    else
      u = (u0, l0)
      update_line(t0, t)
      cache_line = VCABMCache{eltype(t0)}(opts.kmax, [x[1] for x in u[2]], f_line(u...,t0,t))
    end

    caches_vert = map(eachindex(t0)) do t′
      update_time(t0, t, t′)
      VCABMCache{eltype(t0)}(opts.kmax, [x[t,t′] for x in u[1]], f_vert(u...,t0,t,t′))
    end
    cache_diag = VCABMCache{eltype(t0)}(opts.kmax, [x[t,t] for x in u[1]], f_diag(u...,t0,t))

    # Resize time-length of the functions `u` and `v`
    foreach(u -> resize!.(u, last(size(last(u))) + 50), u)

    # Calculate initial dt
    # if iszero(opts.dtini)
    #   dt = initial_step(caches_vert[1].u_prev,last(t0),1,opts.atol,opts.rtol;f0=caches_vert[1].f_prev) do x...
    #     foreach((u,u′) -> u[t+1,1] = u′, u[1], x[1]); update_time([t0; x[2]], t+1, 1)
    #     update_line([t0; x[2]], t+1)
    #     f_vert(u..., [t0; x[2]], t+1, 1)
    #   end
    # else
      dt = opts.dtini
    # end

    VCABMState(u,v0,[t0; last(t0)+dt]), KBCaches{eltype(t0),typeof(cache_diag.u_prev)}(caches_vert, cache_diag, cache_line, cache_diag)
  end

  kbsolve_(f_vert, f_diag, tmax, state, caches, opts, update_time, 
    f_line, update_line, kernel_vert, kernel_diag)
end

function kbsolve_(f_vert, f_diag, tmax, state, caches, opts, update_time, 
  f_line, update_line, kernel_vert, kernel_diag)

  # This will allow us to have a unified use of f_vert and f_diag
  f(t1,t2) = isequal(t1, t2) ? f_diag(state.u..., state.t, t1) : f_vert(state.u..., state.t, t1, t2)
  # function f(t1, t2, predict::Bool)
  #   kernel(t) = isequal(t1, t2) ? t -> kernel_diag(state.t,t1,t) : t -> kernel_vert(state.t,t1,t2,t)
    
  #   if predict
  #     v′ = volterra_predict(kernel, state, caches.master)
  #   else
  #     v_next = [v[t,t′] for x in v]
  #     v′ = volterra_correct(v_next, kernel, caches.master)
  #   end

  #   foreach((v,v′) -> v[t,t′] = v′, v, v′)
  #   return isequal(t1, t2) ? f_diag(state.u..., state.t, t1) : f_vert(state.u..., state.t, t1, t2)
  # end

  # The adaptivity of the integration is controlled by a master cache
  @do_while timeloop!(state,caches.master,tmax,opts) begin
    # Current time index
    t = length(state.t)

    # We are stepping from `t-1` to `t` so we should have `t-1` caches
    @assert length(caches.vert) + 1 == t

    # Resize solution if necessary
    if (s = last(size(last(state.u[1])))) == t
      s += min(max(ceil(Int, (tmax - state.t[end]) / (state.t[end] - state.t[end-1])), 10), 20)
      foreach(u -> resize!.(u, s), state.u)
    end

    # Predictor for the 2-time functions
    for (t′, cache) in enumerate([caches.vert; caches.diag])
      u_next = predict!(state, cache)
      foreach((u,u′) -> u[t,t′] = u′, state.u[1], u_next); update_time(state.t, t, t′)
    end

    # Predictor, corrector and error estimation for the 1-time functions
    if !isnothing(f_line) 
      u_next = predict!(state, caches.line)
      foreach((u,u′) -> u[t] = u′, state.u[2], u_next); update_line(state.t, t)

      u_next = correct!(u_next, f_line(state.u..., state.t, t), caches.line)
      foreach((u,u′) -> u[t] = u′, state.u[2], u_next); update_line(state.t, t)

      caches.line.error_k = estimate_error!(u_next, caches.line, opts.atol, opts.rtol)
    end

    # Corrector and error estimation for the 2-time functions
    if !isnothing(f_line) && caches.line.error_k > one(caches.line.error_k)
      caches.master = caches.line
    else
      for (t′, cache) in enumerate([caches.vert; caches.diag])
        u_next = [x[t,t′] for x in state.u[1]] # saved in state.u[1]
        u_next = correct!(u_next, f(t,t′), cache)
        foreach((u,u′) -> u[t,t′] = u′, state.u[1], u_next); update_time(state.t, t, t′)

        # Error estimation
        cache.error_k = estimate_error!(u_next, cache, opts.atol, opts.rtol)

        # Fail step: Section III.7 Eq. (7.4)
        if cache.error_k > one(cache.error_k)
          caches.master = cache
          break
        end
      end
    end

    # If the step is accepted
    if caches.master.error_k <= one(caches.master.error_k)
      # Set the cache with highest error as the master cache
      # NOTE: this cache is not calculated before since 
      t′, caches.master = reduce((x,y) -> x[2].error_k > y[2].error_k ? x : y,
       enumerate([caches.vert; caches.diag]))

      # Adjust the master cache's order for the next time-step
      if !isnothing(f_line) && caches.line.error_k > caches.master.error_k
        caches.master = caches.line
        u_next = [x[t] for x in state.u[2]]
        adjust_order!(u_next, f_line(state.u..., state.t, t), state, caches.master, opts.kmax, opts.atol, opts.rtol)
      else
        u_next = [x[t,t′] for x in state.u[1]]
        adjust_order!(u_next, f(t,t′), state, caches.master, opts.kmax, opts.atol, opts.rtol)
      end

      # Update all caches
      update_caches!(caches, state, f_vert, f_diag, f_line)
    end
  end # timeloop!

  # Trim solution
  foreach(u -> resize!.(u, length(state.t)), state.u)
  
  return state, caches
end

function update_caches!(caches, state::VCABMState, f_vert, f_diag, f_line)
  t = length(state.t)

  @assert length(caches.vert) + 1 == t

  # Update all vertical caches
  for (t′, cache) in enumerate(caches.vert)
    cache.u_prev = [x[t,t′] for x in state.u[1]] # u_prev was saved in state.u
    cache.f_prev = f_vert(state.u..., state.t, t, t′)
    cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1
    cache.k = min(caches.master.k, cache.k+1) # Ramp up order
  end

  # Update the diagonal cache
  begin
    cache = caches.diag
    cache.u_prev = [x[t,t] for x in state.u[1]] # u_prev was saved in state.u
    cache.f_prev = f_diag(state.u..., state.t, t)
    cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1
    cache.k = min(caches.master.k, cache.k+1) # Ramp up order
  end

  # Update the 1-time cache
  if !isnothing(f_line)
    cache = caches.line
    cache.u_prev = [x[t] for x in state.u[2]] # u_prev was saved in state.u
    cache.f_prev = f_line(state.u...,state.t,t)
    cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1
    cache.k = min(caches.master.k, cache.k+1) # Ramp up order
  end

  # Add a new cache for the next time
  begin
    times = copy(state.t)

    t0 = max(t - caches.master.k, 1)
    u0 = [x[t0,t] for x in state.u[1]]
    f0 = f_vert(state.u..., state.t, t0, t)
    cache = VCABMCache{eltype(state.t)}(length(caches.diag.ϕ_n)-1, u0, f0)

    for t′ in (t0+1):t
      state.t = times[1:t′]; ϕ_and_ϕstar!(state, cache, cache.k+1)

      cache.u_prev = [x[t′,t] for x in state.u[1]]
      cache.f_prev = f_vert(state.u..., times, t′, t)
      cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1
      cache.k = min(caches.master.k, cache.k+1)
    end
    push!(caches.vert, cache)
  end

  @assert length(caches.vert) == t
end

function timeloop!(state,cache,tmax,opts)
  @unpack k, error_k= cache

  # II.4 Automatic Step Size Control, Eq. (4.13)
  q = max(inv(opts.qmax), min(inv(opts.qmin), error_k^(1/(k+1)) / opts.γ))
  dt = min((state.t[end] - state.t[end-1]) / q, opts.dtmax)

  if cache.error_k > one(cache.error_k)
    pop!(state.t) # remove t_prev
  end

  # Don't go over tmax
  if last(state.t) + dt > tmax
    dt = tmax - last(state.t)
  end
  return !iszero(dt) && !opts.stop() && (push!(state.t, last(state.t) + dt); true)
end

mutable struct KBCaches{T,F}
  vert::Vector{VCABMCache{T,F}}
  diag::VCABMCache{T,F}
  line::Union{Nothing, VCABMCache}
  master::VCABMCache # basically a reference to one of the other caches
end
