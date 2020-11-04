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

The initial condition `u` should be an array or tuple of `uᵢⱼ`s and
if 1-time functions `vᵢ` are present it should be of the form (`uᵢⱼ`, `vᵢ`).

It is also mandatory that both `u`s and can `v`s can be **indexed** by 2-time 
and 1-time arguments, respectively, and can be `resized!` at will
  uᵢⱼ = resize!(u, new_size)
  vᵢ = resize!(v, new_size)

Unlike standard ODE solvers, `kbsolve` is designed to mutate the initial 
condition `u`.

The Kadanoff-Baym timestepper is a 2-time generalization of the VCABM stepper 
presented in
Ernst Hairer, Gerhard Wanner, and Syvert P Norsett
Solving Ordinary Differential Equations I: Nonstiff Problems
"""
function kbsolve(f_vert, f_diag, u, t0, tmax; f_line=nothing, init_dt=0.0, 
  max_dt=1e-1, atol=1e-9, rtol=1e-7, max_order=12, qmax=5, qmin=1//5, γ=9//10,
  stop=()->false, update_time! =(x...)->nothing, update_line! =(x...)->nothing,
  st_ch=nothing, kernel_vert=nothing, kernel_diag=nothing, v=nothing)
  
  # Sanity checks
  if max_order < 1 || max_order > 12
    error("max_order must be between 1 and 12")
  end

  # Support for t0 being a vector of times
  if isempty(size(t0))
    t0 = [t0]
  end

  # TODO
  if last(t0) > tmax
    error("Only t0 < tmax supported")
  end

  # Internal representation of (`u`, `v`)
  if isnothing(f_line)
    u = (u,)
  end

  # Initialize state and caches
  state, caches = !isnothing(st_ch) ? st_ch : begin
    if isnothing(f_line)
      cache_line = nothing
    else
      u0 = [x[1] for x in u[2]] # Extract solution at (t0)
      update_line!(t0, 1)
      cache_line = VCABMCache{eltype(t0)}(max_order,u0,f_line(u...,t0,1))
    end

    u0 = [x[1,1] for x in u[1]] # Extract solution at (t0,t0)
    update_time!(t0, 1, 1)
    cache_vert = VCABMCache{eltype(t0)}(max_order,u0,f_vert(u...,t0,1,1))
    cache_diag = VCABMCache{eltype(t0)}(max_order,u0,f_diag(u...,t0,1))

    # Resize time-length of the functions `u` and `v`
    foreach(u -> resize!.(u, last(size(last(u))) + 50), u)

    # Calculate initial dt
    if iszero(init_dt)
      f′ = (u_next, t_next) -> begin
        foreach((u,u′) -> u[2,1] = u′, u[1], u_next)
        update_time!([t0; t_next], 2, 1)
        f_vert(u..., [t0; t_next], 2, 1)
      end
      init_dt = initial_step(f′,u0,last(t0),rtol,atol; f0=cache_vert.f_prev)
    end

    VCABMState(u,t0,init_dt), KBCaches(cache_vert,cache_diag,cache_line)    
  end

  # This will allow us to have a unified use of f_vert and f_diag
  function f(t1, t2, predict::Bool)
    if isequal(t1, t2)
      # if predict
      #   v′ = volterra_predict(t -> kernel_diag(state.t,t1,t), state, caches.master)
      # else
      #   v′ = volterra_correct()
      # end
      # foreach((v,v′) -> v[t,t′] = v′, v, v′)
      f_diag(state.u..., state.t, t1)
    else
      # if predict
      #   v′ = volterra_predict(t -> kernel_vert(state.t,t1,t2,t), state, caches.master)
      # else
      #   v′ = volterra_correct()
      # end
      # foreach((v,v′) -> v[t,t′] = v′, v, v′)
      return f_vert(state.u..., state.t, t1, t2)
  end

  # The adaptivity of the integration is controlled by a master cache
  @do_while timeloop!(state,caches.master,tmax,max_dt,qmax,qmin,γ,stop) begin
    @assert length(caches.vert) + 1 == length(state.t)

    # Resize solution if necessary
    if (s = last(size(last(state.u[1])))) == length(state.t)
      s += min(max(ceil(Int, (tmax - state.t[end]) / state.dt), 10), 20)
      foreach(u -> resize!.(u, s), state.u)
    end

    # Current time index
    t = length(state.t)

    # Predictor & update all times
    for (t′, cache) in enumerate([caches.vert; caches.diag])
      u_next = predict!(state, cache)
      foreach((u,u′) -> u[t,t′] = u′, state.u[1], u_next)
    end
    for (t′, _) in enumerate([caches.vert; caches.diag])
      update_time!(state.t, t, t′)
    end

    # Predictor + corrector on 1-time function
    if !isnothing(f_line) 
      u_next = predict!(state, caches.line)
      foreach((u,u′) -> u[t] = u′, state.u[2], u_next); update_line!(state.t, t)
      u_next = correct!(u_next, f_line(state.u..., state.t, t), caches.line)
      foreach((u,u′) -> u[t] = u′, state.u[2], u_next); update_line!(state.t, t)
      estimate_error!(u_next, caches.line, atol, rtol)
    end

    # Corrector and error estimation
    for (t′, cache) in enumerate([caches.vert; caches.diag])
      u_next = [x[t,t′] for x in state.u[1]] # saved in state.u[1]
      u_next = correct!(u_next, f(t,t′), cache)
      foreach((u,u′) -> u[t,t′] = u′, state.u[1], u_next); update_time!(state.t, t, t′)

      # Error estimation
      error_k = estimate_error!(u_next, cache, atol, rtol)

      # Fail step: Section III.7 Eq. (7.4)
      if error_k > one(error_k)
        caches.master = cache
        break
      end
    end

    if !isnothing(f_line)
      if caches.line.error_k > caches.master.error_k
        caches.master = caches.line
      end
    end

    # If the step is accepted
    if caches.master.error_k <= one(caches.master.error_k)
      # Update all times
      for (t′, _) in enumerate([caches.vert; caches.diag])
        update_time!(state.t, t, t′)
      end

      # Set the cache with biggest error as the master cache
      max_error = (0.0, 1)
      for (t′, cache) in enumerate([caches.vert; caches.diag])
        if cache.error_k > max_error[1]
          max_error = (cache.error_k, t′)
          caches.master = cache
        end
      end
      t′ = max_error[2]

      # Adjust the master cache's order for the next time-step
      u_next = [x[t,t′] for x in state.u[1]]
      adjust_order!(u_next, f(t,t′), state, caches.master, max_order, atol, rtol)

      # Update all caches
      update_caches!(caches, state, f_vert, f_diag, f_line)
    end
  end # timeloop!

  # Trim solution
  foreach(u -> resize!.(u, length(state.t)), state.u)
  
  return state, caches
end

function update_caches!(caches, state::VCABMState, f_vert, f_diag, f_line)
  @assert length(caches.vert) + 1 == length(state.t)

  t = length(state.t)

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

  @assert length(caches.vert) == length(state.t)
end

mutable struct KBCaches{T,F}
  vert::Vector{VCABMCache{T,F}}
  diag::VCABMCache{T,F}
  line::Union{Nothing, VCABMCache}
  master::VCABMCache # basically a reference to one of the other caches

  function KBCaches(vert1::VCABMCache{T,F}, diag, line) where {T,F}
    new{T,F}([vert1,], diag, line, diag)
  end
end

function timeloop!(state,cache,tmax,max_dt,qmax,qmin,γ,stop)
  @unpack k, error_k= cache

  if cache.error_k > one(cache.error_k)
    pop!(state.t) # remove t_prev
  end

  # II.4 Automatic Step Size Control, Eq. (4.13)
  q = max(inv(qmax), min(inv(qmin), error_k^(1/(k+1)) / γ))
  state.dt = min(state.dt / q, max_dt)

  # Don't go over tmax
  if state.t[end] + state.dt > tmax
    state.dt = tmax - state.t[end]
  end

  if stop()
    return false
  elseif state.t[end] < tmax
    push!(state.t, state.t[end] + state.dt) # add t_next
    return true
  else
    return false
  end 
end
