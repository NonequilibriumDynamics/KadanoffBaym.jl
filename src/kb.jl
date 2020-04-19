"""
  Kadanoff-Baym adaptive timestepper

Solves the 2-time Voltera integral differential equations
  dy/dt1 = F[y](t1,t2)
  dy/dt2 = G[y](t1,t2)

The signature of `f_vert` and `f_diag` must be of the form
  f_vert(u, t_grid, t1, t2)
  f_diag(u, t_grid, t1)
And if mixed functions are present
  f_vert(u, v, t_grid, t1, t2)
  f_diag(u, v, t_grid, t1)
  f_line(u, v, t_grid, t1)

The initial condition `u` should be an array or tuple of `uᵢⱼ`s and
if mixed functions `vᵢ` are present, should be of the form (`uᵢⱼ`, `vᵢ`).

It is also mandatory that both `u`s and can `v`s can be **indexed** by 2-time 
and 1-time arguments, respectively, and can be resized! at will
  uᵢⱼ = resize!(u, new_size)
  vᵢ = resize(v, new_size)

The Kadanoff-Baym timestepper is based on the VCABM stepper presented in
Ernst Hairer, Gerhard Wanner, and Syvert P Norsett
Solving Ordinary Differential Equations I: Nonstiff Problems
"""
function kbsolve(f_vert, f_diag, u, t₀, tmax; f_line=nothing, dt=nothing, adaptive=true, 
  max_dt=1e-1, atol=1e-9, rtol=1e-7, max_order=12, qmax=5, qmin=1//5, γ=9//10)
  
  if max_order < 1 || max_order > 12
    error("max_order must be between 1 and 12")
  end

  if t₀ > tmax
    error("Only t₀ < tmax supported")
  end

  if isnothing(f_line) u = (u,) end

  # Initialize state and caches
  state, caches = begin
    # Resize solutions
    foreach(u′->resize!.(u′, 50), u)

    if isnothing(f_line)
      cache_line = nothing
    else
      u₀ = [x[1] for x in u[2]] # Extract solution at (t₀)
      cache_line = VCABMCache{typeof(t₀)}(max_order,u₀,f_line(u...,[t₀],1))
    end

    u₀ = [x[1,1] for x in u[1]] # Extract solution at (t₀,t₀)
    cache_vert = VCABMCache{typeof(t₀)}(max_order,u₀,f_vert(u...,[t₀],1,1))
    cache_diag = VCABMCache{typeof(t₀)}(max_order,u₀,f_diag(u...,[t₀],1))

    # Calculate initial dt
    if dt===nothing
      f′ = (u′,t′) -> begin
        foreach(i->u[1][i][2,1]=u′[i],eachindex(u′))
        f_vert(u...,[t₀,t′],2,1)
      end
      dt = initial_step(f′,u₀,t₀,1,rtol,atol;f₀=cache_vert.f_prev)
    end

    VCABMState(u,t₀,dt), KBCaches(cache_vert,cache_diag,cache_line)    
  end

  # Modify user functions to mutate the `state` variable `u` and refer to `t`
  # This is necessary because internally `f` is a function of `uᵢ` and `tᵢ`
  f_diag!(tᵢ) = (u′,_) -> begin
    foreach(i->state.u[1][i][tᵢ,tᵢ]=u′[i], eachindex(u′))
    f_diag(state.u...,state.t,tᵢ)
  end

  # Time-step
  @do_while timeloop!(state,caches.diag,tmax,max_dt,adaptive,qmax,qmin,γ) begin
    T = length(state.t) # current time index

    # Step vertically
    if !isnothing(f_line)
      u_next = predict!(state, caches.line)
      foreach(i->state.u[2][i][T]=u_next[i], eachindex(u_next))
    end
    for (tⱼ, cache) in enumerate(caches.vert)
      u_next = predict!(state, cache)
      foreach(i->state.u[1][i][T,tⱼ]=u_next[i], eachindex(u_next))
    end
    if !isnothing(f_line)
      u_next = [x[T] for x in state.u[2]]
      u_next = correct!(u_next, f_line(state.u...,state.t,T), caches.line)
      foreach(i->state.u[2][i][T]=u_next[i], eachindex(u_next))
    end
    for (tⱼ, cache) in enumerate(caches.vert)
      u_next = [x[T,tⱼ] for x in state.u[1]]
      u_next = correct!(u_next, f_vert(state.u...,state.t,T,tⱼ), cache)
      foreach(i->state.u[1][i][T,tⱼ]=u_next[i], eachindex(u_next))
    end

    # Step diagonally and control step
    predict_correct!(f_diag!(T),state,caches.diag,max_order,atol,rtol,true)

    # Accept step
    if caches.diag.error_k <= one(caches.diag.error_k)
      update_caches!(caches,state,f_vert,f_line,max_order)
    end

    # Resize solution
    if (s = size(state.u[1][1],1)) == length(state.t)
      s += min(max(ceil(Int,(tmax-state.t[end])/state.dt),5),50)
      foreach(u′->resize!.(u′,s), state.u)
    end
  end # timeloop!

  # Trim solution
  foreach(u′->resize!.(u′, length(state.t)), state.u)
  
  return isnothing(f_line) ? (state.u[1], state.t) : (state.u, state.t)
end

function update_caches!(caches, state::VCABMState, f_vert, f_line, max_k)
  @assert length(caches.vert)+1 == length(state.t)

  T = length(state.t)
  local_max_k = caches.diag.k

  # Update all vertical caches (does the remaining steps of predict_correct!)
  for (tⱼ, cache) in enumerate(caches.vert)
    cache.u_prev = [x[T,tⱼ] for x in state.u[1]] # u_next was saved in state.u by update
    cache.f_prev = f_vert(state.u...,state.t,T,tⱼ)
    cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1
    cache.k = min(local_max_k, cache.k+1) # Ramp up order
  end

  # Update the mixed cache (does the remaining steps of predict_correct!)
  if !isnothing(f_line)
    cache = caches.line
    cache.u_prev = [x[T] for x in state.u[2]] # u_next was saved in state.u by update
    cache.f_prev = f_line(state.u...,state.t,T)
    cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1
    cache.k = min(local_max_k, cache.k+1) # Ramp up order
  end

  # Add a new cache for the next time line!
  begin
    u₀ = [x[T,T] for x in state.u[1]]
    f₀ = f_vert(state.u...,state.t,T,T)
    push!(caches.vert, VCABMCache{eltype(state.t)}(max_k, u₀, f₀))
  end

  @assert length(caches.vert) == length(state.t)
end

mutable struct KBCaches{T,F}
  vert::Vector{VCABMCache{T,F}}
  diag::VCABMCache{T,F}
  line::Union{Nothing, VCABMCache}

  function KBCaches(vert1::VCABMCache{T,F}, diag, line) where {T,F}
    new{T,F}([vert1,], diag, line)
  end
end