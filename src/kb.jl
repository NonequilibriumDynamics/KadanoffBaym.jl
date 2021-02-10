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

  state = let
    # Internal representation of `u` is (`u`, `v`)
    u = isnothing(f_line) ? (u0,) : (u0, l0)

    # Support for initial time-grid
    if isempty(size(t0))
      t0 = [t0]
    else
      @assert issorted(t0) "Initial time-grid is not in ascending order"
    end

    @assert last(t0) < tmax "Only t0 < tmax supported"

    VCABMState(u,v0,[t0; last(t0) + opts.dtini])
  end

  caches = let
    t = length(state.t)-1

    if isnothing(f_line)
      onetime = nothing
    else
      update_line(state.t, t)
      onetime = VCABMCache{eltype(t0)}(opts.kmax, [x[1] for x in state.u[2]], f_line(state.u...,state.t,t))
    end

    twotime = map(eachindex(t0)) do t′
      update_time(state.t, t, t′)
      twotime = VCABMCache{eltype(t0)}(opts.kmax, [x[t,t′] for x in state.u[1]], f_vert(state.u...,state.t,t,t′))
    end
    push!(twotime, VCABMCache{eltype(t0)}(opts.kmax, [x[t,t] for x in state.u[1]], f_diag(state.u...,state.t,t)))

    KBCaches(twotime, onetime)
  end

  kbsolve_(f_vert, f_diag, tmax, state, caches, opts, update_time, 
    f_line, update_line, kernel_vert, kernel_diag)
end

function kbsolve_(f_vert, f_diag, tmax, state, caches, opts, update_time, 
  f_line, update_line, kernel_vert, kernel_diag)

  function update_onetime!(u_next,t1)
    foreach((u,u′) -> u[t1] = u′, state.u[2], u_next)
    update_line(state.t, t1)
  end

  function update_twotime!(u_next,t1,t2)
    foreach((u,u′) -> u[t1,t2] = u′, state.u[1], u_next)
    update_time(state.t, t1, t2)
  end
  
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
  f(t1,t2) = isequal(t1, t2) ? f_diag(state.u..., state.t, t1) : f_vert(state.u..., state.t, t1, t2)
  f(t1) = f_line(state.u..., state.t, t1)
  
  while timeloop!(state,caches,tmax,opts)
    # Current time index
    t = length(state.t)

    @assert length(caches.twotime) == t

    # Predictor for the 2-time functions
    for (t′, cache) in enumerate(caches.twotime)
      u_next = predict!(state, cache)
      update_twotime!(u_next, t, t′)
    end

    # Predictor, corrector and error estimation for the 1-time functions
    if !isnothing(f_line) 
      u_next = predict!(state, caches.onetime)
      update_onetime!(u_next, t)

      u_next = correct!(f(t), caches.onetime)
      update_onetime!(u_next, t)

      estimate_error(u_next, caches.onetime, opts.atol, opts.rtol)
    end

    # Corrector and error estimation for the 2-time functions
    if !isnothing(f_line) && caches.onetime.error_k > one(caches.onetime.error_k)
      caches.master = caches.onetime
    else
      for (t′, cache) in enumerate(caches.twotime)
        u_next = correct!(f(t,t′), cache)
        update_twotime!(u_next, t, t′)
      end
    end

    caches.error_k = estimate_error.(caches.twotime, caches.k, opts.atol, opts.rtol) |> norm
    # If the step is accepted
    if caches.error_k <= one(caches.error_k)

      f_next = map(t′ -> f(t, t′), eachindex(caches.twotime))

      # Control order: Section III.7 Eq. (7.7)
      if t<=5 || caches.k<3
        caches.k = min(3, caches.k+1, opts.kmax)
      else
        error_k1 = estimate_error.(caches.twotime, caches.k-1, opts.atol, opts.rtol) |> norm
        error_k2 = estimate_error.(caches.twotime, caches.k-2, opts.atol, opts.rtol) |> norm
        if max(error_k2, error_k1) <= caches.error_k
          caches.k = caches.k-1
        else
          foreach(i -> ϕ_np1!(caches.twotime[i], f_next[i], caches.k+2), eachindex(caches.twotime))
          error_kstar = estimate_error.(caches.twotime, caches.k, opts.atol, opts.rtol, state.t[end] - state.t[end-1]) |> norm
          if error_kstar < caches.error_k
            caches.k = min(caches.k+1, opts.kmax)
            caches.error_k = one(caches.error_k)   # constant dt
          end
        end
      end

      # Update the 2-time caches
      for (t′, cache) in enumerate(caches.twotime)
        update_cache!(f_next[t′], cache, caches.k)
      end

      # Update the 1-time cache
      if !isnothing(f_line)
        update_cache!(f(t), caches.onetime, caches.k)
      end

      # Add a new cache for the next time
      begin
        times = state.t

        t0 = max(t - caches.k, 1)
        u0 = [x[t0,t] for x in state.u[1]]
        f0 = f_vert(state.u..., state.t, t0, t)
        cache = VCABMCache{eltype(state.t)}(opts.kmax, u0, f0)

        for t′ in (t0+1):t
          state.t = view(times,1:t′); ϕ_and_ϕstar!(state, cache, cache.k+1)
          cache.u_next = [x[t′,t] for x in state.u[1]]
          update_cache!(f_vert(state.u..., times, t′, t), cache, caches.k)
        end

        state.t = times
        insert!(caches.twotime, t, cache)
        @assert length(caches.twotime) == length(state.t) + 1
      end
    end
  end # timeloop!
  
  return state, caches
end

function timeloop!(state,cache,tmax,opts)
  @unpack k, error_k = cache

  # II.4 Automatic Step Size Control, Eq. (4.13)
  q = max(inv(opts.qmax), min(inv(opts.qmin), error_k^(1/(k+1)) / opts.γ))
  dt = min((state.t[end] - state.t[end-1]) / q, opts.dtmax)

  # Remove t_prev if last step failed
  if error_k > one(error_k)
    pop!(state.t) 
  end

  # Don't go over tmax
  if last(state.t) + dt > tmax
    dt = tmax - last(state.t)
  end

  # Reached the end of the integration
  if iszero(dt) || opts.stop()
    # Trim solution
    foreach(u -> resize!.(u, length(state.t)), state.u)

    return false
  end

  # Resize solution if necessary
  if (t = length(state.t)) == last(size(first(first(state.u))))
    foreach(u -> resize!.(u, t + min(50, ceil(Int, (tmax - state.t[end]) / dt))), state.u)
  end

  return (push!(state.t, last(state.t) + dt); true)
end

mutable struct KBCaches{T,F}
  twotime::Vector{VCABMCache{T,F}}
  onetime::Union{Nothing, VCABMCache}
  error_k::Float64
  k::Int64

  KBCaches(tt::Vector{VCABMCache{T,F}}, ot) where {T,F} = new{T,F}(tt,ot,Inf,1)
end
