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
  else
    @assert issorted(t0) "Initial time-grid is not in ascending order"
  end
  @assert last(t0) < tmax "Only t0 < tmax supported"

  state = let
    cache = begin
      t = length(t0)

      u0_ = map(t′ -> [x[t,t′] for x in u0], eachindex(t0))
      u0_ = push!(u0_, [x[t,t] for x in u0])

      f0 = map(t′ -> f_vert(u0,t0,t,t′), eachindex(t0))
      f0 = push!(f0, f_diag(u0,t0,t))

      VCABMCache{eltype(t0)}(opts.kmax, VectorOfArray(u0_), VectorOfArray(f0))
    end

    KBState(u0,v0, [t0; last(t0) + opts.dtini], cache)
  end

  function f(t1,t2)
    if isequal(t1, t2)
      f_diag(state.u, state.t, t1)
    else
      f_vert(state.u, state.t, t1, t2)
    end
  end
  function update_twotime!(u_next,t1,t2)
    foreach((u,u′) -> u[t1,t2] = u′, state.u, u_next)
    update_time(state.t, t1, t2)
  end
  
  while timeloop!(state,state.u_cache,tmax,opts)
    # Current time index
    t = length(state.t)

    # Predictor
    u_next = predict!(state, state.u_cache)
    foreach(t′ -> update_twotime!(u_next[t′],t,t′), 1:t)

    # Corrector NOTE: u[t,t′] is *corrected* before valuating t′+1 [implicit!]
    for t′ in 1:t
      u_next = correct!(f(t,t′), state.u_cache, t′)
      update_twotime!(u_next, t, t′)
    end

    # Calculate error and adjust order
    adjust_order!((f(t, t′) for t′ in 1:t), state, state.u_cache, opts.kmax, opts.atol, opts.rtol)
    
    # Add a new cache for the next time if the step is accepted
    if state.u_cache.error_k <= one(state.u_cache.error_k)
      extend_cache!(t′ -> f_vert(state.u, state.t, t′, t), state, opts.kmax)
    end
  end # timeloop!
  
  return state
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
    foreach(u -> resize!(u, length(state.t)), state.u) # trim solution
    return false
  end

  # Resize solution if necessary
  if (t = length(state.t)) == last(size(first(state.u)))
    foreach(u -> resize!(u, t + min(50, ceil(Int, (tmax - state.t[end]) / dt))), state.u)
  end

  return (push!(state.t, last(state.t) + dt); true)
end

# Holds the information about the integration
mutable struct KBState{T,U,V,F}
  u::U
  v::V
  t::Vector{T}

  u_cache::VCABMCache{T,F}

  function KBState(u::U, v::V, t::Vector{T}, cache::VCABMCache{T,F}) where {T,U,V,F}
    new{T,U,V,F}(u, v, t, cache)
  end
end
