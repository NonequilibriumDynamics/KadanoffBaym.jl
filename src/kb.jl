"""
  kbsolve(f_vert, f_diag, u0, (t0, tmax); ...)

Solves the 2-time (Voltera integral) differential equations

  du/dt1 = f(u,t1,t2)

  du/dt2 = g(u,t1,t2)

for some initial condition `u0` from `t0` to `tmax`.
# Parameters
  - `f_vert(u, ts, t1, t2)` of the differential equation `du/dt1`
  - `f_diag(u, ts, t1)` of the differential equation `d/dt1 + d/dt2`
  - `u0::Vector{<:GreenFunction}`: initial condition for the 2-point functions
  - `(t0, tmax)`: the initial and final time

# Optional keyword parameters
  - `update_time(ts, t1, t2)`: A function that gets called everytime the 2-point function values are updated by the stepper

  - `f_line(u, ts, t1)`:
  - `l0`: initial condition for the 1-point functions
  - `update_line(ts, t1)`:

  - `kernel_vert`: the integral kernel of `du/dt1`
  - `kernel_diag`: the integral kernel of `d/dt1 + d/dt2`
  - `v0`: initial condition for the 2-time volterra integrals
  
  - `kwargs...`: see `VCABMOptions`

# Notes
  - Unlike standard ODE solvers, `kbsolve` is designed to mutate the initial 
    conditions
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

  cache = let
    t = length(t0)

    VCABMCache{eltype(t0)}(opts.kmax, 
      push!(map(t′ -> [x[t,t′] for x in u0], 1:t), [x[t,t] for x in u0]),
      push!(map(t′ -> f_vert(u0,t0,t,t′), 1:t), f_diag(u0,t0,t)))
  end
  state = KBState(u0, v0, [t0; last(t0) + opts.dtini])

  while timeloop!(state,cache,tmax,opts)
    t = length(state.t)

    f(t′) = isequal(t,t′) ? f_diag(state.u, state.t, t) : f_vert(state.u, state.t, t, t′)

    # Predictor
    u_next = predict!(state.t, cache)
    for t′ in 1:t
      foreach((u,u′) -> u[t,t′] = u′, state.u, u_next[t′])
      update_time(state.t, t, t′)
    end

    # Corrector
    u_next = correct!((f(t′) for t′ in 1:t), cache)
    for t′ in 1:t
      foreach((u,u′) -> u[t,t′] = u′, state.u, u_next[t′])
      update_time(state.t, t, t′)
    end

    # Calculate error and, if the step is accepted, adjust order and add a new cache entry
    adjust_order!(t′ -> f_vert(state.u, state.t, t′, t), (f(t′) for t′ in 1:t), state, cache, opts.kmax, opts.atol, opts.rtol)
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
mutable struct KBState{U,V,T}
  u::U
  v::V
  t::Vector{T}
end
