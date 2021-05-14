"""
    kbsolve(f_vert, f_diag, u0, (t0, tmax); ...)

Solves the 2-time (Voltera integral) differential equation

``du/dt1 = f(u,t1,t2) + ∫dτ K[u,t1,t2,τ]``

``du/dt2 = f'(u,t1,t2) + ∫dτ K'[u,t1,t2,τ]``

for some initial condition `u0` from `t0` to `tmax`.

# Parameters
  - `f_vert(u, ts, t1, t2)` of the differential equation `du/dt1`
  - `f_diag(u, ts, t1)` of the differential equation `du/dt1 + du/dt2` at `t1=t2`
  - `u0::Vector{<:GreenFunction}`: initial condition for the 2-point functions
  - `(t0, tmax)`: the initial and final time

# Optional keyword parameters
  - `callback(ts, t1)`: A function that gets called everytime the 2-point function values at slice `(t1, 1:t1)` are updated
  - `f_line(u, ts, t1)`: the right-hand-side for 1-point functions
  - `l0`: initial condition for the 1-point functions
  - `kernel_vert(ts, t1, t2, τ)`: the integral kernel of `du/dt1`
  - `kernel_diag(ts, t1, τ)`: the integral kernel of `du/dt1 + du/dt2`
  - `kwargs...`: see `VCABMOptions`

# Notes
  - Unlike standard ODE solvers, `kbsolve` is designed to mutate the initial conditions
  - The Kadanoff-Baym timestepper is a 2-time generalization of the VCABM stepper 
    presented in Ernst Hairer, Gerhard Wanner, and Syvert P Norsett
    Solving Ordinary Differential Equations I: Nonstiff Problems
"""
function kbsolve(f_vert, f_diag, u0::Vector{<:GreenFunction}, (t0, tmax);
  k_vert=nothing, k_diag=nothing, l0=nothing, f_line=nothing,
  callback=(x...)->nothing, kwargs...)
  
  opts = VCABMOptions(; kwargs...)  

  # Support for initial time-grid
  if isempty(size(t0))
    t0 = [t0]
  else
    @assert issorted(t0) "Initial time-grid is not in ascending order"
  end
  @assert last(t0) < tmax "Only t0 < tmax supported"

  state = begin
    if isnothing(k_vert)
      KBState(u0, nothing, t0)
    else
      v0 = zero.(u0)
      KBState([u0; v0], v0, t0)
    end
  end

  if isnothing(k_vert)
    fv = (t,t′) -> f_vert(state.u,state.t,t,t′)
    f = t -> VectorOfArray([[fv(t,t′) for t′ in 1:t-1]; [f_diag(state.u,state.t,t),]])
  else
    kernel = (t,s) -> VectorOfArray([[k_vert(state.u,state.t,t,t′,s) for t′ in 1:t-1]; [k_diag(state.u,state.t,t,s),]])
    fv = (t,t′) -> [f_vert(state.u,state.t,t,t′); k_vert(state.u,state.t,t,t′,t)]
    f = t -> VectorOfArray([[fv(t,t′) for t′ in 1:t-1]; [[f_diag(state.u,state.t,t); k_diag(state.u,state.t,t,t)],]])

    t = length(state.t)
    cache_volterra = VCABMCacheVolterra{eltype(state.t)}(opts.kmax, VectorOfArray([[v[t,t′] for v in state.v] for t′ in 1:t]))
  end

  cache = let
    t = length(state.t)
    u = VectorOfArray([[x[t,t′] for x in state.u] for t′ in 1:t])
    VCABMCache{eltype(state.t)}(opts.kmax, u, typeof(u)(f(t).u))
  end

  # All mutations to user arguments are done explicitely here
  while timeloop!(state,cache,tmax,opts)
    t = length(state.t)

    # Resize solution
    if t > (last ∘ size ∘ first)(state.u)
      foreach(u -> resize!(u, t + min(50, ceil(Int, (tmax - state.t[end]) / (state.t[end] - state.t[end-1])))), state.u)
    end

    # Extend caches
    extend!(cache, fv, state.t)
    if !isnothing(k_vert)
      extend!(cache_volterra, state.t, (t,t′,s) -> k_vert(state.u, state.t, t, t′, s), (t,s) -> k_diag(state.u, state.t, t, s), cache)
    end

    # Predictor
    u_next = predict!(cache, state.t)
    foreach((u, u′) -> foreach(t′ -> u[t,t′] = u′[t′], 1:t), state.u, eachrow(u_next))
    callback(state.t, t)
    if !isnothing(k_vert)
      v_next = predict!(cache_volterra, state.t, VectorOfArray([[v[t,t′] for v in state.v] for t′ in 1:t]))
      foreach((v, v′) -> foreach(t′ -> v[t,t′] = v′[t′], 1:t), state.v, eachrow(v_next))
    end

    # Corrector
    u_next = correct!(cache, () -> f(t))
    foreach((u, u′) -> foreach(t′ -> u[t,t′] = u′[t′], 1:t), state.u, eachrow(u_next))
    callback(state.t, t)
    if !isnothing(k_vert)
      v_next = correct!(cache_volterra, state.t, VectorOfArray([[v[t,t′] for v in state.v] for t′ in 1:t]))
      foreach((v, v′) -> foreach(t′ -> v[t,t′] = v′[t′], 1:t), state.v, eachrow(v_next))
    end

    # Calculate error and adjust order
    adjust!(cache, state.t, () -> f(t), opts.kmax, opts.atol, opts.rtol)
  end # timeloop!
  
  foreach(u -> resize!(u, length(state.t)), state.u) # trim solution
  return state
end

function timeloop!(state,cache,tmax,opts)
  if isone(cache.k) # first step
    dt = iszero(opts.dtini) ? initial_step(cache.f_prev, cache.u_prev, opts.atol, opts.rtol) : opts.dtini
    push!(state.t, last(state.t) + dt)
    return true
  end

  # II.4 Automatic Step Size Control, Equation (4.13)
  q = max(inv(opts.qmax), min(inv(opts.qmin), cache.error_k^(1/(cache.k+1)) / opts.γ))
  dt = min((state.t[end] - state.t[end-1]) / q, opts.dtmax)

  # Remove t_prev if last step failed
  if cache.error_k > one(cache.error_k)
    pop!(state.t) 
  end

  # Don't go over tmax
  if last(state.t) + dt > tmax
    dt = tmax - last(state.t)
  end

  # Reached the end of the integration
  if iszero(dt) || opts.stop()
    return false
  else
    push!(state.t, last(state.t) + dt)
    return true
  end
end

# Holds the information about the integration
struct KBState{U,V,T}
  u::U          # 2-point functions
  v::V          # 2-point Volterra integrals
  t::Vector{T}  # Time grid
end
