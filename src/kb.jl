"""
    kbsolve(f_vert, f_diag, u0, (t0, tmax); ...)

Solves the 2-time Voltera integro-differential equation

``du/dt1 = f(u,t1,t2) + ∫dτ K[u,t1,t2,τ]``

``du/dt2 = f'(u,t1,t2) + ∫dτ K'[u,t1,t2,τ]``

for some initial condition `u0` from `t0` to `tmax`.

# Parameters
  - `f_vert(u, ts, t1, t2)`: the rhs of `du/dt1`
  - `f_diag(u, ts, t1)`: the rhs of `du/dt1 + du/dt2` at `t1 = t2`
  - `u0::Vector{<:GreenFunction}`: initial condition for the 2-point functions
  - `(t0, tmax)`: the initial time (can be an array) and final time

# Optional keyword parameters
  - `k_vert(u, ts, t1, t2, τ)`: the integral kernel of `du/dt1`
  - `k_diag(u, ts, t1, τ)`: the integral kernel of `du/dt1 + du/dt2` at `t1 = t2`
  - `callback(ts, t1, t2)`: A function that gets called everytime the 2-point function at the indices (t1, t2) is updated
  - `stop(u, ts)`: A function that gets called at every step that when evaluates to `true` stops the integration

  For two approximations of the solution, the local error of the less precise is given by |y1 - y1'| < atol + rtol * max(y0,y1)
  - `atol::Real`: Absolute tolerance (components with magnitude lower than `atol` do not guarantee number of local correct digits)
  - `rtol::Real`: Relative tolerance (roughly the local number of correct digits)
  - `dtini::Real`: Initial step-size
  - `dtmax::Real`: Maximal step-size
  - `qmax::Real`: Maximal step-size increase
  - `qmin::Real`: Minimum step-size decrease
  - `γ::Real`: Safety factor so that the error will be acceptable the next time with high probability
  - `kmax::Integer`: Maximum order of the adaptive Adams method

# Notes
  - Unlike standard ODE solvers, `kbsolve` is designed to mutate the initial conditions
  - The Kadanoff-Baym timestepper is a 2-time generalization of the variable Adams method
    presented in Ernst Hairer, Gerhard Wanner, and Syvert P Norsett
    Solving Ordinary Differential Equations I: Nonstiff Problems
"""
function kbsolve(f_vert, f_diag, u0::Vector{<:GreenFunction}, (t0, tmax); 
  k_vert=nothing, k_diag=nothing, callback=(x...) -> nothing, stop=(x...) -> false,
  atol=1e-8, rtol=1e-6, dtini=0.0, dtmax=Inf, qmax=5, qmin=1 // 5, γ=9 // 10, kmax=12)

  # Support for an initial time-grid
  if isempty(size(t0))
    t0 = [t0]
  else
    @assert issorted(t0) "Initial time-grid is not in ascending order"
  end
  @assert last(t0) < tmax "Only t0 < tmax supported"

  # Holds the information about the integration
  state = KBState(u0, t0)

  # Holds the information necessary to integrate
  caches = let
    t = length(state.t)
    VCABMCache{eltype(state.t)}(kmax, VectorOfArray([u[t, 1:t] for u in state.u])), nothing
  end

  function f(t)
    out = VectorOfArray([[f_vert(state.u, state.t, t, t′) for t′ in 1:(t - 1)]; [f_diag(state.u, state.t, t)]])
    return VectorOfArray([collect(x) for x in eachrow(out)])
  end

  caches[1].f_prev .= f(1)

  while timeloop!(state, caches[1], tmax, dtmax, dtini, atol, rtol, qmax, qmin, γ, stop)
    t = length(state.t)

    # Extend the caches to accomodate the new time column
    extend!(caches, state.t, (t, t′) -> f_vert(state.u, state.t, t, t′))

    # Predictor
    u_next = predict!(caches[1], state.t)
    foreach((u, u′) -> foreach(t′ -> u[t, t′] = u′[t′], 1:t), state.u, u_next.u)
    foreach(t′ -> callback(state.t, t, t′), 1:t)

    # Corrector
    u_next = correct!(caches[1], () -> f(t))
    foreach((u, u′) -> foreach(t′ -> u[t, t′] = u′[t′], 1:t), state.u, u_next.u)
    foreach(t′ -> callback(state.t, t, t′), 1:t)

    # Calculate error and adjust order
    adjust!(caches[1], state.t, () -> f(t), kmax, atol, rtol)
  end # timeloop!
  return state
end

# Controls the step size & resizes the Green functions if required
function timeloop!(state, cache, tmax, dtmax, dtini, atol, rtol, qmax, qmin, γ, stop)
  if isone(length(state.t))
    # Section II.4: Starting Step Size, Eq. (4.14)
    dt = iszero(dtini) ? initial_step(cache.f_prev, cache.u_prev, atol, rtol) : dtini
  else
    # Section II.4: Automatic Step Size Control, Eq. (4.13)
    q = max(inv(qmax), min(inv(qmin), cache.error_k^(1 / (cache.k + 1)) / γ))
    dt = min((state.t[end] - state.t[end - 1]) / q, dtmax)
  end

  # Remove the last element of the time-grid if last step failed
  if cache.error_k > one(cache.error_k)
    pop!(state.t)
  end

  # Don't go over tmax
  if last(state.t) + dt > tmax
    dt = tmax - last(state.t)
  end

  # Reached the end of the integration
  if iszero(dt) || stop(state.u, state.t)
    foreach(u -> resize!(u, length(state.t)), state.u) # trim solution
    return false
  else
    if length(state.t) == (last ∘ size ∘ last)(state.u) # resize solution
      foreach(u -> resize!(u, length(state.t) + min(50, ceil(Int, (tmax - state.t[end]) / dt))), state.u)
    end
    push!(state.t, last(state.t) + dt)
    return true
  end
end

struct KBState{U,T}
  u::U          # 2-point functions
  t::Vector{T}  # time-grid
end
