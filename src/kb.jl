"""
    kbsolve!(fv!, fd!, u0, (t0, tmax); ...)

Solves the 2-time Voltera integro-differential equation

``du/dt1 = fv = v(u,t1,t2) + ∫_{t0]^{t1} dτ K1v[u,t1,t2,τ] + ∫_{t0}^{t2} dτ K2v[u,t1,t2,τ]``

``du/dt2 = fh = h(u,t1,t2) + ∫_{t0}^{t1} dτ K1h[u,t1,t2,τ] + ∫_{t0}^{t2} dτ K2h[u,t1,t2,τ]``

for some initial condition `u0` from `t0` to `tmax`.

# Parameters
  - `fv!(out, ts, w1, w2, t1, t2)`: the rhs of `du/dt1` at indices (`t1`, `t2`) 
    in the time-grid `ts` x `ts`. The weights `w1` and `w2` can be used to integrate
    the Volterra kernels `K1` and `K2` as `sum_i w1_i K1_i` and  `sum_i w2_i K2_i`,
    respectively. The output should be saved in-place in `out`, which has the 
    same shape as `u0`
  - `fd!(out, ts, w1, w2, t1, t2)`: the rhs of `du/dt1 + du/dt2`
  - `u0::Vector{<:GreenFunction}`: list of 2-point functions to be time-stepped
  - `(t0, tmax)`: a tuple with the initial time(s) `t0` – can be a vector of past times – and final time `tmax`

# Optional keyword parameters
  - `callback(ts, w1, w2, t1, t2)`: A function that gets called everytime the 2-point function at the indices (`t1`, `t2`) is updated
  - `stop(ts)`: A function that gets called at every step that stops the integration when evaluates to `true`
  - `atol::Real`: Absolute tolerance (components with magnitude lower than `atol` do not guarantee number of local correct digits)
  - `rtol::Real`: Relative tolerance (roughly the local number of correct digits)
  - `dtini::Real`: Initial step-size
  - `dtmax::Real`: Maximal step-size
  - `qmax::Real`: Maximum step-size factor when adjusting the time-step
  - `qmin::Real`: Minimum step-size factor when adjusting the time-step
  - `γ::Real`: Safety factor for the calculated time-step such that it is accepted with a higher probability
  - `kmax::Integer`: Maximum order of the adaptive Adams method

# Notes
  - Due to high memory and computation costs, `kbsolve!` mutates the initial condition `u0` 
    and only works with in-place rhs functions, unlike standard ODE solvers.
  - The Kadanoff-Baym timestepper is a 2-time generalization of the variable Adams method
    presented in Ernst Hairer, Gerhard Wanner, and Syvert P Norsett
    Solving Ordinary Differential Equations I: Nonstiff Problems
"""
function kbsolve!(fv!, fd!, u0::Vector{<:GreenFunction}, (t0, tmax); 
                  callback=(x...)->nothing, stop=(x...)->false,
                  atol=1e-8, rtol=1e-6, dtini=0.0, dtmax=Inf, qmax=5, qmin=1 // 5, γ=9 // 10, kmax=12)

  # Support for an initial time-grid
  if isempty(size(t0))
    t0 = [t0]
  else
    @assert issorted(t0) "Initial time-grid is not in ascending order"
  end
  @assert last(t0) <= tmax "Only t0 <= tmax supported"

  # Holds the information about the integration
  state = (u=u0, t=t0, w=VolterraWeights(t0))

  # Holds the information necessary to integrate
  cache = let
    t1 = length(state.t)
    VCABMCache{eltype(state.t)}(kmax, VectorOfArray([[u[t1, t2] for t2 in 1:t1] for u in state.u]))
  end

  # The rhs is seen as a univariate problem
  function f!(t1)
    Threads.@threads for t2 in 1:(t1-1)
      fv!(view(cache.f_next, t2, :), state.t, state.w.ws[t1], state.w.ws[t2], t1, t2)
    end
    fd!(view(cache.f_next, t1, :), state.t, state.w.ws[t1], state.w.ws[t1], t1, t1)
    return cache.f_next
  end

  cache.f_prev .= f!(length(state.t))

  while timeloop!(state, cache, tmax, dtmax, dtini, atol, rtol, qmax, qmin, γ, stop)
    t1 = length(state.t)

    # Extend the caches to accomodate the new time column
    extend!(cache, state, (t1, t2) -> fv!(view(cache.f_next, t2, :), state.t, state.w.ws[t1], state.w.ws[t2], t1, t2))

    # Predictor
    u_next = predict!(cache, state.t)
    foreach((u, u′) -> foreach(t2 -> u[t1, t2] = u′[t2], 1:t1), state.u, u_next.u)
    foreach(t2 -> callback(state.t, state.w.ws[t1], state.w.ws[t2], t1, t2), 1:t1)

    # Corrector
    u_next = correct!(cache, () -> f!(t1))
    foreach((u, u′) -> foreach(t2 -> u[t1, t2] = u′[t2], 1:t1), state.u, u_next.u)
    foreach(t2 -> callback(state.t, state.w.ws[t1], state.w.ws[t2], t1, t2), 1:t1)

    # Calculate error and adjust order
    adjust!(cache, state.t, () -> f!(t1), kmax, atol, rtol)
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

  # Remove the last element of the time-grid / weights if last step failed
  if cache.error_k > one(cache.error_k)
    pop!(state.t)
    pop!(state.w.ks)
    pop!(state.w.ws)
  end

  # Don't go over tmax
  if last(state.t) + dt > tmax
    dt = tmax - last(state.t)
  end

  # Reached the end of the integration
  if iszero(dt) || stop(state.t)
    foreach(u -> resize!(u, length(state.t)), state.u) # trim solution
    return false
  else
    if length(state.t) == (last ∘ size ∘ last)(state.u) # resize solution
      foreach(u -> resize!(u, length(state.t) + min(50, ceil(Int, (tmax - state.t[end]) / dt))), state.u)
    end
    push!(state.t, last(state.t) + dt)
    push!(state.w.ks, cache.k)
    push!(state.w.ws, calculate_weights(state.t, state.w.ks, atol, rtol))
    return true
  end
end
