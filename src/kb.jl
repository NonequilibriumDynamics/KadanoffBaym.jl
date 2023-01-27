"""
    kbsolve!(fv!, fd!, u0, (t0, tmax); ...)

Solves the 2-time Voltera integro-differential equation

``
du(t_1,t_2) / dt_1 = f_v(t_1,t_2) = v[u,t_1,t_2] + ∫_{t0}^{t1} dτ K_1^v[u,t_1,t_2,τ] + ∫_{t0}^{t2} dτ K_2^v[u,t_1,t_2,τ]
``

``
du(t_1,t_2) / dt_2 = f_h(t_1,t_2) = h[u,t_1,t_2] + ∫_{t0}^{t1} dτ K_1^h[u,t_1,t_2,τ] + ∫_{t0}^{t2} dτ K_2^h[u,t_1,t_2,τ]
``

for 2-point functions `u0` from `t0` to `tmax`.

# Parameters
  - `fv!(out, ts, w1, w2, t1, t2)`: The right-hand side of ``du(t_1,t_2)/dt_1`` 
    on the time-grid (`ts` x `ts`). The weights `w1` and `w2` can be used to integrate
    the Volterra kernels `K1v` and `K2v` as `sum_i w1_i K1v_i` and  `sum_i w2_i K2v_i`,
    respectively. The output is saved in-place in `out`, which has the same shape as `u0`
  - `fd!(out, ts, w1, w2, t1, t2)`: The right-hand side of ``(du(t_1,t_2)/dt_1 + du(t_1,t_2)/dt_2)|_{t_2 → t_1}``
  - `u0::Vector{<:GreenFunction}`: List of 2-point functions to be integrated
  - `(t0, tmax)`: A tuple with the initial time(s) `t0` – can be a vector of 
    past times – and final time `tmax`

# Optional keyword parameters
  - `f1!(out, ts, w1, t1)`: The right-hand-side of ``dv(t_1)/dt_1``. The weight `w1`
    can be used to integrate the Volterra kernel and the output is saved in-place in `out`, 
    which has the same shape as `v0`
  - `v0::Vector`: List of 1-point functions to be integrated
  - `callback(ts, w1, w2, t1, t2)`: A function that gets called everytime the 
    2-point function at *indices* (`t1`, `t2`) is updated. Can be used to update
    functions which are not being integrated, such as self-energies
  - `stop(ts)`: A function that gets called at every time-step that stops the 
    integration when it evaluates to `true`
  - `atol::Real`: Absolute tolerance (components with magnitude lower than 
    `atol` do not guarantee number of local correct digits)
  - `rtol::Real`: Relative tolerance (roughly the local number of correct digits)
  - `dtini::Real`: Initial step-size
  - `dtmax::Real`: Maximal step-size
  - `qmax::Real`: Maximum step-size factor when adjusting the time-step
  - `qmin::Real`: Minimum step-size factor when adjusting the time-step
  - `γ::Real`: Safety factor for the calculated time-step such that it is 
    accepted with a higher probability
  - `kmax::Integer`: Maximum order of the adaptive Adams method
  - `kmax_vie::Integer`: Maximum order of interpolant of the Volterra integrals
    Heuristically, it seems that having too high of a `kmax_vie` can result in numerical
    instabilities

# Notes
  - Due to high memory and computation costs, `kbsolve!` mutates the initial condition `u0` 
    and only works with in-place rhs functions, unlike standard ODE solvers.
  - The Kadanoff-Baym timestepper is a 2-time generalization of the variable Adams method
    presented in E. Hairer, S. Norsett and G. Wanner, *Solving Ordinary Differential Equations I: Non-
    stiff Problems*, vol. 8, Springer-Verlag Berlin Heidelberg, ISBN 978-3-540-56670-0,
    [doi:10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1) (1993).
"""
function kbsolve!(fv!::Function, fd!::Function, u0::Vector{<:AbstractGreenFunction}, (t0, tmax)::Tuple{Union{Real, Vector{<:Real}}, Real}; 
                  f1! =nothing, v0::Vector=[],
                  callback=(x...)->nothing, stop=(x...)->false,
                  atol=1e-8, rtol=1e-6, dtini=0.0, dtmax=Inf, qmax=5, qmin=1 // 5, γ=9 // 10, kmax=12, kmax_vie=kmax ÷ 2)

  # Support for an initial time-grid
  if t0 isa Real
    t0 = [t0]
  else
    @assert issorted(t0) "Initial time-grid is not in ascending order"
  end
  @assert last(t0) <= tmax "Only t0 <= tmax supported"

  # Holds the information about the integration
  state = (u=u0, v=v0, t=t0, w=initialize_weights(t0), start=[true])

  # Holds the information necessary to integrate
  cache = let
    t1 = length(state.t)
    VCABMCache{eltype(state.t)}(kmax, OrdinaryDiffEq.VectorOfArray([[u[t1, t2] for t2 in 1:t1] for u in state.u]))
  end

  cache_v = isempty(state.v) ? nothing : let
    t1 = length(state.t)
    cache_v = VCABMCache{eltype(state.t)}(kmax, OrdinaryDiffEq.VectorOfArray([[v[t1],] for v in state.v]))
    cache_v.dts = cache.dts
    cache_v
  end

  # The rhs is reshaped into a univariate problem
  f2v!(t1, t2) = fv!(view(cache.f_next, t2, :), state.t, state.w[t1], state.w[t2], t1, t2)
  f2d!(t1, t2) = fd!(view(cache.f_next, t2, :), state.t, state.w[t1], state.w[t2], t1, t2)

  function f2t!()
    t1 = length(state.t)
    Threads.@threads for t2 in 1:(t1 - 1)
      f2v!(t1, t2)
    end
    f2d!(t1, t1)
    return cache.f_next
  end

  function f1t!()
    t1 = length(state.t)
    f1!(view(cache_v.f_next, :), state.t, state.w[t1], t1)
    return cache_v.f_next
  end

  cache.f_prev .= f2t!()
  if !isempty(state.v)
    cache_v.f_prev .= f1t!()
  end

  while timeloop!(state, cache, tmax, dtmax, dtini, atol, rtol, qmax, qmin, γ, kmax_vie, stop)
    t1 = length(state.t)

    # Extend the caches to accomodate the new time column
    extend!(cache, state, f2v!)

    # Predictor
    u_next = predict!(cache, state.t)
    foreach((u, u′) -> foreach(t2 -> u[t1, t2] = u′[t2], 1:t1), state.u, u_next.u)
    if !isempty(state.v)
      u_next = predict!(cache_v, state.t)
      foreach((v, v′) -> v[t1] = v′[1], state.v, u_next.u)
    end
    foreach(t2 -> callback(state.t, state.w[t1], state.w[t2], t1, t2), 1:t1)

    # Corrector
    u_next = correct!(cache, f2t!)
    foreach((u, u′) -> foreach(t2 -> u[t1, t2] = u′[t2], 1:t1), state.u, u_next.u)
    if !isempty(state.v)
      u_next = correct!(cache_v, f1t!)
      foreach((v, v′) -> v[t1] = v′[1], state.v, u_next.u)
    end
    foreach(t2 -> callback(state.t, state.w[t1], state.w[t2], t1, t2), 1:t1)

    # Calculate error and adjust order
    adjust!(cache, cache_v, state.t, f2t!, f1t!, kmax, atol, rtol)
  end # timeloop!
  return (t=state.t, w=state.w)
end

# Controls the step size & resizes the Green functions if required
function timeloop!(state, cache, tmax, dtmax, dtini, atol, rtol, qmax, qmin, γ, kmax_vie, stop)
  if isone(length(state.t)) || state.start[1]
    # Section II.4: Starting Step Size, Eq. (4.14)
    dt = iszero(dtini) ? initial_step(cache.f_prev, cache.u_prev, atol, rtol) : dtini
    state.start[1] = false
  else
    # Section II.4: Automatic Step Size Control, Eq. (4.13)
    q = max(inv(qmax), min(inv(qmin), cache.error_k^(1 / (cache.k + 1)) / γ))
    dt = min((state.t[end] - state.t[end - 1]) / q, dtmax)
  end

  # Don't go over tmax
  if last(state.t) + dt > tmax
    dt = tmax - last(state.t)
  end

  # Remove the last element of the time-grid / weights if last step failed
  if cache.error_k > one(cache.error_k)
    pop!(state.t)
    pop!(state.w)    
  end

  # Reached the end of the integration
  if iszero(dt) || stop(state.t)
    # trim solution
    foreach(u -> resize!(u, length(state.t)), state.u)
    foreach(v -> resize!(v, length(state.t)), state.v)
    return false
  else
    if length(state.t) == (last ∘ size ∘ last)(state.u)
      l = length(state.t) + min(50, ceil(Int, (tmax - state.t[end]) / dt))
      # resize solution
      foreach(u -> resize!(u, l), state.u)
      foreach(v -> resize!(v, l), state.v)
    end
    push!(state.t, last(state.t) + dt)
    push!(state.w, update_weights(last(state.w), state.t, min(cache.k, kmax_vie)))
    return true
  end
end
