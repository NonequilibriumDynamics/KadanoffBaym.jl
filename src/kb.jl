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
  - `f_line(u, ts, t1)`: the right-hand-side for 1-point functions
  - `l0`: initial condition for the 1-point functions
  - `kernel_vert(ts, t1, t2, τ)`: the integral kernel of `du/dt1`
  - `kernel_diag(ts, t1, τ)`: the integral kernel of `du/dt1 + du/dt2`
  - `callback(ts, t1)`: A function that gets called everytime the 2-point function values at slice `(t1, 1:t1)` are updated
  - `kwargs...`: see `VCABMOptions`

# Notes
  - Unlike standard ODE solvers, `kbsolve` is designed to mutate the initial conditions
  - The Kadanoff-Baym timestepper is a 2-time generalization of the VCABM stepper 
    presented in Ernst Hairer, Gerhard Wanner, and Syvert P Norsett
    Solving Ordinary Differential Equations I: Nonstiff Problems
"""
function kbsolve(f_vert, f_diag, u0::Vector{<:GreenFunction}, (t0, tmax); 
  k_vert=nothing, k_diag=nothing, f_line=nothing, l0=nothing, callback=(x...) -> nothing, kwargs...)
  opts = VCABMOptions(; kwargs...)

  # Support for an initial time-grid
  if isempty(size(t0))
    t0 = [t0]
  else
    @assert issorted(t0) "Initial time-grid is not in ascending order"
  end
  @assert last(t0) < tmax "Only t0 < tmax supported"

  # Holds the information about the integration
  state = KBState(u0, t0)

  # Holds the information necessary to time-step
  cache = begin
    t = length(state.t)
    u_prev = VectorOfArray([[x[t, t′] for x in state.u] for t′ in 1:t])

    if isnothing(k_vert)
      # The system is viewed as 1-time ODE whose rhs is the concatenation of all rhs up to time the index `t`
      f = t -> VectorOfArray([[f_vert(state.u, state.t, t, t′) for t′ in 1:(t - 1)]; [f_diag(state.u, state.t, t)]])
    else
      # The same is done for the Volterra kernels
      k = (t, s) -> VectorOfArray([[k_vert(state.u, state.t, t, t′, s) for t′ in 1:(t - 1)]; [k_diag(state.u, state.t, t, s)]])

      # Holds the information necessary to integrate Volterra kernels
      cache_v = VCABMVolterraCache{eltype(state.t)}(opts.kmax, typeof(u_prev)(k(t, t).u))

      f = t -> begin
        # Integrates the Volterra kernels v(t) = ₀∫ᵗ ds K(t, s)
        v = quadrature!(cache_v, state.t, s -> k(t, s))
        VectorOfArray([[f_vert(state.u, v[t′], state.t, t, t′) for t′ in 1:(t - 1)]; [f_diag(state.u, v[t], state.t, t)]])
      end
    end

    VCABMCache{eltype(state.t)}(opts.kmax, u_prev, typeof(u_prev)(f(t).u))
  end

  while timeloop!(state, cache, tmax, opts)
    t = length(state.t)

    # Extend the cache to accomodate the new time column
    if isnothing(k_vert)
      extend!(cache, state.t, (t, t′) -> f_vert(state.u, state.t, t, t′))
    else
      extend!(cache, cache_v, state.t, (v, t, t′) -> f_vert(state.u, v, state.t, t, t′), (t, t′, s) -> k_vert(state.u, state.t, t, t′, s))
    end

    # Predictor
    u_next = predict!(cache, state.t)
    foreach((u, u′) -> foreach(t′ -> u[t, t′] = u′[t′], 1:t), state.u, eachrow(u_next))
    callback(state.t, t)

    # Corrector
    u_next = correct!(cache, () -> f(t))
    foreach((u, u′) -> foreach(t′ -> u[t, t′] = u′[t′], 1:t), state.u, eachrow(u_next))
    callback(state.t, t)

    # Calculate error and adjust order
    adjust!(cache, state.t, () -> f(t), opts.kmax, opts.atol, opts.rtol)
  end # timeloop!
  return state
end

# Controls the step size & resizes the Green functions if required
function timeloop!(state, cache, tmax, opts)
  if isone(length(state.t))
    # Section II.4: Starting Step Size, Eq. (4.14)
    dt = iszero(opts.dtini) ? initial_step(cache.f_prev, cache.u_prev, opts.atol, opts.rtol) : opts.dtini
  else
    # Section II.4: Automatic Step Size Control, Eq. (4.13)
    q = max(inv(opts.qmax), min(inv(opts.qmin), cache.error_k^(1 / (cache.k + 1)) / opts.γ))
    dt = min((state.t[end] - state.t[end - 1]) / q, opts.dtmax)
  end

  # Remove the last element of the time grid if last step failed
  if cache.error_k > one(cache.error_k)
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
  else
    if length(state.t) == (last ∘ size ∘ first)(state.u) # resize solution
      foreach(u -> resize!(u, length(state.t) + min(50, ceil(Int, (tmax - state.t[end]) / dt))), state.u)
    end
    push!(state.t, last(state.t) + dt)
    return true
  end
end

struct KBState{U,T}
  u::U          # 2-point functions
  t::Vector{T}  # time grid
end
