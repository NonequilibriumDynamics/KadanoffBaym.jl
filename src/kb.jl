# A time slice of the product of field spaces, with owned values on assignment.
struct FieldSlice{T,F} <: AbstractVector{T}
  fields::F
  j::Int
end
FieldSlice(fields, j) = FieldSlice{eltype(eltype(fields)),typeof(fields)}(fields, j)
Base.size(out::FieldSlice) = (length(out.fields),)
Base.getindex(out::FieldSlice, a::Int) = out.fields[a][out.j]
function Base.setindex!(out::FieldSlice, value, a::Int)
  destination = out.fields[a][out.j]
  axes(destination) == axes(value) || throw(DimensionMismatch("RHS values must retain the field's axes"))
  out.fields[a][out.j] = destination isa AbstractArray{<:Number} && value isa AbstractArray{<:Number} ?
    _recursivecopy!(destination, value) : recursivecopy(value)
  return out
end

"""
    kbsolve!(fv!::Function, fd!::Function, u0::Vector{<:AbstractGreenFunction}, (t0, tmax)::Tuple{Union{Real, Vector{<:Real}}, Real})

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
    respectively. The output is saved in-place in `out`, which has the same shape as `u0`.
    Each output field owns its values. Assigning `out[a] = value` copies its
    numerical values, converting to field `a`'s stored type and retaining its
    axes. Use `out[a] .= value` to write directly into an array-valued buffer.
    Assignments do not create aliases between output fields or retain the
    supplied array; finish computing a workspace before assigning it to `out`.
    Vertical RHS evaluations may run concurrently; reusable workspaces must
    be local to each thread.
  - `fd!(out, ts, w1, w2, t1, t2)`: The right-hand side of ``(du(t_1,t_2)/dt_1 + du(t_1,t_2)/dt_2)|_{t_2 → t_1}``
  - `u0::Vector{<:GreenFunction}`: List of 2-point functions to be integrated
  - `(t0, tmax)`: A tuple with the initial time(s) `t0` – can be a vector of 
    past times – and final time `tmax`

# Optional keyword parameters
  - `f1!(out, ts, w1, t1)`: The right-hand-side of ``dv(t_1)/dt_1``. The weight `w1`
    can be used to integrate the Volterra kernel and the output is saved in-place in `out`, 
    which has the same shape as `v0` and the same owned-value assignment contract
  - `v0::Vector{<:GreenFunction}`: List of 1-point functions to be integrated
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
  - `kmax_vie::Integer`: Maximum polynomial degree of the memory quadrature.
    Higher degrees require smoother kernels and can amplify sample errors on
    uneven grids; this bound does not provide a memory-error estimate.
  - `quadrature::Symbol`: `:smooth` uses the usual high-order Volterra quadrature.
    `:piecewise` supports continuous kernels with a derivative cusp at `t2`: it
    integrates each panel using its endpoints and the nearest earlier point on
    the same side of the cusp at least one panel width from its left endpoint.
    If no such point exists, it uses a trapezoid. For kernels with three bounded
    derivatives on each side, this gives third-order memory quadrature without a
    step-ratio restriction. The option caps `kmax_vie` at 2
    and `kmax` at `kmax_vie + 1`; lower degrees remain available. Panel weights
    depend only on the time grid, so a supplied history reconstructs the same rule.
    `fv!` and `callback` must accept integration weights as an `AbstractVector`;
    copy these temporary weights if retaining them. The returned `sol.w` remains
    the ordinary cumulative weights.

# Notes
  - Due to high memory and computation costs, `kbsolve!` mutates the initial condition `u0` 
    and only works with in-place rhs functions, unlike standard ODE solvers.
  - High-order Volterra quadrature requires kernels smooth across their interpolation
    stencils. Subtracting cumulative weights does not restart those stencils at an
    interior cusp. Use `quadrature=:piecewise` for a continuous kernel with a derivative
    cusp at `t2`; jumps require separate one-sided endpoint values.
  - `atol` and `rtol` control the Adams local error estimate. Check Volterra quadrature
    convergence by reducing `dtmax`, and supply an initial history fine enough to
    resolve its kernels; tightening tolerances cannot repair past quadrature error.
    Residuals are scaled componentwise by `atol + rtol * max(abs(u_prev), abs(u_next))`.
    Acceptance uses the maximum over fields and time coordinates, with component
    RMS within each scalar, vector, or matrix value. Adding history or inactive
    fields therefore cannot dilute an isolated local error estimate.
  - The Kadanoff-Baym timestepper is a 2-time generalization of the variable Adams method
    presented in E. Hairer, S. Norsett and G. Wanner, *Solving Ordinary Differential Equations I: Non-
    stiff Problems*, vol. 8, Springer-Verlag Berlin Heidelberg, ISBN 978-3-540-56670-0,
    [doi:10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1) (1993).
"""
function kbsolve!(fv!::Function, fd!::Function, u0::Vector{<:AbstractGreenFunction}, (t0, tmax)::Tuple{Union{Real, Vector{<:Real}}, Real}; 
                  f1! =nothing, v0::Vector=[],
                  callback=(x...)->nothing, stop=(x...)->false,
                  atol=1e-8, rtol=1e-6, dtini=0.0, dtmax=Inf, qmax=5, qmin=1 // 5, γ=9 // 10, kmax=12, kmax_vie=max(1, kmax ÷ 2), quadrature=:smooth)

  quadrature in (:smooth, :piecewise) || throw(ArgumentError("quadrature must be :smooth or :piecewise"))
  if quadrature == :piecewise
    kmax_vie = min(kmax_vie, 2)
    kmax = min(kmax, kmax_vie + 1)
  end

  # Support for an initial time-grid
  if t0 isa Real
    t0 = [t0]
  end
  T = float(promote_type(eltype(t0), typeof(tmax)))
  t0 = T.(t0)
  tmax = T(tmax)
  @assert all(>(0), diff(t0)) "Initial time-grid must be strictly increasing"
  @assert last(t0) <= tmax "Only t0 <= tmax supported"

  # Holds the information about the integration
  w1 = quadrature == :piecewise && kmax_vie == 2 ? Vector{Vector{T}}() : nothing
  state = (u=u0, v=v0, t=t0, w=initialize_weights(t0, kmax_vie, quadrature), w1=w1)
  isnothing(w1) || piecewise_weights!(w1, state.t, last(state.w))

  n0, n2 = length(state.t), length(state.u)
  u = [[copy(g[n0, j]) for j in eachindex(state.t)] for g in state.u]
  if !isempty(state.v)
    v = [[copy(v[n0])] for v in state.v]
    u = collect(Union{eltype(u),eltype(v)}, Iterators.flatten((u, v)))
  end
  cache = VCABMCache{eltype(state.t)}(kmax, u)

  weights(i, j) = !isnothing(state.w1) && 1 < j < i ? state.w1[j] : state.w[i]
  f2v!(out, i, j) = fv!(FieldSlice(view(out, 1:n2), j), state.t, weights(i, j), state.w[j], i, j)
  function f!(out)
    i = length(state.t)
    Threads.@threads for j in 1:(i - 1)
      f2v!(out, i, j)
    end
    fd!(FieldSlice(view(out, 1:n2), i), state.t, state.w[i], state.w[i], i, i)
    if !isempty(state.v)
      f1!(FieldSlice(view(out, (n2 + 1):length(out)), 1), state.t, state.w[i], i)
    end
    return out
  end
  f!(cache.ϕ_n[1])

  if iszero(dtini)
    n = length(state.t)
    f2v!(cache.ϕ_np1[1], n, n)
    # The diagonal derivative is already present; add the new vertical derivative at the same initial value.
    u_initial = [i <= n2 ? vcat(cache.u_prev[i], [last(cache.u_prev[i])]) : cache.u_prev[i] for i in eachindex(cache.u_prev)]
    f_initial = [i <= n2 ? vcat(cache.ϕ_n[1][i], [last(cache.ϕ_np1[1][i])]) : cache.ϕ_n[1][i] for i in eachindex(cache.ϕ_n[1])]
    dtini = initial_step(f_initial, u_initial, atol, rtol)
    dtini = max(dtini, nextfloat(last(t0)) - last(t0))
  end

  function write_row!(i)
    for (a, g) in enumerate(state.u), j in 1:i
      value = cache.u_next[a][j]
      g[i, j] = eltype(g) <: Number ? value : copy(value)
    end
    for (a, v) in enumerate(state.v)
      value = cache.u_next[n2 + a][1]
      v[i] = eltype(v) <: Number ? value : copy(value)
    end
    foreach(j -> callback(state.t, weights(i, j), state.w[j], i, j), 1:i)
  end

  while timeloop!(state, cache, tmax, dtmax, dtini, qmax, qmin, γ, kmax_vie, quadrature, stop)
    t1 = length(state.t)

    extend!(cache, state.t, f2v!, n2)
    predict!(cache, state.t; n2)
    write_row!(t1)
    correct!(cache, f!)
    write_row!(t1)
    adjust!(cache, state.t, f!, kmax, atol, rtol; n2)
  end # timeloop!
  return (t=state.t, w=state.w)
end

"""Adapt the existing weighted-RHS callbacks to the experimental integral interface."""
function weighted_equations(fv!, fd!; f1! = nothing, callback=(args...) -> nothing)
  return (
    fv! = (out,t,Q,i,j) -> fv!(out,t,Q.prefix[i],Q.prefix[j],i,j),
    fd! = (out,t,Q,i,j) -> fd!(out,t,Q.prefix[i],Q.prefix[j],i,j),
    f1! = isnothing(f1!) ? nothing : (out,t,Q,i) -> f1!(out,t,Q.prefix[i],i),
    callback = (t,Q,i,j) -> callback(t,Q.prefix[i],Q.prefix[j],i,j),
  )
end

# Controls the step size & resizes the Green functions if required
function timeloop!(state, cache, tmax, dtmax, dtini, qmax, qmin, γ, kmax_vie, quadrature, stop)
  if iszero(cache.steps) && iszero(cache.error_k)
    dt = dtini
  else
    # Section II.4: Automatic Step Size Control, Eq. (4.13)
    q = max(inv(qmax), min(inv(qmin), cache.error_k^(1 / (cache.k + 1)) / γ))
    dt = (state.t[end] - state.t[end - 1]) / q
  end

  # Remove the last element of the time-grid / weights if last step failed
  if cache.error_k > one(cache.error_k)
    t_rejected = pop!(state.t)
    pop!(state.w)    
  end

  dt = min(dt, dtmax, tmax - last(state.t))

  # Reached the end of the integration
  if last(state.t) == tmax || stop(state.t)
    # trim solution
    foreach(u -> resize!(u, length(state.t)), state.u)
    foreach(v -> resize!(v, length(state.t)), state.v)
    return false
  else
    t_next = min(tmax, oftype(last(state.t), last(state.t) + dt))
    if t_next - last(state.t) > dtmax
      t_next = prevfloat(t_next)
    end
    if cache.error_k > one(cache.error_k)
      t_next = min(t_next, prevfloat(t_rejected))
    end
    t_next > last(state.t) || error("Time step $dt cannot advance time from $(last(state.t)).")
    for functions in (state.u, state.v), u in functions
      if length(state.t) == last(size(u))
        l = length(state.t) + ceil(Int, min(50, (tmax - state.t[end]) / dt))
        resize!(u, l)
      end
    end
    push!(state.t, t_next)
    k = quadrature == :piecewise ? kmax_vie : min(cache.k, kmax_vie)
    push!(state.w, update_weights(last(state.w), state.t, k, quadrature))
    isnothing(state.w1) || piecewise_weights!(state.w1, state.t, last(state.w))
    return true
  end
end
