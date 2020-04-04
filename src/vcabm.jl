using Parameters: @unpack

# do-while notation in Julia
macro do_while(condition, block)
  quote
    let
      $block
      while $condition
        $block
      end
    end
  end |> esc
end

function vcabm(f, u₀, t₀, tmax; dt=nothing, adaptive=true, max_dt=1e-1,
  atol=1e-9, rtol=1e-7, max_order=12, qmax=5, qmin=1//5, γ=9//10)
  
  if max_order < 1 || max_order > 12
    error("max_order must be between 1 and 12")
  end

  if t₀ > tmax
    error("Only t₀ < tmax supported")
  end

  state, cache = initialize(f, u₀, t₀, dt, max_order, atol, rtol)

  @do_while timeloop!(state,cache,tmax,max_dt,adaptive,qmax,qmin,γ) begin
    predict_correct!(f, state, cache, max_order, atol, rtol, adaptive)

    # update state
    if cache.error_k <= one(cache.error_k)
      push!(state.u, cache.u_prev)
    end
  end

  return state.u, state.t
end

function timeloop!(state,cache,tmax,max_dt,adaptive,qmax,qmin,γ)
  @unpack k, error_k= cache

  if adaptive
    if cache.error_k > one(cache.error_k)
      pop!(state.t) # remove t_prev
    end

    # II.4 Automatic Step Size Control, Eq. (4.13)
    q = max(inv(qmax), min(inv(qmin), error_k^(1/(k+1)) / γ))
    state.dt = min(state.dt / q, max_dt)

    # Don't go over tmax
    if state.t[end] + state.dt > tmax
      state.dt = tmax - state.t[end]
    end
  end

  push!(state.t, state.t[end] + state.dt) # add t_next

  return state.t[end] < tmax
end

# Holds the information about the integration
mutable struct VCABMState{T,U}
  u::U
  t::Vector{T}
  dt::T

  function VCABMState(u::U, t₀::T, dt::T) where {T,U}
    new{T,U}(u,[t₀,t₀+dt],dt)
  end
end

mutable struct VCABMCache{T,F,U}
  u_prev::U
  f_prev::F
  ϕ_n::Vector{F}
  ϕ_np1::Vector{F}
  ϕstar_n::Vector{F}
  ϕstar_nm1::Vector{F}
  c::Matrix{T}
  g::Vector{T}
  k::Int # order
  error_k::T

  function VCABMCache{T}(max_k, u_prev::U, f_prev::F) where {T,F,U} # k = 1
    ϕ_n = Vector{F}(undef, max_k+1); ϕ_n[1] = copy(f_prev)
    ϕstar_nm = Vector{F}(undef, max_k+1); ϕstar_nm[1] = copy(f_prev)
    ϕstar_n = Vector{F}(undef, max_k+1); ϕstar_n[1] = copy(f_prev)
    ϕ_np = Vector{F}(undef, max_k+2)
    new{T,F,U}(u_prev,f_prev,ϕ_n,ϕ_np,ϕstar_n,ϕstar_nm,zeros(T,max_k+1,max_k+1),zeros(T,max_k+1),1,zero(T))
  end
end

# Initialize VCABM state
function initialize(f, u₀, t₀, dt₀, max_k, atol, rtol)
  f₀ = f(u₀, t₀)

  if dt₀ === nothing
    dt₀ = initial_step(f, u₀, t₀, 1, rtol, atol; f₀=f₀)
  end

  return VCABMState([u₀,],t₀,dt₀), VCABMCache{typeof(t₀)}(max_k,u₀,f₀)
end

# Solving Ordinary Differential Equations I: Nonstiff Problems, by Ernst Hairer, Gerhard Wanner, and Syvert P Norsett
function predict_correct!(f, state, cache, max_k, atol, rtol, adaptive; update=nothing)
  @inbounds begin
    @unpack t,dt = state
    @unpack u_prev,g,ϕ_np1,ϕstar_n,k = cache

    t_next = t[end]

    # Explicit Adams: Section III.5 Eq. (5.7)
    ϕ_and_ϕstar!(state, cache, k+1)
    u_next = muladd(g[1], ϕstar_n[1], u_prev)
    for i = 2:k-1
      u_next = muladd(g[i], ϕstar_n[i], u_next)
    end

    # Implicit corrector
    du_np1 = f(u_next, t_next)
    ϕ_np1!(cache, du_np1, k+1)
    u_next = muladd(g[k], ϕ_np1[k], u_next)

    if adaptive
      # Calculate error: Section III.7 Eq. (7.3)
      error_k = error_estimate((g[k+1]-g[k]) * ϕ_np1[k+1], u_prev, u_next, atol, rtol) |> norm
      cache.error_k = error_k

      # Fail step: Section III.7 Eq. (7.4)
      if error_k > one(error_k)
        return
      end

      # Accept step
      f_next = f(u_next, t_next)

      if length(t)<=5 || k<3
        cache.k = min(k+1, 3, max_k)
      else
        # Control order: Section III.7 Eq. (7.7)
        error_k1 = error_estimate((g[k]-g[k-1]) * ϕ_np1[k], u_prev, u_next, atol, rtol) |> norm
        error_k2 = error_estimate((g[k-1]-g[k-2]) * ϕ_np1[k-1], u_prev, u_next, atol, rtol) |> norm
        if max(error_k2, error_k1) <= error_k
          cache.k = k-1
        else
          ϕ_np1!(cache, f_next, k+2)
          error_kstar = error_estimate(dt * γstar[k+2] * ϕ_np1[k+2], u_prev, u_next, atol, rtol) |> norm
          if error_kstar < error_k
            cache.k = min(k+1, max_k)
            cache.error_k = one(error_k)   # constant dt
          end
        end
      end
    end # adaptive
    (update !== nothing) && (update(u_next); return) # needed for KB-stepper
    cache.u_prev = u_next
    cache.f_prev = f_next
    cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1
  end
end

# Section III.5: Eq (5.9-5.10)
function ϕ_and_ϕstar!(state, cache, k)
  @inbounds begin
    @unpack t, dt = state
    @unpack f_prev, ϕstar_nm1, ϕ_n, ϕstar_n, c, g = cache
    t = reverse(t)
    t_next = t[1]
    t_prev = t[2]
    β = one(dt)

    for i = 1:k
      # Calculation of Φ
      if i == 1
        ϕ_n[1] = f_prev
        ϕstar_n[1] = f_prev
      else
        β = β * (t_next - t[i]) / (t_prev - t[i+1])
        ϕ_n[i] = ϕ_n[i-1] - ϕstar_nm1[i-1]
        ϕstar_n[i] = β * ϕ_n[i]
      end

      # Calculation of g
      for q = 1:k-(i-1)
        if i > 2
          c[i,q] = muladd(-dt/(t_next - t[i]), c[i-1,q+1], c[i-1,q])
        elseif i == 1
          c[i,q] = inv(q)
        elseif i == 2
          c[i,q] = inv(q*(q+1))
        end
      end
      g[i] = c[i,1] * dt
    end
  end
end
function ϕ_np1!(cache, du_np1, k)
  @inbounds begin
    @unpack ϕ_np1, ϕstar_n = cache
    ϕ_np1[1] = du_np1
    for i = 2:k
      ϕ_np1[i] = ϕ_np1[i-1] - ϕstar_n[i-1]
    end
  end
end

# Coefficients for the implicit Adams methods
const γstar = [1,-1/2,-1/12,-1/24,-19/720,-3/160,-863/60480,-275/24192,-33953/3628800,-0.00789255,-0.00678585,-0.00592406,-0.00523669,-0.0046775,-0.00421495,-0.0038269]

# Starting Step Size: Section II.4
function initial_step(f, u₀, t₀, k, atol, rtol; f₀=f(u₀, t₀))
  sc = atol + norm(u₀) * rtol
  d₀ = norm(u₀ ./ sc)
  d₁ = norm(f₀ ./ sc)

  dt₀ = min(d₀,d₁) < 1e-5 ? 1e-6 : 1e-2 * d₀/d₁
  
  f₁ = f(muladd(dt₀, f₀, u₀), t₀+dt₀)
  d₂ = norm((f₁ - f₀) ./ sc) / dt₀
  
  dt₁ = max(d₁,d₂) <= 1e-15 ? max(1e-6, 1e-3 * dt₀) : (1e-2 / max(d₁,d₂))^(1/(k+1))
  return min(dt₁, 1e2 * dt₀) * 1e-3
end

# Error estimation and norm: Section II.4 Eq. (4.11)
# @inline function error_estimate(ũ::Array{T}, u₀::Array{T}, u₁::Array{T}, atol::Real, rtol::Real) where {T<:Number}
@inline function error_estimate(ũ::Array, u₀::Array, u₁::Array, atol::Real, rtol::Real)
  err = similar(ũ)
  @. err = error_estimate(ũ, u₀, u₁, atol, rtol)
  err
end
@inline function error_estimate(ũ::Number, u₀::Number, u₁::Number, atol::Real, rtol::Real)
  ũ / (atol + max(norm(u₀), norm(u₁)) * rtol)
end
@inline norm(u::Union{AbstractFloat,Complex}) = abs(u)
# @inline norm(u::Array{T}) where T<:Union{AbstractFloat,Complex} = sqrt(sum(abs2,u) / length(u))
@inline norm(u::Array) = sqrt(sum(abs2,u) / length(u))
