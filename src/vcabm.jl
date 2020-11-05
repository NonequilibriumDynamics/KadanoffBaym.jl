struct VCABMOptions
  atol::Number
  rtol::Number
  dtini::Number
  dtmax::Number
  qmax::Number
  qmin::Number
  γ::Number
  kmax::Number
  stop::Function
end

function VCABMOptions(; atol=1e-8, rtol=1e-6, dtini=0.0, dtmax=1e-1, qmax=5, 
  qmin=1//5, γ=9//10, kmax=12, stop=()->false)

  if kmax < 1 || kmax > 12
    error("kmax must be between 1 and 12")
  end

  return VCABMOptions(atol, rtol, dtini, dtmax, qmax, qmin, γ, kmax, stop)
end

# Holds the information about the integration
mutable struct VCABMState{T,U,V}
  u::U
  v::V
  t::T

  function VCABMState(u::U, v::V, t::T) where {T,U,V}
    new{T,U,V}(u, v, t)
  end
end

mutable struct VCABMCache{T,U}
  u_prev::U
  f_prev::U
  ϕ_n::Vector{U}
  ϕ_np1::Vector{U}
  ϕstar_n::Vector{U}
  ϕstar_nm1::Vector{U}
  c::Matrix{T}
  g::Vector{T}
  k::Int # order
  error_k::T

  function VCABMCache{T}(max_k, u_prev::U, f_prev) where {T,U} # k = 1
    ϕ_n = Vector{U}(undef, max_k+1); ϕ_n[1] = copy(f_prev)
    ϕstar_nm = Vector{U}(undef, max_k+1); ϕstar_nm[1] = copy(f_prev)
    ϕstar_n = Vector{U}(undef, max_k+1); ϕstar_n[1] = copy(f_prev)
    ϕ_np = Vector{U}(undef, max_k+2)
    new{T,U}(u_prev,f_prev,ϕ_n,ϕ_np,ϕstar_n,ϕstar_nm,zeros(T,max_k+1,max_k+1),zeros(T,max_k+1),1,zero(T))
  end
end

# Explicit Adams: Section III.5 Eq. (5.5)
function predict!(state, cache)
  @inbounds begin
    @unpack u_prev,g,ϕstar_n,k = cache
    ϕ_and_ϕstar!(state, cache, k+1)
    u_next = muladd(g[1], ϕstar_n[1], u_prev)
    for i = 2:k-1 # NOTE: Check this (-1)
      u_next = muladd(g[i], ϕstar_n[i], u_next)
    end
  end
  u_next
end

# Implicit Adams: Section III.5 Eq (5.7)
function correct!(u_next, du, cache)
  @unpack g,ϕ_np1,k = cache
  @inbounds begin
    ϕ_np1!(cache, du, k+1)
    u_next = muladd(g[k], ϕ_np1[k], u_next)
  end
  u_next
end

function ϕ_np1!(cache, du, k)
  @unpack ϕ_np1, ϕstar_n = cache
  @inbounds begin
    ϕ_np1[1] = du
    for i = 2:k
      ϕ_np1[i] = ϕ_np1[i-1] - ϕstar_n[i-1]
    end
  end
end

# Calculate error: Section III.7 Eq. (7.3)
function estimate_error!(u_next, cache, atol, rtol)
  @unpack u_prev,g,ϕ_np1,ϕstar_n,k = cache
  error_k = error_estimate((g[k+1]-g[k]) * ϕ_np1[k+1], u_prev, u_next, atol, rtol) |> norm
  cache.error_k = error_k
end

# Control order: Section III.7 Eq. (7.7)
@muladd function adjust_order!(u_next, f_next, state, cache, max_k, atol, rtol)
  @inbounds begin
    @unpack t = state
    @unpack u_prev,g,ϕ_np1,ϕstar_n,error_k,k = cache

    # Fail step: Section III.7 Eq. (7.4)
    if error_k > one(error_k)
      @assert false
    end

    if length(t)<=5 || k<3
      cache.k = min(k+1, 3, max_k)
    else
      error_k1 = error_estimate((g[k]-g[k-1]) * ϕ_np1[k], u_prev, u_next, atol, rtol) |> norm
      error_k2 = error_estimate((g[k-1]-g[k-2]) * ϕ_np1[k-1], u_prev, u_next, atol, rtol) |> norm
      if max(error_k2, error_k1) <= error_k
        cache.k = k-1
      else
        ϕ_np1!(cache, f_next, k+2)
        error_kstar = error_estimate((t[end] - t[end-1]) * γstar[k+2] * ϕ_np1[k+2], u_prev, u_next, atol, rtol) |> norm
        if error_kstar < error_k
          cache.k = min(k+1, max_k)
          cache.error_k = one(error_k)   # constant dt
        end
      end
    end
  end
end

# Section III.5: Eq (5.9-5.10)
function ϕ_and_ϕstar!(state, cache, k)
  @inbounds begin
    @unpack t = state
    @unpack f_prev, ϕstar_nm1, ϕ_n, ϕstar_n, c, g = cache
    t = reverse(t)
    t_next = t[1]
    t_prev = t[2]
    β = one(eltype(t))

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
          c[i,q] = muladd(-(t_next - t_prev)/(t_next - t[i]), c[i-1,q+1], c[i-1,q])
        elseif i == 1
          c[i,q] = inv(q)
        elseif i == 2
          c[i,q] = inv(q*(q+1))
        end
      end
      g[i] = c[i,1] * (t_next - t_prev)
    end
  end
end

# Coefficients for the implicit Adams methods
const γstar = [1,-1/2,-1/12,-1/24,-19/720,-3/160,-863/60480,-275/24192,-33953/3628800,-0.00789255,-0.00678585,-0.00592406,-0.00523669,-0.0046775,-0.00421495,-0.0038269]

# Error estimation and norm: Section II.4 Eq. (4.11)
@inline function error_estimate(ũ::AbstractArray, u₀::AbstractArray, u₁::AbstractArray, atol::Real, rtol::Real)
  err = similar(ũ)
  @. err = error_estimate(ũ, u₀, u₁, atol, rtol)
  err
end
@inline function error_estimate(ũ::Number, u₀::Number, u₁::Number, atol::Real, rtol::Real)
  ũ / (atol + max(norm(u₀), norm(u₁)) * rtol)
end
@inline norm(u) = LinearAlgebra.norm(u) / sqrt(length(u))

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
