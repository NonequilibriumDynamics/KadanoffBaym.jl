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

function VCABMOptions(; atol=1e-8, rtol=1e-6, dtini=1e-6, dtmax=Inf, qmax=5, 
  qmin=1//5, γ=9//10, kmax=12, stop=()->false)

  if kmax < 1 || kmax > 12
    error("kmax must be between 1 and 12")
  end

  return VCABMOptions(atol, rtol, dtini, dtmax, qmax, qmin, γ, kmax, stop)
end

# Part of the following code is licensed under the MIT "Expact" Lience, 
# from https://github.com/SciML/OrdinaryDiffEq.jl
mutable struct VCABMCache{T,U}
  u_prev::U
  u_next::U
  f_prev::U
  ϕ_n::Vector{U}
  ϕ_np1::Vector{U}
  ϕstar_n::Vector{U}
  ϕstar_nm1::Vector{U}
  c::Matrix{T}
  g::Vector{T}
  k::Int # order
  error_k::T
  u_erro::U

  function VCABMCache{T}(max_k, u_prev::U, f_prev) where {T,U} # k = 1
    ϕ_n = [zero.(f_prev) for _ in 1:max_k+1]
    ϕstar_nm = [zero.(f_prev) for _ in 1:max_k+1]
    ϕstar_n = [zero.(f_prev) for _ in 1:max_k+1]
    ϕ_np = [zero.(f_prev) for _ in 1:max_k+2]
    new{T,U}(u_prev,zero.(u_prev),f_prev,ϕ_n,ϕ_np,ϕstar_n,ϕstar_nm,zeros(T,max_k+1,max_k+1),zeros(T,max_k+1),1,T(Inf),zero.(u_prev))
  end
end

function insert_cache!(a::VCABMCache, i, item::VCABMCache)
  insert!(a.u_prev.u, i, item.u_prev)
  insert!(a.u_next.u, i, item.u_next)
  insert!(a.f_prev.u, i, item.f_prev)
  insert!(a.u_erro.u, i, item.u_erro)
  for k in 1:length(a.ϕ_n)
    insert!(a.ϕ_np1[k].u, i, item.ϕ_np1[k])
    insert!(a.ϕ_n[k].u, i, item.ϕ_n[k])
    insert!(a.ϕstar_nm1[k].u, i, item.ϕstar_nm1[k])
    insert!(a.ϕstar_n[k].u, i, item.ϕstar_n[k])
  end
end

function update_cache!(f_next, cache, max_k)
  @. cache.u_prev = cache.u_next
  @. cache.f_prev = f_next
  cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1
  cache.k = min(max_k, cache.k+1) # Ramp up order
end

# Explicit Adams: Section III.5 Eq. (5.5)
function predict!(state, cache)
  @inbounds begin
    @unpack u_prev,u_next,g,ϕstar_n,k = cache
    ϕ_and_ϕstar!(state, cache, k+1)
    @. u_next = muladd(g[1], ϕstar_n[1], u_prev)
    for i = 2:k-1 # NOTE: Check this (-1)
      @. u_next = muladd(g[i], ϕstar_n[i], u_next)
    end
  end
  u_next
end

# Implicit Adams: Section III.5 Eq (5.7)
function correct!(du, cache, T)
  @unpack u_next,g,ϕ_np1,k = cache
  @inbounds begin
    ϕ_np1!(cache, du, k+1, T)
    @. u_next[T] = muladd(g[k], ϕ_np1[k][T], u_next[T])
  end
  u_next[T]
end

function ϕ_np1!(cache, du, k, T)
  @unpack ϕ_np1, ϕstar_n = cache
  @inbounds begin
    @. ϕ_np1[1][T] = du
    for i = 2:k
      @. ϕ_np1[i][T] = ϕ_np1[i-1][T] - ϕstar_n[i-1][T]
    end
  end
end

# Control order: Section III.7 Eq. (7.7)
function adjust_order!(f_next, state, cache, max_k, atol, rtol)
  @inbounds begin
    @unpack t = state
    @unpack u_prev,u_next,g,ϕ_np1,ϕstar_n,k,u_erro = cache

    # Calculate error: Section III.7 Eq. (7.3)
    cache.error_k = error_estimate!(u_erro, (g[k+1]-g[k]) * ϕ_np1[k+1], u_prev, u_next, atol, rtol) |> norm
    
    # Fail step: Section III.7 Eq. (7.4)
    if cache.error_k > one(cache.error_k)
      return
    end

    f_next = VectorOfArray(collect(f_next))

    if length(t)<=5 || k<3
      cache.k = min(k+1, 3, max_k)
    else
      error_k1 = error_estimate!(u_erro, (g[k]-g[k-1]) * ϕ_np1[k], u_prev, u_next, atol, rtol) |> norm
      error_k2 = error_estimate!(u_erro, (g[k-1]-g[k-2]) * ϕ_np1[k-1], u_prev, u_next, atol, rtol) |> norm
      if max(error_k2, error_k1) <= cache.error_k
        cache.k = k-1
      else
        foreach(t′ -> ϕ_np1!(cache, f_next[t′], k+2, t′), eachindex(state.t))
        error_kstar = error_estimate!(u_erro, (t[end] - t[end-1]) * γstar[k+2] * ϕ_np1[k+2], u_prev, u_next, atol, rtol) |> norm
        if error_kstar < cache.error_k
          cache.k = min(k+1, max_k)
          cache.error_k = one(cache.error_k)   # constant dt
        end
      end
    end
    @. cache.u_prev = cache.u_next
    @. cache.f_prev = f_next
    cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1
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
        ϕ_n[1] .= f_prev
        ϕstar_n[1] .= f_prev
      else
        β = β * (t_next - t[i]) / (t_prev - t[i+1])
        @. ϕ_n[i] = ϕ_n[i-1] - ϕstar_nm1[i-1]
        @. ϕstar_n[i] = β * ϕ_n[i]
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
@inline function error_estimate!(out::AbstractArray,ũ::AbstractArray, u₀::AbstractArray, u₁::AbstractArray, atol::Real, rtol::Real)
  @. out = error_estimate!(out, ũ, u₀, u₁, atol, rtol)
  out
end
@inline function error_estimate!(out::AbstractArray{<:Number},ũ::AbstractArray{<:Number}, u₀::AbstractArray{<:Number}, u₁::AbstractArray{<:Number}, atol::Real, rtol::Real)
  @. out = error_estimate(ũ, u₀, u₁, atol, rtol)
  out
end
@inline function error_estimate(ũ::Number, u₀::Number, u₁::Number, atol::Real, rtol::Real)
  ũ / (atol + max(norm(u₀), norm(u₁)) * rtol)
end
@inline norm(u) = LinearAlgebra.norm(u) / sqrt(total_length(u))
@inline total_length(u::Number) = length(u)
@inline total_length(u::AbstractArray{<:Number}) = length(u)
@inline total_length(u::AbstractArray{<:AbstractArray}) = sum(total_length, u)

# Starting Step Size: Section II.4
function initial_step(f, u0, t0, k, atol, rtol; f0=f(u0, t0))
  sc = atol + rtol * norm(u0) 
  d0 = norm(u0 ./ sc)
  d1 = norm(f0 ./ sc)

  dt0 = min(d0,d1) < 1e-5 ? 1e-6 : 1e-2 * d0/d1
  
  f1 = f(muladd(dt0, f0, u0), t0+dt0)
  d2 = norm((f1 - f0) ./ sc) / dt0
  
  dt1 = max(d1,d2) <= 1e-15 ? max(1e-6, 1e-3 * dt0) : (1e-2 / max(d1,d2))^(1/(k+1))
  return min(dt1, 1e2 * dt0)
end
