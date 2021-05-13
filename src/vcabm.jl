"""
  VCABMOptions(...)

Returns a the parameters needed to control a VCABM integrator
"""
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

function VCABMOptions(; atol=1e-8, rtol=1e-6, dtini=0.0, dtmax=Inf, qmax=5, 
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
  u_erro::U
  f_prev::U

  ϕ_n::Vector{U}
  ϕ_np1::Vector{U}
  ϕstar_n::Vector{U}
  ϕstar_nm1::Vector{U}

  c::Matrix{T}
  g::Vector{T}
  k::Int
  error_k::T

  function VCABMCache{T}(kmax, u_prev::U, f_prev::U) where {T,U}
    new{T,typeof(u_prev)}(
      u_prev,zero.(u_prev),zero.(u_prev),f_prev,
      [zero.(f_prev) for _ in 1:kmax+1],
      [zero.(f_prev) for _ in 1:kmax+2],
      [zero.(f_prev) for _ in 1:kmax+1],
      [zero.(f_prev) for _ in 1:kmax+1],
      zeros(T,kmax+1,kmax+1),zeros(T,kmax+1),1,zero(T))
  end
end

mutable struct VCABMCacheVolterra{T,V}
  v_next::V
  f_prev::Vector{V}
  g::Vector{T}
  k::Int
  function VCABMCacheVolterra{T}(kmax, f_prev::V) where {T,V}
    new{T, typeof(v_next)}(v_next, [zero.(f_prev) for _ in 1:kmax+1], zeros(T, kmax+1), 1)
  end
end

# Explicit Adams: Section III.5 Eq. (5.5)
function predict!(cache::VCABMCache, times)
  @inbounds begin
    @unpack u_prev,u_next,g,ϕstar_n,k = cache
    ϕ_and_ϕstar!(cache, times, k+1)
    g_coeffs!(cache, times, k+1)
    @. u_next = muladd(g[1], ϕstar_n[1], u_prev)
    for i = 2:k-1
      @. u_next = muladd(g[i], ϕstar_n[i], u_next)
    end
  end
  u_next
end

# Implicit Adams: Section III.5 Eq (5.7)
function correct!(cache::VCABMCache, f)
  @unpack u_next,g,ϕ_np1,ϕstar_n,k = cache
  @inbounds begin
    ϕ_np1!(cache, f(), k+1)
    @. u_next = muladd(g[k], ϕ_np1[k], u_next)
  end
  u_next
end

# Control order: Section III.7 Eq. (7.7)
function adjust!(cache::VCABMCache, times, f, kmax, atol, rtol)
  @inbounds begin
    @unpack u_prev,u_next,g,ϕ_np1,ϕstar_n,k,u_erro = cache

    # Calculate error: Section III.7 Eq. (7.3)
    cache.error_k = norm(g[k+1]-g[k]) * norm(error!(u_erro, ϕ_np1[k+1], u_prev, u_next, atol, rtol))
    
    # Fail step: Section III.7 Eq. (7.4)
    if cache.error_k > one(cache.error_k)
      return
    end

    cache.f_prev = f()

    if length(times)<=5 || k<3
      cache.k = min(k+1, 3, kmax)
    else
      error_k1 = norm(g[k]-g[k-1]) * norm(error!(u_erro, ϕ_np1[k], u_prev, u_next, atol, rtol))
      error_k2 = norm(g[k-1]-g[k-2]) * norm(error!(u_erro, ϕ_np1[k-1], u_prev, u_next, atol, rtol))
      if max(error_k2, error_k1) <= cache.error_k
        cache.k = k-1
      else
        ϕ_np1!(cache, cache.f_prev, k+2)
        error_kstar = norm((times[end] - times[end-1]) * γstar[k+2]) * norm(error!(u_erro, ϕ_np1[k+2], u_prev, u_next, atol, rtol))
        if error_kstar < cache.error_k
          cache.k = min(k+1, kmax)
          cache.error_k = one(cache.error_k)   # constant dt
        end
      end
    end
    @. cache.u_prev = cache.u_next
    cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1    
  end
end

function extend!(cache::VCABMCache, f_vert, times)
  @unpack f_prev, u_prev, u_next, u_erro, ϕ_n, ϕ_np1, ϕstar_n, ϕstar_nm1, k = cache

  t = length(times) - 1 # `t` from the last iteration

  if cache.error_k > one(cache.error_k)
    return
  end
  
  insert!(f_prev.u, t, f_vert(t, t))
  insert!(u_prev.u, t, copy.(u_prev[t]))
  insert!(u_next.u, t, zero.(u_prev[t]))
  insert!(u_erro.u, t, zero.(u_erro[t]))

  _ϕ_n = [zero.(f_prev[t]) for _ in eachindex(ϕ_n)]
  _ϕ_np1 = [zero.(f_prev[t]) for _ in eachindex(ϕ_np1)]
  _ϕstar_n = [zero.(f_prev[t]) for _ in eachindex(ϕstar_n)]
  _ϕstar_nm1 = [zero.(f_prev[t]) for _ in eachindex(ϕstar_nm1)]

  for k′ in 1:k
    ϕ_and_ϕstar!((f_prev=f_vert(max(1,t-1-k+k′), t), ϕ_n=_ϕ_n, ϕstar_n=_ϕstar_n, 
      ϕstar_nm1=_ϕstar_nm1), view(times, 1:(t-k+k′)), k′)
    _ϕstar_nm1, _ϕstar_n = _ϕstar_n, _ϕstar_nm1
  end

  foreach((ϕ, ϕ′) -> insert!(ϕ.u, t, ϕ′), ϕ_n, _ϕ_n)
  foreach((ϕ, ϕ′) -> insert!(ϕ.u, t, ϕ′), ϕ_np1, _ϕ_np1)
  foreach((ϕ, ϕ′) -> insert!(ϕ.u, t, ϕ′), ϕstar_n, _ϕstar_n)
  foreach((ϕ, ϕ′) -> insert!(ϕ.u, t, ϕ′), ϕstar_nm1, _ϕstar_nm1)
end

function predict!(cache::VCABMCacheVolterra, times)
  @unpack f_prev, g, k = cache
  t = reverse(times)
  @inbounds begin
    δ(j, n=1) = j == 0 ? f_prev[n] : (δ(j-1, n) - δ(j-1, n+1))/(t[n+1] - t[n+1+j])
    g[1] = t[1] - t[2]
    cache_volterra.v_next = g[1] * δ(0)
    for i = 2:k-1
      g[i] = g[i-1] * (t[1] - t[1+i])
      @. cache_volterra.v_next = muladd(g[i], δ(i-1), v_next)
    end
  end
  cache_volterra.v_next
end

function correct!(cache::VCABMCacheVolterra, u_next, times)
  @unpack v_next, f_prev, g, k = cache

  f_prev[1] = u_next

  t = reverse(times)
  @inbounds begin
    δ(j, n=1) = j == 0 ? f_prev[n] : (δ(j-1, n) - δ(j-1, n+1))/(t[n+1] - t[n+1+j])
    g[k] = g[k-1] * (t[1] - t[1+k])
    @. v_next = muladd(g[k], δ(k-1), v_next)
  end
  v_next
end

function extend!(cache::VCABMCacheVolterra, times, k_vert, k)
  cache_volterra.k = cache.k

  t = length(times)

  # Extend
  for k in 1:k-1
    insert!(f_prev[k].u, t, integrate(view(times, 1:t-1), [k_vert(s) for s in 1:(t-1)]))
  end

  for k in k:-1:2
    f_prev[k] = f_prev[k-1]
  end

  t = length(times)

  # Update f_prev
  for k in 2:k
    @. f_prev[k] += 0.5 * (times[t-1] - times[t-2]) * (kernel(t-1-k, t-1) + kernel(t-1-k, t-2)) # TODO: beyond trapezoidal
  end
end

# Section III.5: Eq (5.9-5.10)
function ϕ_and_ϕstar!(cache, times, k)
  @unpack f_prev, ϕstar_nm1, ϕ_n, ϕstar_n = cache
  @inbounds begin
    t = reverse(times)
    β = one(eltype(t))
    ϕ_n[1] .= f_prev
    ϕstar_n[1] .= f_prev
    for i = 2:k
      β = β * (t[1] - t[i]) / (t[2] - t[i+1])
      @. ϕ_n[i] = ϕ_n[i-1] - ϕstar_nm1[i-1]
      @. ϕstar_n[i] = β * ϕ_n[i]
    end
  end
end

function ϕ_np1!(cache, du, k)
  @unpack ϕ_np1, ϕstar_n = cache
  @inbounds begin
    @. ϕ_np1[1] = du
    for i = 2:k
      @. ϕ_np1[i] = ϕ_np1[i-1] - ϕstar_n[i-1]
    end
  end
end

function g_coeffs!(cache, times, k)
  @unpack c, g = cache
  @inbounds begin
    t = reverse(times)
    dt = t[1] - t[2]
    for i = 1:k
      for q = 1:k-(i-1)
        if i > 2
          c[i,q] = muladd(-dt/(t[1] - t[i]), c[i-1,q+1], c[i-1,q])
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

# Coefficients for the implicit Adams methods
const γstar = [1,-1/2,-1/12,-1/24,-19/720,-3/160,-863/60480,-275/24192,-33953/3628800,-0.00789255,-0.00678585,-0.00592406,-0.00523669,-0.0046775,-0.00421495,-0.0038269]
