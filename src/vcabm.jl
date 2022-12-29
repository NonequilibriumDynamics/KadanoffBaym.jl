# Part of the following code is licensed under the MIT "Expact" Lience, 
# from https://github.com/SciML/OrdinaryDiffEq.jl
mutable struct VCABMCache{T,U} <: OrdinaryDiffEq.OrdinaryDiffEqMutableCache
  u_prev::U
  u_next::U
  u_erro::U

  f_prev::U
  f_next::U

  ϕ_n::Vector{U}
  ϕ_np1::Vector{U}
  ϕstar_n::Vector{U}
  ϕstar_nm1::Vector{U}

  c::Matrix{T}
  g::Vector{T}
  k::Int
  error_k::T

  β::Vector{T}
  dts::Vector{T}
  ξ::T 
  ξ0::T

  function VCABMCache{T}(kmax, u_prev::U) where {T,U}
    return new{T,typeof(u_prev)}(u_prev, zero.(u_prev), zero.(u_prev), zero.(u_prev), zero.(u_prev), [zero.(u_prev) for _ in 1:(kmax + 1)],
                                 [zero.(u_prev) for _ in 1:(kmax + 2)], [zero.(u_prev) for _ in 1:(kmax + 1)], [zero.(u_prev) for _ in 1:(kmax + 1)],
                                 zeros(T, kmax + 1, kmax + 1), zeros(T, kmax + 1), 1, zero(T), zeros(T, kmax + 1), zeros(T, kmax + 1), zero(T), zero(T))
  end
end

# Explicit Adams: Section III.5 Eq. (5.5)
function predict!(cache::VCABMCache, times; update_dt = true)
  (; u_prev, u_next, g, ϕstar_n, k) = cache
  @inbounds begin
    if update_dt && cache.error_k <= one(cache.error_k)
      for i = (length(cache.dts) - 1):-1:1
        cache.dts[i+1] = cache.dts[i]
      end
      cache.dts[1] = times[end] - times[end - 1]
    end

    OrdinaryDiffEq.ϕ_and_ϕstar!(cache, cache.f_prev, k)
    OrdinaryDiffEq.g_coefs!(cache, k + 1)
    @. u_next = muladd(g[1], ϕstar_n[1], u_prev)
    for i in 2:(k - 1)
      @. u_next = muladd(g[i], ϕstar_n[i], u_next)
    end
  end
  return u_next
end

# Implicit Adams: Section III.5 Eq (5.7)
function correct!(cache::VCABMCache, f)
  (; u_next, g, ϕ_np1, ϕstar_n, k) = cache
  @inbounds begin
    OrdinaryDiffEq.ϕ_np1!(cache, f(), k + 1)
    @. u_next = muladd(g[k], ϕ_np1[k], u_next)
  end
  return u_next
end

# Control order: Section III.7 Eq. (7.7)
function adjust!(cache::VCABMCache, cache_v, times, f2, f1, kmax, atol, rtol)
  (; u_prev, u_next, g, ϕ_np1, ϕstar_n, k, u_erro) = cache
  @inbounds begin
    # Calculate error: Section III.7 Eq. (7.3)
    OrdinaryDiffEq.calculate_residuals!(u_erro, ϕ_np1[k + 1], u_prev, u_next, atol, rtol, norm, nothing)
    cache.error_k = norm(g[k + 1] - g[k]) * norm(u_erro)

    # Fail step: Section III.7 Eq. (7.4)
    if cache.error_k > one(cache.error_k)
      return
    end

    cache.f_prev .= f2()
    if !isnothing(cache_v)
      cache_v.f_prev .= f1()
    end

    if length(times) <= 5 || k < 3
      cache.k = min(k + 1, 3, kmax)
    else
      OrdinaryDiffEq.calculate_residuals!(u_erro, ϕ_np1[k], u_prev, u_next, atol, rtol, norm, nothing)
      error_k1 = norm(g[k] - g[k - 1]) * norm(u_erro)
      OrdinaryDiffEq.calculate_residuals!(u_erro, ϕ_np1[k - 1], u_prev, u_next, atol, rtol, norm, nothing)
      error_k2 = norm(g[k - 1] - g[k - 2]) * norm(u_erro)

      if max(error_k2, error_k1) <= cache.error_k
        cache.k = k - 1
      else
        OrdinaryDiffEq.expand_ϕ_and_ϕstar!(cache, k+1)
        OrdinaryDiffEq.ϕ_np1!(cache, cache.f_prev, k + 2)

        if !isnothing(cache_v)
          OrdinaryDiffEq.expand_ϕ_and_ϕstar!(cache_v, k+1)
          OrdinaryDiffEq.ϕ_np1!(cache_v, cache_v.f_prev, k + 2)
        end
        OrdinaryDiffEq.calculate_residuals!(u_erro, ϕ_np1[k + 2], u_prev, u_next, atol, rtol, norm, nothing)
        error_kstar = norm((times[end] - times[end - 1]) * OrdinaryDiffEq.γstar[k + 2]) * norm(u_erro)
        if error_kstar < cache.error_k
          cache.k = min(k + 1, kmax)
          cache.error_k = one(cache.error_k)   # constant dt
        end
      end
    end
    @. cache.u_prev = cache.u_next
    cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1

    if !isnothing(cache_v)
      cache_v.k = cache.k
      @. cache_v.u_prev = cache_v.u_next
      cache_v.ϕstar_nm1, cache_v.ϕstar_n = cache_v.ϕstar_n, cache_v.ϕstar_nm1
    end
  end
end

function extend!(cache::VCABMCache, state, fv!)
  (; f_prev, f_next, u_prev, u_next, u_erro, ϕ_n, ϕ_np1, ϕstar_n, ϕstar_nm1, error_k) = cache
  @inbounds begin
    t = length(state.t) - 1 # `t` from the last iteration
    k = isone(cache.k) ? 0 : state.w.ks[end-1] # `k` from the last iteration

    if error_k > one(error_k)
      return
    end

    _ϕ_n = [zero.(f_next[t, :]) for _ in eachindex(ϕ_n)]
    _ϕ_np1 = [zero.(f_next[t, :]) for _ in eachindex(ϕ_np1)]
    _ϕstar_n = [zero.(f_next[t, :]) for _ in eachindex(ϕstar_n)]
    _ϕstar_nm1 = [zero.(f_next[t, :]) for _ in eachindex(ϕstar_nm1)]

    # When extending, i.e., adding a new time column, an interpolant for the rhs in this column
    # has to be built. Since sometimes fv! is discontinuous at the diagonal it is assumed that 
    # fv! is built always assuming t1 > t2. This way, the interpolant (which requires points t1 < t2)
    # is smooth and the solver does not stall.
    for k′ in 1:k
      fv!(max(1, t - 1 - k + k′), t) # result is stored in f_next
      OrdinaryDiffEq.ϕ_and_ϕstar!((dts=view(cache.dts,1+(k-k′):13), ϕstar_nm1=_ϕstar_nm1, ϕ_n=_ϕ_n, ϕstar_n=_ϕstar_n, β = cache.β), f_next[t, :], k′)
      _ϕstar_nm1, _ϕstar_n = _ϕstar_n, _ϕstar_nm1
    end

    for i in eachindex(f_prev.u)
      foreach((ϕ, ϕ′) -> insert!(ϕ.u[i], t, ϕ′[i]), ϕ_n, _ϕ_n)
      foreach((ϕ, ϕ′) -> insert!(ϕ.u[i], t, ϕ′[i]), ϕ_np1, _ϕ_np1)
      foreach((ϕ, ϕ′) -> insert!(ϕ.u[i], t, ϕ′[i]), ϕstar_n, _ϕstar_n)
      foreach((ϕ, ϕ′) -> insert!(ϕ.u[i], t, ϕ′[i]), ϕstar_nm1, _ϕstar_nm1)
    end

    fv!(t, t) # result is stored in f_next
    for i in eachindex(u_prev.u)
      insert!(f_prev.u[i], t, copy(f_next[t, i]))
      insert!(f_next.u[i], t, zero(f_next[t, i]))
      insert!(u_prev.u[i], t, copy(u_prev[t, i]))
      insert!(u_next.u[i], t, zero(u_prev[t, i]))
      insert!(u_erro.u[i], t, zero(u_prev[t, i]))
    end
  end
end
