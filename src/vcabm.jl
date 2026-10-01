# Part of the following code is licensed under the MIT "Exact" License,
# from https://github.com/SciML/OrdinaryDiffEq.jl
mutable struct VCABMCache{T,U}
  u_prev::U
  u_next::U
  u_erro::U

  f_pred::U
  # Accepted Newton differences remain unchanged during a trial; ϕ_n[1] = f_n.
  ϕ_n::Vector{U}
  ϕ_np1::Vector{U}
  ϕstar_n::Vector{U}

  c::Vector{T}
  g::Vector{T}
  k::Int
  steps::Int
  error_k::T

  function VCABMCache{T}(kmax, u_prev::U) where {T,U}
    return new{T,U}(u_prev, zero.(u_prev), zero.(u_prev), zero.(u_prev),
                   [zero.(u_prev) for _ in 1:(kmax + 2)], [zero.(u_prev) for _ in 1:(kmax + 2)], [zero.(u_prev) for _ in 1:(kmax + 1)],
                   zeros(T, kmax + 1), zeros(T, kmax + 1), 1, 0, zero(T))
  end
end

# The i-th time difference is known for s ≤ t[n-i]. Complete it with its
# degree-(q-i) interpolant through t[n-i], …, t[n-q]. Diagonal evolution is excluded.
function complete_triangle!(ϕ, times, degree, depth, scratch, fields)
  n = length(times)
  q = min(degree, n - 1)
  columns = length(first(ϕ[1])) - 1
  @inbounds for i in 1:min(q, depth), col in (n - i + 1):columns
    for block in fields
      d = ϕ[i + 1][block]
      d[col] = _recursivecopy!(d[col], d[n - i])
    end
    for j in (i + 1):q
      L = prod((times[col] - times[n - m]) / (times[n - j] - times[n - m]) for m in i:q if m != j; init=one(eltype(times)))
      for block in fields
        d, δ = ϕ[i + 1][block], scratch[block]
        δ[col] = muladd!(δ[col], -1, d[n - i], d[n - j])
        d[col] = muladd!(d[col], L, δ[col], d[col])
      end
    end
  end
  return ϕ
end

# Explicit Adams: Section III.5 Eq. (5.5)
function predict!(cache::VCABMCache, times; n2=0)
  (; u_prev, u_next, g, ϕstar_n, k) = cache
  @inbounds begin
    ϕstar!(cache, times, min(k + 1, cache.steps + 1))
    if n2 > 0
      complete_triangle!(ϕstar_n, view(times, 1:(length(times) - 1)), k - 1, cache.steps, cache.u_erro, 1:n2)
    end
    g_coeffs!(cache, times, min(k + 2, length(times), length(g)))
    muladd!(u_next, g[1], ϕstar_n[1], u_prev)
    for i in 2:k
      muladd!(u_next, g[i], ϕstar_n[i], u_next)
    end
  end
  return u_next
end

# Adams–Moulton evaluated at the predictor (one PECE correction): Section III.5 Eq (5.7)
function correct!(cache::VCABMCache, f!)
  (; u_next, g, ϕ_np1, k) = cache
  f!(ϕ_np1[1])
  ϕ_np1!(cache, k + 1)
  muladd!(u_next, g[max(k, 2)], ϕ_np1[k + 1], u_next)
  return u_next
end

# E_j = A_{j+1}(f_c) - A_j(f_p).
function adams_error!(cache::VCABMCache, j, atol, rtol; n2=0)
  (; u_erro, u_prev, u_next, ϕ_np1, ϕstar_n, g) = cache
  f_p = cache.f_pred
  muladd!(u_erro, -1, f_p, ϕ_np1[1])
  muladd!(u_erro, g[j], u_erro, 0)
  muladd!(u_erro, g[j + 1] - g[j], ϕ_np1[j + 1], u_erro)
  if n2 > 0
    n = length(first(u_prev))
    # D_m = ϕ_np1[m] - ϕ_np1[m-1] + ϕstar_n[m-1] vanishes outside reconstructed columns.
    # Their contribution to E_j is ∑ (g[j] - g[m-1]) D_m.
    for m in 3:j, r in 1:(m - 2), block in 1:n2
      α = g[j] - g[m - 1]
      E = u_erro[block]
      E[n - r] = muladd!(E[n - r], α, ϕ_np1[m][block][n - r], E[n - r])
      E[n - r] = muladd!(E[n - r], -α, ϕ_np1[m - 1][block][n - r], E[n - r])
      E[n - r] = muladd!(E[n - r], α, ϕstar_n[m - 1][block][n - r], E[n - r])
    end
  end
  calculate_residuals!(u_erro, u_erro, u_prev, u_next, atol, rtol)
  return state_norm(u_erro)
end

# Control order: Section III.7 Eq. (7.7)
function adjust!(cache::VCABMCache, times, f!, kmax, atol, rtol; n2=0)
  (; g, k) = cache
  @inbounds begin
    cache.f_pred, cache.ϕ_np1[1] = cache.ϕ_np1[1], cache.f_pred
    f!(cache.ϕ_np1[1])
    ϕ_np1!(cache, min(k + 2, cache.steps + 2))
    if n2 > 0
      complete_triangle!(cache.ϕ_np1, times, k + 1, cache.steps + 1, cache.u_erro, 1:n2)
    end

    # Calculate error: Section III.7 Eq. (7.3)
    if k == 1
      muladd!(cache.u_erro, -1, cache.ϕstar_n[1], cache.f_pred)
      calculate_residuals!(cache.u_erro, cache.u_erro, cache.u_prev, cache.u_next, atol, rtol)
      cache.error_k = abs(g[2] - g[1]) * state_norm(cache.u_erro)
    else
      cache.error_k = adams_error!(cache, k, atol, rtol; n2)
    end
    isnan(cache.error_k) && error("Local error is NaN at t = $(last(times)); the trial state or right-hand side is undefined.")
    cache.error_k > one(cache.error_k) && return

    if cache.steps < 4 || k < 3
      cache.k = min(k + 1, 3, kmax)
    else
      error_k1 = adams_error!(cache, k - 1, atol, rtol; n2)
      error_k2 = adams_error!(cache, k - 2, atol, rtol; n2)
      if max(error_k2, error_k1) <= cache.error_k
        cache.k = k - 1
      elseif k < kmax && cache.steps >= k
        calculate_residuals!(cache.u_erro, cache.ϕ_np1[k + 2], cache.u_prev, cache.u_next, atol, rtol)
        error_kstar = abs(g[k + 2] - g[k + 1]) * state_norm(cache.u_erro)
        if error_kstar < cache.error_k
          cache.k = k + 1
          cache.error_k = one(cache.error_k)   # Limit step growth when increasing order.
        end
      end
    end
    cache.steps += 1
    cache.u_prev, cache.u_next = cache.u_next, cache.u_prev
    cache.ϕ_n, cache.ϕ_np1 = cache.ϕ_np1, cache.ϕ_n
  end
end

function extend!(cache::VCABMCache, times, fv!, n2)
  (; u_prev, u_next, u_erro, f_pred, ϕ_n, ϕ_np1, ϕstar_n, error_k) = cache
  @inbounds begin
    t = length(times) - 1
    error_k > one(error_k) && return

    for i in 1:n2
      insert!(f_pred[i], t, zero(u_prev[i][t]))
      for history in (ϕ_n, ϕ_np1, ϕstar_n), ϕ in history
        insert!(ϕ[i], t, zero(u_prev[i][t]))
      end
      insert!(u_prev[i], t, copy(u_prev[i][t]))
      insert!(u_next[i], t, zero(u_prev[i][t]))
      insert!(u_erro[i], t, zero(u_prev[i][t]))
    end
    fv!(ϕ_n[1], t, t)
  end
end

# Scale the accepted endpoint differences to the proposed step: Section III.5 Eq. (5.10)
function ϕstar!(cache, times, k)
  (; ϕ_n, ϕstar_n) = cache
  @inbounds begin
    t = @view times[end:-1:1]
    β = one(eltype(times))
    _recursivecopy!(ϕstar_n[1], ϕ_n[1])
    for i in 2:k
      β = β * (t[1] - t[i]) / (t[2] - t[i + 1])
      muladd!(ϕstar_n[i], β, ϕ_n[i], 0)
    end
  end
end

function g_coeffs!(cache, times, k)
  (; c, g) = cache
  @inbounds begin
    t = @view times[end:-1:1]
    dt = t[1] - t[2]
    for i in 1:k
      # Ascending q leaves c[q + 1] in the preceding recurrence row.
      for q in 1:(k - (i - 1))
        if i > 2
          c[q] = muladd(-dt / (t[1] - t[i]), c[q + 1], c[q])
        elseif i == 1
          c[q] = one(dt) / q
        elseif i == 2
          c[q] = one(dt) / (q * (q + 1))
        end
      end
      g[i] = c[1] * dt
    end
  end
end

function ϕ_np1!(cache, k)
  (; ϕ_np1, ϕstar_n) = cache
  @inbounds for i in 2:k
    muladd!(ϕ_np1[i], -1, ϕstar_n[i - 1], ϕ_np1[i - 1])
  end
end
