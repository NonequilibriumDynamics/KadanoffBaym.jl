struct Volterra∫{F<:Function}
  i::Int
  j::Int
  K::F
end

function (∫::Volterra∫)(state, master_cache)
  # Determine boundaries of the integral
  t0, tN = 1, length(state.t) # min(∫.i,∫.j), max(∫.i,∫.j)
  
  # Initialize cache
  f0 = ∫.K(t0)
  u0 = zero(f0)
  cache = VCABMCache{eltype(state.t)}(length(master_cache.ϕ_n)-1, u0, f0)

  # mutation tricks
  times = copy(state.t)

  for ti ∈ t0:tN
    state.t = times[1:ti]
    cache.u_prev = predict!(state, cache)
    cache.f_prev = ∫.K(ti)
    cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1
    cache.k = min(master_cache.k, cache.k+1)
  end

  return cache.u_prev
end