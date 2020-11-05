function volterra_predict(kernel, state, cache)  
  # Initialize cache
  f0 = kernel(1)
  u0 = zero(f0)
  cache = VCABMCache{eltype(state.t)}(length(cache.ϕ_n)-1, u0, f0)

  # mutation tricks
  times = copy(state.t)

  for ti in 1:length(state.t)
    state.t = times[1:ti]
    cache.u_prev = predict!(state, cache)
    cache.f_prev = kernel(ti)
    cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1
    cache.k = min(cache.k, cache.k+1)
  end

  return cache.u_prev
end

function volterra_correct(v_next, kernel, cache)
  # mutation tricks
  times = copy(state.t)

  for ti in 1:length(state.t)
    state.t = times[1:ti]
  end
end