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

# function f(t1, t2, predict::Bool)
#   kernel(t) = isequal(t1, t2) ? t -> kernel_diag(state.t,t1,t) : t -> kernel_vert(state.t,t1,t2,t)
  
#   if predict
#     v′ = volterra_predict(kernel, state, caches.master)
#   else
#     v_next = [v[t,t′] for x in v]
#     v′ = volterra_correct(v_next, kernel, caches.master)
#   end

#   foreach((v,v′) -> v[t,t′] = v′, v, v′)
#   return isequal(t1, t2) ? f_diag(state.u..., state.t, t1) : f_vert(state.u..., state.t, t1, t2)
# end
