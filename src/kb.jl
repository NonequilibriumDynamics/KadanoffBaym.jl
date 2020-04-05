"""
  Kadanoff-Baym adaptive timestepper

Solves the 2-time Voltera integral differential equations
  dy/dt1 = F[y](t1,t2)
  dy/dt2 = G[y](t1,t2)

The signature of `f_vert` and `f_diag` must be of the form
  f(u, t_grid, t1, t2)

It is also mandatory that u ≡ u₀ can be indexed by 2 time arguments and can be
resized at will
  uᵢⱼ = u[i,j]
  u_augmented = resize!(u, new_size...)

The Kadanoff-Baym timestepper is based on the VCABM stepper presented in
Ernst Hairer, Gerhard Wanner, and Syvert P Norsett
Solving Ordinary Differential Equations I: Nonstiff Problems
"""
function kbsolve(f_vert, f_diag, u, t₀, tmax; dt=nothing, adaptive=true, 
  max_dt=1e-1, atol=1e-9, rtol=1e-7, max_order=12, qmax=5, qmin=1//5, γ=9//10)
  
  if max_order < 1 || max_order > 12
    error("max_order must be between 1 and 12")
  end

  if t₀ > tmax
    error("Only t₀ < tmax supported")
  end

  # Resize solution
  resize!.(u, 50, 50)

  # Initialize state, caches and determine dt
  state, caches = begin
    # Extract solution at (t₀,t₀)
    u₀ = [x[1,1,..] for x in u]

    f₀_vert = f_vert(u,[t₀],1,1)
    cache_vert = VCABMCache{typeof(t₀)}(max_order,u₀,f₀_vert)

    f₀_diag = f_diag(u,[t₀],1,1)
    cache_diag = VCABMCache{typeof(t₀)}(max_order,u₀,f₀_diag)

    if dt === nothing
      # initial_step will call a function at t=(2,1)
      f′ = (u′,t′) -> begin
        foreach(i->u[i][2,1,..]=u′[i],1:length(u))
        f_vert(u,[t₀,t′],2,1)
      end
      dt = initial_step(f′,u₀,t₀,1,rtol,atol;f₀=f₀_vert)
    end

    VCABMState(u,t₀,dt), KBCaches(cache_vert, cache_diag)    
  end

  # Modify user functions to always refer to the `state` variables `u` and `t`
  # This is necessary because internally `f` is a function of `uᵢ` and `tᵢ`
  inplace!(f,tᵢ,tⱼ) = (u′,_) -> begin
    foreach(i->state.u[i][tᵢ,tⱼ,..]=u′[i],1:length(u))
    f(state.u,state.t,tᵢ,tⱼ)
  end

  # Time-step
  @do_while timeloop!(state,caches.diag,tmax,max_dt,adaptive,qmax,qmin,γ) begin
    T = length(state.t) # current time index

    # Step vertically
    for (tⱼ, cache) in enumerate(caches.vert)
      predict_correct!(inplace!(f_vert,T,tⱼ), state, cache, max_order, 
        atol, rtol, true)#; update=uⱼ->(state.u[T,tⱼ,:] = uⱼ))
    end

    # Step diagonally and control step
    predict_correct!(inplace!(f_diag,T,T), state, caches.diag, max_order, 
      atol, rtol, true)

    # Accept step
    if caches.diag.error_k <= one(caches.diag.error_k)
      update_caches!(caches, state, f_vert, max_order)
    end

    # Resize solution
    if size(state.u[1],1) == length(state.t)
      size_hint = size(state.u[1],1) + min(max(ceil(Int,(tmax-state.t[end])/state.dt),5),50)
      resize!.(state.u,size_hint,size_hint)
    end
  end

  # Trim solution
  resize!.(state.u, length(state.t), length(state.t))
  return state.u, state.t
end

function update_caches!(caches, state::VCABMState, f_vert, max_k)
  @assert length(caches.vert)+1 == length(state.t)

  T = length(state.t)
  # local_max_k = caches.diag.k

  # for (tⱼ, cache) in enumerate(caches.vert)
  #   cache.u_prev = state.u[T,tⱼ,:] # u_next was saved in state.u by update
  #   cache.f_prev = f_vert(state.u,state.t,T,tⱼ)
  #   cache.ϕstar_nm1, cache.ϕstar_n = cache.ϕstar_n, cache.ϕstar_nm1
  #   cache.k = min(local_max_k, cache.k+1) # Ramp up order
  # end

  begin # Add a new cache!
    u₀ = [x[T,T,..] for x in state.u]
    f₀ = f_vert(state.u,state.t,T,T)
    push!(caches.vert, VCABMCache{eltype(state.t)}(max_k, u₀, f₀))
  end

  @assert length(caches.vert) == length(state.t)
end

mutable struct KBCaches{T,F}
  vert::Vector{VCABMCache{T,F}}
  diag::VCABMCache{T,F}

  function KBCaches(vert1::VCABMCache{T,F}, diag::VCABMCache{T,F}) where {T,F}
    new{T,F}([vert1,], diag)
  end
end