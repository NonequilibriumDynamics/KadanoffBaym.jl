# Run: julia --project --threads=1 benchmarks/interaction_picture.jl [samples=3]
# Constant-H interaction picture: G(t,s) = P(t) X(t,s) P(s)', P(t) = exp(-im*H0*t).
# Related KB propagation: https://arxiv.org/html/0906.1704v1#S4
# This experiment uses t0=0 and a common constant H0 for both spins.
# Errors concern stored physical values, not interpolation between them.
include(joinpath(@__DIR__, "quantum.jl"))

function frame_problem(factory; shift=0.0, hopping=1.0, frame=:none)
  model = factory()
  model.H .*= hopping
  for i in axes(model.H, 1)
    model.H[i, i] += shift
  end
  H = copy(model.H)
  H0 = frame == :full ? H : shift * Matrix{ComplexF64}(I, size(H)...)
  frame == :none && return (; model..., physical=model.G, H0=zeros(ComplexF64, size(H)))
  # Evaluate the physical remainder directly, without cancellation against H0*G.
  model.H .-= H0
  X = copy.(model.G)
  P = [Matrix{ComplexF64}(I, size(H)...)]
  spectrum = eigen(Hermitian(H0))
  function callback(ts, w1, w2, i, j)
    if j == 1
      foreach(g -> resize!(g, size(first(X), 3)), model.G)
      Pi = spectrum.vectors * Diagonal(cis.(-spectrum.values .* ts[i])) * spectrum.vectors'
      if length(P) < i
        push!(P, Pi)
      else
        P[i] = Pi
      end
      # Onsite interaction formulas apply in the physical site basis.
      for (g, x) in zip(model.G, X), k in 1:i
        g[i, k] = P[i] * x[i, k] * P[k]'
      end
    end
    model.callback(ts, w1, w2, i, j)
    nothing
  end
  function fv!(out, ts, w1, w2, i, j)
    model.fv!(out, ts, w1, w2, i, j)
    for a in eachindex(X)
      out[a] = P[i]' * out[a] * P[j]
    end
    nothing
  end
  function fd!(out, ts, w1, w2, i, j)
    fv!(out, ts, w1, w2, i, j)
    out .-= adjoint.(out)
    nothing
  end
  return (; G=X, physical=model.G, H, H0, fv!, fd!, callback)
end

function solve_frame(factory; frame=:none, shift=0.0, hopping=1.0, tol=1e-7)
  p = frame_problem(factory; shift, hopping, frame)
  sol = kbsolve!(p.fv!, p.fd!, p.G, (0.0, 1.0); p.callback,
                 atol=tol, rtol=tol, dtini=0.001, dtmax=0.2)
  foreach(g -> resize!(g, length(sol.t)), p.physical)
  return (; p..., t=sol.t)
end

function free_error(sol)
  P = [exp(-im * sol.H * t) for t in sol.t]
  maximum(norm(g[i,j]-P[i]*g[1,1]*P[j]') for g in sol.physical
          for i in eachindex(sol.t) for j in 1:i)
end

function frame_reference(ts; shift, hopping, factor)
  refined = [first(ts); [t for k in 1:length(ts)-1
                        for t in range(ts[k], ts[k+1]; length=factor+1)[2:end]]]
  @assert refined[1:factor:end] == ts
  p = frame_problem(quantum_born; shift, hopping, frame=:full)
  reference_trapezoid!(p.fv!, p.fd!, p.G, refined; p.callback, atol=1e-13)
  return [g.data[:, :, 1:factor:end, 1:factor:end] for g in p.physical]
end

function interaction_benchmarks(samples=3)
  @assert samples >= 1
  BLAS.set_num_threads(1)
  println("Adaptive Adams with default quadrature; errors cover all stored two-time matrix blocks.")
  println("Richardson pair gaps diagnose reference convergence; they are not certified error bounds.")
  for frame in (:none, :full)
    sol = solve_frame(quantum_dimer; shift=10.0, frame)
    error = free_error(sol)
    frame == :full && @assert error < 1e-12
    println((problem=:free_dimer, frame, steps=length(sol.t)-1, error))
  end
  flush(stdout)

  configs = ((; hopping=1.0, shift=5.0, frame=:none, tol=3e-6),
             (; hopping=1.0, shift=5.0, frame=:carrier, tol=1e-5),
             (; hopping=5.0, shift=0.0, frame=:none, tol=3e-6),
             (; hopping=5.0, shift=0.0, frame=:full, tol=3e-6))
  for config in configs
    solve_frame(quantum_born; config...)
    runs = map(1:samples) do _
      GC.gc()
      @timed solve_frame(quantum_born; config...)
    end
    sol = last(runs).value
    coarse, medium, fine = [frame_reference(sol.t; config.shift, config.hopping, factor)
                            for factor in (2, 4, 8)]
    high = [GreenFunction((4 .* f .- m) ./ 3, SkewHermitian) for (m, f) in zip(medium, fine)]
    low = [GreenFunction((4 .* m .- c) ./ 3, SkewHermitian) for (c, m) in zip(coarse, medium)]
    error = quantum_reference_error(sol.physical, high)
    reference_gap = quantum_reference_error(low, high)
    @assert error < 1e-4 && reference_gap < 1e-6
    println((problem=:second_born, config..., steps=length(sol.t)-1,
             seconds=sort([r.time for r in runs])[(samples+1)÷2],
             bytes=minimum(r.bytes for r in runs), error, reference_gap,
             quantum_invariants(sol.physical, sol.t)...))
    flush(stdout)
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  interaction_benchmarks(isempty(ARGS) ? 3 : parse(Int, ARGS[1]))
end
