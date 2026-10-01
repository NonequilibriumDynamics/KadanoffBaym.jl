# Run with julia --project benchmarks/solver.jl [minimum_time_steps]; use 0 for adaptive meshes.
using KadanoffBaym, LinearAlgebra

function level(n)
  G = GreenFunction(fill(-1.0im, 1, 1), SkewHermitian)
  fv!(out, ts, w1, w2, i, j) = (out[1] = -im * G[i, j]; nothing)
  fd!(out, ts, w1, w2, i, j) = (out[1] = 0; nothing)
  sol = kbsolve!(fv!, fd!, [G], (0.0, 10.0); atol=1e-8, rtol=1e-8, dtini=0.001, dtmax=10 / n)
  return (steps=length(sol.t) - 1, error=abs(G[length(sol.t), 1] + im * exp(-10im)))
end

function dimer(n)
  G0 = ComplexF64[im 0; 0 0]
  G = GreenFunction(reshape(copy(G0), 2, 2, 1, 1), SkewHermitian)
  H = ComplexF64[0.05 1; 1 -0.05]
  fv!(out, ts, w1, w2, i, j) = (out[1] = -im * H * G[i, j]; nothing)
  function fd!(out, ts, w1, w2, i, j)
    fv!(out, ts, w1, w2, i, j)
    out[1] .-= adjoint(out[1])
    nothing
  end
  sol = kbsolve!(fv!, fd!, [G], (0.0, 10.0); atol=1e-8, rtol=1e-8, dtini=0.001, dtmax=10 / n)
  return (steps=length(sol.t) - 1, error=norm(G[length(sol.t), 1] - exp(-10im * H) * G0))
end

function volterra(n; quadrature=:smooth)
  G = GreenFunction(ones(1, 1), Symmetrical)
  function fv!(out, ts, w1, w2, i, j)
    out[1] = 1 - sum(w1[s] * G[s, j] for s in eachindex(w1)) + sum(w2[s] * G[s, j] for s in eachindex(w2))
    nothing
  end
  fd!(out, ts, w1, w2, i, j) = (out[1] = 0; nothing)
  sol = kbsolve!(fv!, fd!, [G], (0.0, 4.0); atol=1e-8, rtol=1e-8, dtini=0.001, dtmax=4 / n, quadrature)
  error = maximum(abs(G[i, j] - cos(t - s) - sin(abs(t - s))) for (i, t) in enumerate(sol.t), (j, s) in enumerate(sol.t))
  return (steps=length(sol.t) - 1, error=error)
end

volterra_piecewise(n) = volterra(n; quadrature=:piecewise)

function ornstein_uhlenbeck(n)
  G = GreenFunction(ones(1, 1), Symmetrical)
  fv!(out, ts, w1, w2, i, j) = (out[1] = -G[i, j]; nothing)
  fd!(out, ts, w1, w2, i, j) = (out[1] = 0; nothing)
  sol = kbsolve!(fv!, fd!, [G], (0.0, 1.0); atol=1e-7, rtol=1e-7, dtmax=1 / n)
  error = maximum(abs(G[i, j] - exp(-abs(t - s))) for (i, t) in enumerate(sol.t), (j, s) in enumerate(sol.t))
  return (steps=length(sol.t) - 1, error=error)
end

n = isempty(ARGS) ? 400 : parse(Int, only(ARGS))
for solve in (level, dimer, volterra, volterra_piecewise, ornstein_uhlenbeck)
  solve(n)
  runs = [(@timed solve(n)) for _ in 1:5]
  println((problem=nameof(solve), threads=Threads.nthreads(),
           seconds=sort([r.time for r in runs])[3], bytes=minimum(r.bytes for r in runs),
           last(runs).value...))
end
