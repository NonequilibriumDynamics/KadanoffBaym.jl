# Run: julia --project --threads=1 benchmarks/quantum.jl [largest_production_mesh=64] [samples=3]
# Equations: examples/fermionic-dimer.ipynb and examples/fermi-hubbard-2B.ipynb.
# The latter uses two sites, constant U=0.25, and the first two listed occupations.
# For the stated +U*n↑*n↓ interaction and G< = im*ρ, ΣHF = -im*U*G< = U*ρ.
# This corrects the opposite Hartree sign in the original notebook.
using KadanoffBaym, LinearAlgebra
include(joinpath(@__DIR__, "..", "test", "reference.jl"))
include(joinpath(@__DIR__, "..", "test", "lobatto_reference.jl"))

function quantum_weighted_callbacks(equations)
  function integral(ts, w1, w2, i, j)
    function integrate(K, a, b; sample_domain=min(a,b):max(a,b), init)
      w = a == 1 && b == i ? w1 : a == 1 && b == j ? w2 :
          throw(ArgumentError("weighted quantum callbacks only provide integrals from 1 to i or j"))
      value = KadanoffBaym.recursivecopy(init)
      a == b && return value
      for k in eachindex(w)
        iszero(w[k]) || (k in sample_domain && k <= length(ts)) ||
          throw(ArgumentError("quadrature samples node $k outside the smooth sampling domain"))
      end
      for k in eachindex(w)
        iszero(w[k]) || (value = KadanoffBaym.muladd!(value, w[k], K(k), value))
      end
      return value
    end
  end
  return (
    fv! = (out,ts,w1,w2,i,j) -> equations.fv!(out,ts,integral(ts,w1,w2,i,j),i,j),
    fd! = (out,ts,w1,w2,i,j) -> equations.fd!(out,ts,integral(ts,w1,w2,i,j),i,j),
    callback = (ts,w1,w2,i,j) -> equations.callback(ts,integral(ts,w1,w2,i,j),i,j),
  )
end

function quantum_dimer()
  H = ComplexF64[0.05 1; 1 -0.05]
  GL = GreenFunction(reshape(ComplexF64[im 0; 0 0], 2, 2, 1, 1), SkewHermitian)
  GG = GreenFunction(reshape(ComplexF64[0 0; 0 -im], 2, 2, 1, 1), SkewHermitian)
  function fv!(out, ts, integrate, i, j)
    out[1] = -im * H * GL[i, j]
    out[2] = -im * H * GG[i, j]
    nothing
  end
  function fd!(out, ts, integrate, i, j)
    fv!(out, ts, integrate, i, j)
    out .-= adjoint.(out)
    nothing
  end
  equations = (; fv!, fd!, f1! = nothing, callback=(args...) -> nothing)
  return (; G=[GL, GG], H, equations, quantum_weighted_callbacks(equations)...)
end

function quantum_born()
  H = ComplexF64[0 -1; -1 0]
  U = 0.25
  ρu, ρd = Diagonal([0.7, 0.0]), Diagonal([0.0, 0.25])
  G = [GreenFunction(reshape(Matrix{ComplexF64}(g), 2, 2, 1, 1), SkewHermitian)
       for g in (im * ρu, -im * (I - ρu), im * ρd, -im * (I - ρd))]
  GLu, GGu, GLd, GGd = G
  ΣLu, ΣGu, ΣLd, ΣGd = Σ = [zero(g) for g in G]

  function callback(ts, integrate, i, j)
    if size(GLu, 3) > size(ΣLu, 3)
      foreach(s -> resize!(s, size(GLu, 3)), Σ)
    end
    ΣLu[i, j] = @views U^2 .* GLu.data[:, :, i, j] .* GLd.data[:, :, i, j] .* transpose(GGd.data[:, :, j, i])
    ΣLd[i, j] = @views U^2 .* GLu.data[:, :, i, j] .* GLd.data[:, :, i, j] .* transpose(GGu.data[:, :, j, i])
    ΣGu[i, j] = @views U^2 .* GGu.data[:, :, i, j] .* GGd.data[:, :, i, j] .* transpose(GLd.data[:, :, j, i])
    ΣGd[i, j] = @views U^2 .* GGu.data[:, :, i, j] .* GGd.data[:, :, i, j] .* transpose(GLu.data[:, :, j, i])
    nothing
  end

  function fv!(out, ts, integrate, i, j)
    for (spin, (GL, GG, ΣL, ΣG, GLother)) in enumerate(((GLu, GGu, ΣLu, ΣGu, GLd),
                                                       (GLd, GGd, ΣLd, ΣGd, GLu)))
      H_HF = H - im * U * Diagonal(GLother[i, i])
      for (a, (g, σ)) in enumerate(((GL, ΣL), (GG, ΣG)))
        collision = integrate(k -> @views((ΣG.data[:,:,i,k] - ΣL.data[:,:,i,k]) * g.data[:,:,k,j]), 1, i;
                              sample_domain=eachindex(ts), init=zero(g[i,j]))
        collision = integrate(k -> @views(σ.data[:,:,i,k] * (GL.data[:,:,k,j] - GG.data[:,:,k,j])), 1, j;
                              sample_domain=eachindex(ts), init=collision)
        out[2spin - 2 + a] = -im * (H_HF * g[i, j] + collision)
      end
    end
    nothing
  end
  function fd!(out, ts, integrate, i, j)
    fv!(out, ts, integrate, i, j)
    out .-= adjoint.(out)
    nothing
  end
  callback([0.0], nothing, 1, 1)
  equations = (; fv!, fd!, f1! = nothing, callback)
  return (; G, H, equations, quantum_weighted_callbacks(equations)...)
end

function quantum_solve(problem, method, n)
  (; G, equations, fv!, fd!, callback) = problem()
  ts = collect(range(0.0, 1.0; length=n + 1))
  if method == :lobatto
    @assert iseven(n)
    sol = reference_lobatto!(equations, G, range(0.0, 1.0; length=n ÷ 2 + 1); atol=1e-13)
    @assert sol.t == ts
  elseif method == :trapezoid
    sol = reference_trapezoid!(equations, G, ts; atol=1e-13)
  else
    # Large tolerances and a dyadic step cap isolate fixed-mesh convergence.
    sol = kbsolve!(fv!, fd!, G, (0.0, 1.0); callback, quadrature=method,
                   dtini=1 / n, dtmax=1 / n, γ=1, atol=1e2, rtol=1e2)
    @assert sol.t == ts
  end
  return (; G, t=sol.t)
end

function quantum_invariants(G, t)
  N0 = [imag(tr(G[a][1, 1])) for a in 1:2:length(G)]
  particle = maximum(abs(sum(imag(tr(G[a][i, i])) for a in 1:2:length(G)) - sum(N0))
                     for i in eachindex(t))
  spin = maximum(abs(imag(tr(G[a][i, i])) - N0[(a + 1) ÷ 2])
                 for i in eachindex(t), a in 1:2:length(G))
  canonical = maximum(norm(G[a+1][i, i] - G[a][i, i] + im * I)
                      for i in eachindex(t), a in 1:2:length(G))
  hermiticity = maximum(norm(G[a][i, i] + G[a][i, i]')
                        for i in eachindex(t), a in 1:2:length(G))
  return (; particle, spin, canonical, hermiticity)
end

function quantum_exact_error(G, ts)
  H = ComplexF64[0.05 1; 1 -0.05]
  P = [exp(-im * H * t) for t in ts]
  return maximum(norm(g[i, j] - P[i] * g[1, 1] * P[j]')
                 for g in G for i in eachindex(ts) for j in 1:i)
end

function quantum_reference_error(G, reference)
  n = size(first(G), 3) - 1
  m = size(first(reference), 3) - 1
  @assert iszero(m % n)
  r = m ÷ n
  return maximum(norm(g[i, j] - exact[1+r*(i-1), 1+r*(j-1)])
                 for (g, exact) in zip(G, reference) for i in 1:(n+1) for j in 1:i)
end

function quantum_benchmarks(nmax=64, samples=3)
  @assert nmax >= 16 && ispow2(nmax) && samples >= 1
  BLAS.set_num_threads(1)
  meshes = [2^k for k in 3:trailing_zeros(nmax)]

  # Overlapping Richardson pairs check refinement consistency; their gap is not an error bound.
  quantum_solve(quantum_born, :trapezoid, 4)
  coarse = quantum_solve(quantum_born, :trapezoid, nmax ÷ 2)
  medium = quantum_solve(quantum_born, :trapezoid, nmax)
  fine = quantum_solve(quantum_born, :trapezoid, 2nmax)
  reference = [GreenFunction((4 .* f.data[:, :, 1:2:end, 1:2:end] .- m.data) ./ 3, SkewHermitian)
               for (m, f) in zip(medium.G, fine.G)]
  lower = [GreenFunction((4 .* m.data[:, :, 1:2:end, 1:2:end] .- c.data) ./ 3, SkewHermitian)
           for (c, m) in zip(coarse.G, medium.G)]
  reference_gap = quantum_reference_error(lower, reference)
  println((problem=:second_born, reference=:two_trapezoidal_Richardson_pairs,
           fine_steps=2nmax, pair_gap=reference_gap))
  println("The second-Born reference is extrapolated; pair_gap is a convergence diagnostic, not a certified error bound.")
  println("Lobatto is an experimental fixed-block reference for smooth kernels; it solves each pair of new rows jointly.")
  println("Errors are maximum matrix Frobenius norms over both times and all lesser/greater functions.")
  println("density_error compares the final lesser Green functions, hence the final density matrices.")
  println("Hermiticity measures equal-time density matrices; off-diagonal symmetry is imposed by storage.")
  println("The second-Born Hartree term uses +U*ρ, correcting the original example's sign.")

  rows = NamedTuple[]
  for problem in (quantum_dimer, quantum_born), method in (:smooth, :piecewise, :trapezoid, :lobatto)
    quantum_solve(problem, method, 8)
    for n in (problem == quantum_dimer && method == :trapezoid ? [meshes; 2nmax] : meshes)
      runs = map(1:samples) do _
        GC.gc()
        @timed quantum_solve(problem, method, n)
      end
      (; G, t) = last(runs).value
      error = problem == quantum_dimer ? quantum_exact_error(G, t) : quantum_reference_error(G, reference)
      density_error = if problem == quantum_dimer
        P = exp(-im * ComplexF64[0.05 1; 1 -0.05])
        maximum(norm(g[n+1, n+1] - P * g[1, 1] * P') for g in G[1:2:end])
      else
        maximum(norm(g[n+1, n+1] - exact[nmax+1, nmax+1]) for (g, exact) in zip(G[1:2:end], reference[1:2:end]))
      end
      row = (; problem=nameof(problem), method, steps=n, threads=Threads.nthreads(),
              seconds=sort([r.time for r in runs])[(samples + 1) ÷ 2],
              bytes=minimum(r.bytes for r in runs), error, density_error, quantum_invariants(G, t)...)
      push!(rows, row)
      println(row)
    end
  end
  for problem in (nameof(quantum_dimer), nameof(quantum_born)), target in (1e-4, 1e-5, 1e-6)
    for method in (:smooth, :piecewise, :trapezoid, :lobatto)
      eligible = filter(r -> r.problem == problem && r.method == method && r.error <= target, rows)
      if !isempty(eligible)
        best = eligible[argmin([r.seconds for r in eligible])]
        println((matched_target=target, problem, method, steps=best.steps,
                 seconds=best.seconds, measured_error=best.error))
      else
        println((matched_target=target, problem, method, status=:not_reached))
      end
    end
  end
  return (; rows, reference_gap)
end

if abspath(PROGRAM_FILE) == @__FILE__
  quantum_benchmarks(isempty(ARGS) ? 64 : parse(Int, ARGS[1]),
                     length(ARGS) < 2 ? 3 : parse(Int, ARGS[2]))
end
