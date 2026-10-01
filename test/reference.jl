using LinearAlgebra: norm
using KadanoffBaym: PanelQuadrature, weighted_equations

"""
    reference_trapezoid!(fv!, fd!, u, ts; f1! = nothing, v = [],
                        callback = (args...) -> nothing, start = 1,
                        atol = 1e-13, maxiters = 100)

Fixed-grid reference for the coupled two-time integral equations. Both evolution
and memory use composite trapezoidal integration, independently of the production
Adams formulas and adaptive stencil selection. Callbacks have the `kbsolve!` signatures.
The accepted history occupies indices `1:start`, including any callback-managed
history. Each subsequent row is solved together with its diagonal and one-time
variables by Picard iteration.

`atol` bounds the absolute discrete equation residual, not the integration error.
Picard convergence requires a sufficiently small step; failure throws. Kernels
must be continuous and smooth on either side of each grid-aligned diagonal cusp.
This reference interpolates sampled integrands, including nonlinear kernels.
Weighted callbacks do not distinguish endpoint traces at jumps.
Callbacks must be algebraic updates or independently controlled auxiliary solves.

    reference_trapezoid!(equations::NamedTuple, u, ts; v=[], start=1,
                        atol=1e-13, maxiters=100)

The kernel interface takes `equations = (; fv!, fd!, f1!, callback)`, with `f1! =
nothing` when unused. Signatures are `fv!(out,ts,integral,i,j)`, likewise `fd!`,
`f1!(out,ts,integral,i)`, and `callback(ts,integral,i,j)`. `integral` is a
`PanelQuadrature`: the model supplies its kernels, bounds, and smooth sampling
domains. Separate kernel calls can provide distinct memory endpoint traces;
discontinuous evolution coefficients still require separate RHS traces.
"""
function reference_trapezoid!(fv!, fd!, u, ts; f1! = nothing, callback = (args...) -> nothing, kwargs...)
  return reference_trapezoid!(weighted_equations(fv!, fd!; f1!, callback), u, ts; kwargs...)
end

function reference_trapezoid!(equations::NamedTuple, u, ts; v = [], start = 1,
                             atol = 1e-13, maxiters = 100)
  (; fv!, fd!, f1!, callback) = equations
  @assert 1 <= start <= length(ts) && all(>(0), diff(ts))
  @assert all(g -> last(size(g)) >= start, (u..., v...))
  foreach(g -> resize!(g, length(ts)), (u..., v...))

  Q = PanelQuadrature(ts, [i:i+1 for i in 1:length(ts)-1])

  function derivatives!(i)
    t = view(ts, 1:i)
    integral = PanelQuadrature(t, Q.panels, Q.weights, Q.prefix)
    for j in 1:i
      callback(t, integral, i, j)
    end
    F = [zero(g[i, j]) for j in 1:i, g in u]
    d = [zero(g[i, i]) for g in u]
    f = [zero(g[i]) for g in v]
    for j in 1:i
      fv!(view(F, j, :), t, integral, i, j)
    end
    fd!(d, t, integral, i, i)
    isempty(v) || f1!(f, t, integral, i)
    return F, d, f
  end

  F, d, f = derivatives!(start)
  iterations = zeros(Int, length(ts) - start)
  for i in (start + 1):length(ts)
    h = ts[i] - ts[i-1]
    B = [g[i-1, j] + h / 2 * F[j, a] for j in 1:(i-1), (a, g) in enumerate(u)]
    D = [g[i-1, i-1] + h / 2 * d[a] for (a, g) in enumerate(u)]
    V = [g[i-1] + h / 2 * f[a] for (a, g) in enumerate(v)]
    for (a, g) in enumerate(u)
      for j in 1:(i-1)
        g[i, j] = B[j, a] + h / 2 * F[j, a]
      end
      g[i, i] = D[a] + h / 2 * d[a]
    end
    for (a, g) in enumerate(v)
      g[i] = V[a] + h / 2 * f[a]
    end

    converged = false
    for k in 1:maxiters
      F, d, f = derivatives!(i)
      R = [B[j, a] + h / 2 * F[j, a] - g[i, j] for j in 1:(i-1), (a, g) in enumerate(u)]
      r = [D[a] + h / 2 * d[a] - g[i, i] for (a, g) in enumerate(u)]
      s = [V[a] + h / 2 * f[a] - g[i] for (a, g) in enumerate(v)]
      residual = maximum(x -> norm(x, Inf), Iterators.flatten((R, r, s)); init=0.0)
      if residual <= atol
        iterations[i-start] = k
        converged = true
        break
      end
      for (a, g) in enumerate(u)
        for j in 1:(i-1)
          g[i, j] += R[j, a]
        end
        g[i, i] += r[a]
      end
      for (a, g) in enumerate(v)
        g[i] += s[a]
      end
    end
    converged || error("Trapezoidal iteration did not converge at time $(ts[i]).")
  end
  return (; t=ts, w=Q.prefix, iterations)
end
