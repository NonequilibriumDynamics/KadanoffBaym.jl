using LinearAlgebra: norm
using KadanoffBaym: PanelQuadrature, weighted_equations

"""
    reference_lobatto!(fv!, fd!, u, blocks; f1! = nothing, v = [],
                      callback = (args...) -> nothing, atol = 1e-13, maxiters = 100)

Experimental fixed-block reference for smooth two-time equations. `blocks` are
strictly increasing macro endpoints; the returned grid includes each midpoint.
Evolution and memory integrate the same quadratic Lobatto interpolant. The new
triangular corner integrates a total-degree-two interpolant of the vertical RHS.
All stages, diagonal values, and one-time variables are solved together.

Only the initial values may be supplied. This is not an adaptive or restartable
replacement for `kbsolve!`. Kernels must be smooth across the whole interpolation
block, including across the time diagonal. `fv!` is evaluated only at `i >= j`.

Callbacks have the reference-trapezoid signatures, but different sample support:
midpoint weights include the block endpoint. Sum over `eachindex(w)`, not `1:i`.
Every stage's callback runs before any stage derivative; auxiliary equations that
need their own coupled solve cannot be implemented by sequential callbacks alone.

`atol` bounds the absolute fixed-point residual, not integration error. The corner
uses the newly mapped diagonal, so its original equation residual is at most
`2atol`. Picard failure throws; sufficiently small blocks are required even for
linear oscillations. Tighten the residual tolerance when refining the grid.

    reference_lobatto!(equations::NamedTuple, u, blocks; v=[], atol=1e-13, maxiters=100)

The same kernel interface as `reference_trapezoid!` exposes each integrand and
its integration limits. Midpoint integrals require an explicitly declared smooth
sampling domain covering the block endpoint. Separate branches may supply their
own smooth continuations; declaring one does not construct or validate it.
"""
function reference_lobatto!(fv!, fd!, u, blocks; f1! = nothing, callback=(args...) -> nothing, kwargs...)
  return reference_lobatto!(weighted_equations(fv!, fd!; f1!, callback), u, blocks; kwargs...)
end

function reference_lobatto!(equations::NamedTuple, u, blocks; v = [], atol=1e-13, maxiters=100)
  (; fv!, fd!, f1!, callback) = equations
  @assert length(blocks) >= 2 && all(isfinite, blocks) && all(>(0), diff(blocks))
  @assert all(g -> last(size(g)) == 1, (u..., v...)) "Only initial data, not a stored history, are supported"
  t = sort!([float.(blocks); (blocks[1:end-1] + blocks[2:end]) / 2])
  @assert all(>(0), diff(t)) "Midpoints must be representable"
  foreach(g -> resize!(g, length(t)), (u..., v...))
  A = [5//24 1//3 -1//24; 1//6 2//3 1//6]
  Q = PanelQuadrature(t, [p:p+2 for p in 1:2:length(t)-2])

  function derivatives(i, ts, integral)
    F = [zero(g[i, j]) for j in 1:i, g in u]
    d = [zero(g[i, i]) for g in u]
    f = [zero(g[i]) for g in v]
    for j in 1:i
      fv!(view(F, j, :), ts, integral, i, j)
    end
    fd!(d, ts, integral, i, i)
    isempty(v) || f1!(f, ts, integral, i)
    return F, d, f
  end

  initial = PanelQuadrature(view(t, 1:1), Q.panels, Q.weights, Q.prefix)
  callback(initial.t, initial, 1, 1)
  F0, d0, f0 = derivatives(1, initial.t, initial)
  iterations = zeros(Int, length(blocks)-1)
  for p in 1:2:length(t)-2
    a, z = p+1, p+2
    h = t[z]-t[p]
    ts = view(t, 1:z)
    integral = PanelQuadrature(ts, Q.panels, Q.weights, Q.prefix)
    for g in u, i in a:z
      for j in 1:p
        g[i, j] = copy(g[p, j])
      end
      for j in a:i
        g[i, j] = copy(g[p, p])
      end
    end
    for g in v, i in a:z
      g[i] = copy(g[p])
    end

    converged = false
    for k in 1:maxiters
      for i in a:z, j in 1:i
        callback(ts, integral, i, j)
      end
      F1, d1, f1 = derivatives(a, ts, integral)
      F2, d2, f2 = derivatives(z, ts, integral)
      D = [g[p, p] + h*(A[r, 1]*d0[b] + A[r, 2]*d1[b] + A[r, 3]*d2[b])
           for r in 1:2, (b, g) in enumerate(u)]
      V = [g[p] + h*(A[r, 1]*f0[b] + A[r, 2]*f1[b] + A[r, 3]*f2[b])
           for r in 1:2, (b, g) in enumerate(v)]
      U = [[g[p, j] + h*(A[r, 1]*F0[j, b] + A[r, 2]*F1[j, b] + A[r, 3]*F2[j, b])
            for j in 1:p, (b, g) in enumerate(u)] for r in 1:2]
      # Integral of the quadratic triangular interpolant over t ∈ [t[a], t[z]].
      C = [D[1, b] + h/4*(F1[a, b] + F2[a, b]) -
           h/24*(F2[p, b] - 2F1[p, b] + F0[p, b]) for b in eachindex(u)]
      residual = maximum((norm(U[r][j,b]-g[p+r,j], Inf)
                          for r in 1:2 for j in 1:p for (b,g) in enumerate(u)); init=0.0)
      residual = max(residual,
          maximum((norm(D[r,b]-g[p+r,p+r], Inf) for r in 1:2 for (b,g) in enumerate(u)); init=0.0),
          maximum((norm(C[b]-g[z,a], Inf) for (b,g) in enumerate(u)); init=0.0),
          maximum((norm(V[r,b]-g[p+r], Inf) for r in 1:2 for (b,g) in enumerate(v)); init=0.0))
      if residual <= atol
        iterations[(p+1)÷2] = k
        F0, d0, f0 = F2, d2, f2
        converged = true
        break
      end
      for (b,g) in enumerate(u)
        for r in 1:2, j in 1:p
          g[p+r,j] = U[r][j,b]
        end
        g[a,a], g[z,z], g[z,a] = D[1,b], D[2,b], C[b]
      end
      for (b,g) in enumerate(v), r in 1:2
        g[p+r] = V[r,b]
      end
    end
    converged || error("Lobatto iteration did not converge on [$(t[p]), $(t[z])].")
  end
  return (; t, w=Q.prefix, iterations)
end
