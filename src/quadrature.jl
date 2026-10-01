"""Weights of the Lagrange interpolant on `[a,b]`, with sample locations `ts`."""
function integration_weights(ts, a, b)
  h = b - a
  iszero(h) && return zero.(ts)
  # Endpoint scaling keeps coordinates bounded even on a very short interval.
  s = maximum(t -> abs(t - b), ts; init=abs(h))
  x = (ts .- b) ./ s
  return h .* (Vandermonde(x)' \ [(-h / s)^(j - 1) / j for j in eachindex(ts)])
end

function initialize_weights(ts::Vector{T}, k=1, quadrature=:smooth) where T
  ws = [zeros(T, 1), ]
  for i in 2:length(ts)
    push!(ws, update_weights(last(ws), view(ts, 1:i), k, quadrature))
  end
  return ws
end

function piecewise_panel(ts, k=2)
  n = length(ts)
  h = ts[n] - ts[n - 1]
  z = zero(h)
  iszero(k) && return n - 1, (z, z, h), (z, z, z)
  l = k == 1 ? nothing : findlast(t -> ts[n - 1] - t >= h, view(ts, 1:(n - 2)))
  isnothing(l) && return n - 1, (z, h / 2, h / 2), (z, z, z)
  r = h / (ts[n - 1] - ts[l])
  η = h * r / 6
  ξ = η / (1 + r)
  return l, (-r * ξ, h / 2 + η, h / 2 - ξ), (r * ξ, -η, ξ)
end

function piecewise_weights!(weights, ts, base)
  n = length(base)
  while length(weights) < n
    push!(weights, similar(base, n))
  end
  resize!(weights, n)
  for w in weights
    resize!(w, n)
    copyto!(w, base)
  end
  # Each column integrates the two branches separated by ts[j].
  for i in 2:n
    l, _, δ = piecewise_panel(view(ts, 1:i))
    for j in (l + 1):(i - 1)
      weights[j][l] += δ[1]
      weights[j][i - 1] += δ[2]
      weights[j][i] += δ[3]
    end
  end
  return weights
end

function update_weights(ws, ts, k, quadrature=:smooth)
  if quadrature == :smooth && length(ts) <= k + 1
    return reverse!(integration_weights(view(ts, length(ts):-1:1), first(ts), last(ts)))
  end
  ws = [ws; zero(eltype(ws))]
  if quadrature == :piecewise
    l, q, _ = piecewise_panel(ts, k)
    ws[l] += q[1]
    ws[end - 1] += q[2]
    ws[end] += q[3]
    return ws
  end
  l = length(ts)
  r = l:-1:max(1, l - k)
  @views ws[r] .+= integration_weights(ts[r], ts[l-1], ts[l])
  return ws
end

"""
    PanelQuadrature(ts, panels)
    Q(K, a, b; sample_domain=min(a,b):max(a,b), init)

Experimental nodal integral of `K(k)` over `[ts[a], ts[b]]`. Each panel is a
contiguous range of node indices; adjacent panels share one endpoint. Integrate
its Lagrange interpolant, using only nonzero weights inside `sample_domain`.
The domain declares where this particular kernel has a smooth continuation;
it does not change the interpolation rule. Separate branches use separate calls.

`init` supplies an additive initial value and compatible accumulation type; it
is not mutated. Equal bounds return a copy without evaluating the kernel.
Reversed bounds negate the integral. The time grid and cached panel partition
must remain unchanged.
Only nodes in `ts` are available; no off-grid reconstruction is implied.

`weights[p][k,a,b]` integrates basis function `k` of panel `p` between local
nodes `a,b`. `prefix` caches cumulative weights for initial-time integrals and
legacy callbacks; general intervals are assembled locally, never by subtracting
cumulative weights (which loses accuracy after a much larger preceding panel).
"""
struct PanelQuadrature{T,P,W,V}
  t::T
  panels::P
  weights::W
  prefix::V
end

function PanelQuadrature(t, panels)
  @assert !isempty(t) && all(isfinite, t) && all(>(0), diff(t))
  panels = collect(panels)
  endpoint = 1
  for nodes in panels
    @assert length(nodes) >= 2 && first(nodes) == endpoint && last(nodes) <= length(t)
    @assert step(nodes) == 1
    endpoint = last(nodes)
  end
  @assert endpoint == length(t)
  T = typeof(zero(eltype(t)) / 2)
  weights = map(panels) do nodes
    W = zeros(T, length(nodes), length(nodes), length(nodes))
    for a in 1:length(nodes)-1, b in a+1:length(nodes)
      W[:,a,b] = integration_weights(view(t, nodes), t[nodes[a]], t[nodes[b]])
    end
    W
  end
  prefix = [zeros(T, 1)]
  for (nodes, W) in zip(panels, weights), b in 2:length(nodes)
    w = zeros(T, last(nodes))
    w[1:first(nodes)] .= prefix[first(nodes)]
    @views w[nodes] .+= W[:, 1, b]
    push!(prefix, w)
  end
  return PanelQuadrature(t, panels, weights, prefix)
end

function interval_weights(Q::PanelQuadrature, a, b)
  a == 1 && return eachindex(Q.prefix[b]), Q.prefix[b]
  p = searchsortedlast(Q.panels, a; by=last) + 1
  q = searchsortedfirst(Q.panels, b; by=first) - 1
  support = first(Q.panels[p]):last(Q.panels[q])
  w = zeros(eltype(Q.prefix[1]), length(support))
  for r in p:q
    nodes, W = Q.panels[r], Q.weights[r]
    l, u = max(a, first(nodes))-first(nodes)+1, min(b, last(nodes))-first(nodes)+1
    @views w[nodes .- first(support) .+ 1] .+= W[:, l, u]
  end
  return support, w
end

function (Q::PanelQuadrature)(K, a, b; sample_domain=min(a,b):max(a,b), init)
  1 <= a <= length(Q.t) && 1 <= b <= length(Q.t) || throw(BoundsError(Q.t, (a,b)))
  value = recursivecopy(init)
  a == b && return value
  nodes, w = interval_weights(Q, min(a,b), max(a,b))
  for (k, ω) in zip(nodes, w)
    if !iszero(ω) && !(k in sample_domain && k <= length(Q.t))
      throw(ArgumentError("quadrature samples node $k outside the available smooth sampling domain"))
    end
  end
  for (k, ω) in zip(nodes, w)
    iszero(ω) || (value = muladd!(value, a < b ? ω : -ω, K(k), value))
  end
  return value
end
