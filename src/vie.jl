mutable struct VolterraWeights{T}
  ws::Vector{T}   # integration weights
  ks::Vector{Int} # integrator orders

  function VolterraWeights(ts::T) where {T}
    # TODO: Not entirely clear how to determine the order of the weights
    ks = isone(length(ts)) ? Int64[] : [1; 2ones(Int64, length(ts) - 3); 1]
    ws = T[[0.0,]]

    for k in eachindex(ks)
      push!(ws, @views calculate_weights(ts[1:(k+1)], ks[1:k]))
    end

    return new{T}(ws, ks)
  end
end

# Calculates the weights used for evaluating ∫dt f(t) as ∑ᵢ hᵢ fᵢ
function calculate_weights(ts, ks, atol=1e-8, rtol=1e-3)
  @assert length(ts) == length(ks) + 1

  # Integral of Lagrange polynom
  L(; i, j, ks) = quadgk(ts[i], ts[i+1]; atol=atol, rtol=rtol) do x
    prod((x - ts[i+1 - m]) / (ts[i+1 - j] - ts[i+1 - m]) for m in ks if m != j)
  end[1]

  # Compute weights
  ws = zero(ts)
  for (i, k) in enumerate(ks)
    r = -min((k-1), length(ks)-i):min(k,i)
    for j in r
      ws[i+1-j] += L(i=i, j=j, ks=r)
    end
  end

  return ws
end

function update_weights!(ws, ts, ks, atol=1e-8, rtol=1e-3)
  push!(ws, zero(eltype(ws)))

  # Integral of Lagrange polynom
  L(; i, j, ks) = quadgk(ts[i], ts[i+1]; atol=atol, rtol=rtol) do x
    prod((x - ts[i+1 - m]) / (ts[i+1 - j] - ts[i+1 - m]) for m in ks if m != j)
  end[1]

  i_last = length(ks)
  k_last = 2ks[end]-1

  for i in min((i_last-k_last),1):i_last
    ws[i] = zero(eltype(ws))
  end

  for i in min((i_last-k_last),1):i_last
    k = ks[i]
    r = -min((k-1), length(ks)-i):min(k,i)
    for j in r
      ws[i+1-j] += L(i=i, j=j, ks=r)
    end
  end

  return ws
end
