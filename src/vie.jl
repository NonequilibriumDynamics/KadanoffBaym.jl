mutable struct VolterraWeights{T}
  ws::Vector{T}   # integration weights
  ks::Vector{Int} # integrator orders

  function VolterraWeights(ts::T) where {T}
    # if starting from a point in the past, consider trapezium
    ks = ones(Int64, length(ts)-1)

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
  L(; i, j, k) = quadgk(ts[i], ts[i+1]; atol=atol, rtol=rtol) do x
    prod((x - ts[i+1 - m]) / (ts[i+1 - j] - ts[i+1 - m]) for m in 0:k if m != j)
  end[1]

  # Compute weights
  ws = zero(ts)
  for (i, k) in enumerate(ks)
    for j in 0:k
      ws[i+1-j] += L(i=i, j=j, k=k)
    end
  end

  return ws
end