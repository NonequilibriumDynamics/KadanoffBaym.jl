mutable struct VolterraWeights{T}
  ws::Vector{T}   # integration weights
  ks::Vector{Int} # integrator orders

  function VolterraWeights(ts::T) where T
    # TODO: Not entirely clear how to determine the order of the weights
    ks = isone(length(ts)) ? Int64[] : [1; 2ones(Int64, length(ts) - 3); 1]
    ws = T[[0.0,]]

    for k in eachindex(ks)
      push!(ws, (@views calculate_weights(ts[1:(k+1)], ks[1:k])))
    end

    return new{T}(ws, ks)
  end
end

function calculate_weights(ts, ks)
  ws = zero(ts)
  for (i, k) in enumerate(ks)
    r = max(1, i - (k - 1)):min(length(ts), i + k) # too large of an interpolant, imo
    ws[r] += Vandermonde(ts[r])' \ [(ts[i+1]^j - ts[i]^j) / j for j in eachindex(r)]
  end
  ws
end
