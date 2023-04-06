function initialize_weights(ts::Vector{T}) where T
  ws = [zeros(T, 1), ]
  for i in 2:length(ts)
    push!(ws, update_weights(last(ws), ts[1:i], 1))
  end
  return ws
end

function update_weights(ws, ts, k)
  ws = copy(ws)
  push!(ws, zero(eltype(ws)))
  l = length(ts)
  r = max(1, l - k):l
  t0 = ts[r[1]] # This subtraction controls large numerical errors when `ts` >> 1. For large Δts, switch to weight integration (see previous implementation)
  ws[r] += Vandermonde(ts[r] .- t0)' \ [((ts[l] - t0)^j - (ts[l-1] - t0)^j) / j for j in eachindex(r)]
  return ws
end
