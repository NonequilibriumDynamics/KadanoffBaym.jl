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
  ws[r] += Vandermonde(ts[r])' \ [(ts[l]^j - ts[l-1]^j) / j for j in eachindex(r)]
  return ws
end
