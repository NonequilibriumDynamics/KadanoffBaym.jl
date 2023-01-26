function initialize_weights(ts::Vector{T}) where T
  ws = [zeros(T, 1), ]
  for i in 2:length(ts)
    update_weights!(ws, ts[1:i], 1)
  end
  return ws
end

function update_weights!(ws, ts, k)
  _ws = copy(ws[end])
  push!(_ws, zero(eltype(_ws)))
  l = length(ts)
  r = max(1, l - k):l
  _ws[r] += Vandermonde(ts[r])' \ [(ts[l]^j - ts[l-1]^j) / j for j in eachindex(r)]
  push!(ws, _ws)
end
