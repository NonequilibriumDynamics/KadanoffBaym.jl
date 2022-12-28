mutable struct VolterraWeights{T}
  ws::Vector{T}   # integration weights
  ks::Vector{Int} # integrator orders

  function VolterraWeights(ts::T, atol=1e-8, rtol=1e-3) where {T}
    # TODO: Not entirely clear how to determine the order of the weights
    ks = isone(length(ts)) ? Int64[] : [1; 2ones(Int64, length(ts) - 3); 1]
    ws = T[[0.0,]]

    for k in eachindex(ks)
      push!(ws, (@views calculate_weights(ts[1:(k+1)], ks[1:k])))
    end

    return new{T}(ws, ks)
  end
end

struct B{N,T} <: AbstractArray{T, 1}
  l::T
  u::T

  B{N}(l::T, u::T) where {N,T} = new{N,T}(l, u) 
end

Base.size(b::B{N,T}) where {N,T} = (N, )
Base.getindex(b::B, i) = (b.u^(i) - b.l^(i)) / (i)

function calculate_weights(ts, ks)
  ws = zero(ts)
  for (i, k) in enumerate(ks)
    ws[i:(i+k)] .+= Vandermonde(ts[i:(i+k)])' \ B{k+1}(ts[i], ts[i+1])
  end
  ws
end