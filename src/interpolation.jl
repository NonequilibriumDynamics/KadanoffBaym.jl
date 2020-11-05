import OrdinaryDiffEq: _searchsortedfirst
using NormalHermiteSplines

function interpolate!(tvals, ts)
  @inbounds tdir = sign(ts[end]-ts[1])
  
  i₋,i₊ = 1, 2
  j₋,j₊ = 1, 2
  
  @inbounds for (t,t′) ∈ tvals
    i₊ = min(lastindex(ts), _searchsortedfirst(ts,t,i₊,tdir > 0))
    i₋ = i₊ > 1 ? i₊ - 1 : i₊
    
    j₊ = min(lastindex(ts), _searchsortedfirst(ts,t,j₊,tdir > 0))
    j₋ = j₊ > 1 ? j₊ - 1 : j₊
    
    nodes = [i₋ i₋ i₊ i₊; j₋ j₊ j₋ j₊]
    times = [ts[i₋] ts[i₋] ts[i₊] ts[i₊]; ts[j₋] ts[j₊] ts[j₋] ts[j₊]]
    
    n = size(nodes, 2)
    u = Vector{Float64}(undef, n)
    d_nodes = Matrix{Float64}(undef, 2, 2n)  # directional derivative nodes 
    es = Matrix{Float64}(undef, 2, 2n)       # derivative directions 
    du = Vector{Float64}(undef, 2n)          # directional derivative values
    
    for i in 1:n
      u[i] = data.GL[1,1,nodes_[1,i], nodes_[2,i]] |> imag

      # Horizontal derivative
      d_nodes[1,2i-1] = nodes[1,i]
      d_nodes[2,2i-1] = nodes[2,i]

      du[2i-1] = rhs_vert(data, ts, nodes_[1,i], nodes_[2,i])[1][1,1] |> imag

      es[1,2i-1] = 1.0
      es[2,2i-1] = 0.0

      # Vertical derivative
      d_nodes[1,2i] = nodes[1,i]
      d_nodes[2,2i] = nodes[2,i]

      du[2i] = rhs_hori(data, ts, nodes_[1,i], nodes_[2,i])[1][1,1] |> imag

      es[1,2i] = 0.0
      es[2,2i] = 1.0
    end
    return NormalHermiteSplines._prepare(nodes, d_nodes, es, RK_H2())
  end
end