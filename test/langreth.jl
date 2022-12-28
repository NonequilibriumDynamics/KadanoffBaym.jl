function integrate(x::AbstractVector, y::AbstractVector)
  if isone(length(y))
    return zero(first(y))
  end
  @inbounds retval = (x[2] - x[1]) * (y[1] + y[2])
  @inbounds @fastmath @simd for i in 2:(length(y)-1)
    retval += (x[i+1] - x[i]) * (y[i] + y[i+1])
  end
  return 1 // 2 * retval
end

function compute(A, B, ts)
  l = zero(lesser(A))
  for i in 1:size(l, 1)
    for j in 1:i
      l[i,j] += integrate(ts, greater(A)[i, 1:i] .* lesser(B)[1:i, j])
      l[i,j] -= integrate(ts, lesser(A)[i, 1:j] .* greater(B)[1:j, j])
      l[i,j] -= integrate(ts[j:i], lesser(A)[i, j:i] .* lesser(B)[j:i, j])

      l[j,i] = -conj(l[i,j])
    end
    l[i,i] = eltype(l) <: Real ? 0.0 : im * imag(l[i,i])
  end
  
  g = zero(greater(A))
  for i in 1:size(g, 1)
    for j in 1:i
      g[i,j] -= integrate(ts, lesser(A)[i, 1:i] .* greater(B)[1:i, j])
      g[i,j] += integrate(ts, greater(A)[i, 1:j] .* lesser(B)[1:j, j])
      g[i,j] += integrate(ts[j:i], greater(A)[i, j:i] .* greater(B)[j:i, j])

      g[j,i] = -conj(g[i,j])
    end
    g[i,i] = eltype(g) <: Real ? 0.0 : im * imag(g[i,i])
  end
  
  TimeOrderedGreenFunction(l, g)
end

N = 100

# Suppose a non-equidistant time-grid
ts = sort(N*rand(N));

# And 2 time-ordered GFs defined on that grid
a = let
  x = rand(ComplexF64,N,N)
  y = rand(ComplexF64,N,N)
  TimeOrderedGreenFunction(x - x', y - y') # Skew-symmetric L and G components
end
b = let
  x = rand(ComplexF64,N,N)
  y = rand(ComplexF64,N,N)
  TimeOrderedGreenFunction(x - x', y - y') # Skew-symmetric L and G components
end

dts = reduce(hcat, ([KadanoffBaym.calculate_weights(ts[1:i], ones(Int64, i-1)); zeros(length(ts)-i)] for i in eachindex(ts))) |> UpperTriangular

⋆(a, b) = conv(a, b, dts)

@testset "Langreth's rules" begin
  @test greater(a ⋆ b) ≈ greater(compute(a, b, ts))
  @test lesser(a ⋆ b) ≈ lesser(compute(a, b, ts))
  @test retarded(a ⋆ b) ≈ retarded(compute(a, b, ts))
  @test greater(a ⋆ b ⋆ a) ≈ greater(compute(compute(a, b, ts), a, ts))
  @test retarded(a ⋆ b ⋆ a) ≈ retarded(compute(compute(a, b, ts), a, ts))
end
