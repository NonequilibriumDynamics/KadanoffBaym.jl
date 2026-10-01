@testset "Langreth's rules" begin
  trapezoid(ts, f) = sum(
    (ts[i+1] - ts[i]) * (f[i] + f[i+1]) / 2 for i in 1:(length(f) - 1);
    init=zero(eltype(f)),
  )

  function reference(A, B, ts)
    component(a, b) = [
      trapezoid(ts[1:i], (greater(A)[i, 1:i] - lesser(A)[i, 1:i]) .* b[1:i, j]) +
      trapezoid(ts[1:j], a[i, 1:j] .* (lesser(B)[1:j, j] - greater(B)[1:j, j]))
      for i in eachindex(ts), j in eachindex(ts)
    ]
    return TimeOrderedGreenFunction(
      component(lesser(A), lesser(B)),
      component(greater(A), greater(B)),
    )
  end

  ts = [0.0, 0.07, 0.15, 0.38, 0.9, 1.3, 2.0]
  ws = UpperTriangular(reduce(hcat, [
    [w; zeros(length(ts) - length(w))] for w in KadanoffBaym.initialize_weights(ts)
  ]))
  A = TimeOrderedGreenFunction(
    [im * exp(-im * (t - s)) for t in ts, s in ts],
    [-2im * exp(-2im * (t - s)) for t in ts, s in ts],
  )
  B = TimeOrderedGreenFunction(
    [im * (t + s) for t in ts, s in ts],
    [-im * (1 + t * s) for t in ts, s in ts],
  )
  C = conv(A, B, ws)
  expected = reference(A, B, ts)
  @test lesser(C) ≈ lesser(expected)
  @test greater(C) ≈ greater(expected)

  nested = conv(C, A, ws)
  expected_nested = reference(expected, A, ts)
  @test lesser(nested) ≈ lesser(expected_nested)
  @test greater(nested) ≈ greater(expected_nested)
  @test retarded(im * A) ≈ im * retarded(A)
  @test advanced(im * A) ≈ im * advanced(A)
  @test lesser(conv(im * A, B, ws)) ≈ im * lesser(C)
  @test greater(conv(A, im * B, ws)) ≈ im * greater(C)

  @testset "Exact polynomial convolution" begin
    A = TimeOrderedGreenFunction(im * ones(length(ts), length(ts)), -im * ones(length(ts), length(ts)))
    B = TimeOrderedGreenFunction([im * (t + s) for t in ts, s in ts], zeros(ComplexF64, length(ts), length(ts)))
    C = conv(A, B, ws)
    @test lesser(C) ≈ [t^2 + 2t*s - 3s^2 / 2 for t in ts, s in ts]
    @test greater(C) ≈ [3s^2 / 2 for t in ts, s in ts]
    @test retarded(C) ≈ [t >= s ? 3s^2 - t^2 - 2t*s : 0.0 for t in ts, s in ts]
    @test advanced(C) ≈ [t <= s ? t^2 + 2t*s - 3s^2 : 0.0 for t in ts, s in ts]
  end
end

@testset "Langreth products preserve noncommuting matrix order" begin
  ts = [0.0, 0.1, 0.4, 0.8]
  ws = UpperTriangular(reduce(hcat, [
    [w; zeros(length(ts) - length(w))] for w in KadanoffBaym.initialize_weights(ts)
  ]))
  M = ComplexF64[1 2im; 3 4]
  N = ComplexF64[2im 1; 0 3]
  A = TimeOrderedGreenFunction(
    [(1+t)*M + s*N for t in ts, s in ts],
    [t*s*M + (2+t)*N for t in ts, s in ts],
  )
  B = TimeOrderedGreenFunction(
    [t*N + (1+s)*M for t in ts, s in ts],
    [t*M - s*N for t in ts, s in ts],
  )
  function reference(A, B, a, b)
    [sum(ws[k,i] * (greater(A)[i,k] - lesser(A)[i,k]) * b[k,j] for k in 1:i) +
     sum(ws[k,j] * a[i,k] * (lesser(B)[k,j] - greater(B)[k,j]) for k in 1:j)
     for i in eachindex(ts), j in eachindex(ts)]
  end
  C = conv(A, B, ws)
  CL = reference(A, B, lesser(A), lesser(B))
  CG = reference(A, B, greater(A), greater(B))
  @test all(isapprox.(lesser(C), CL))
  @test all(isapprox.(greater(C), CG))

  expected = TimeOrderedGreenFunction(CL, CG)
  nested = conv(C, A, ws)
  @test all(isapprox.(lesser(nested), reference(expected, A, CL, lesser(A))))
  @test all(isapprox.(greater(nested), reference(expected, A, CG, greater(A))))
end
