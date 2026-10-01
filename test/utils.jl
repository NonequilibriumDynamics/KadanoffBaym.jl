@testset "Dense arithmetic reuses matrix buffers" begin
  for T in (Float32, Float64, ComplexF64)
    x = reshape(T.(1:6), 2, 3)
    y = reverse(x)
    a = T(0.37)
    for b in (y, T(-0.7))
      expected = muladd.(a, x, b)
      out = similar(x)
      @test KadanoffBaym.muladd!(out, a, x, b) === out
      @test out ≈ expected rtol=4eps(real(one(T)))
      out = copy(x)
      @test KadanoffBaym.muladd!(out, a, out, b) ≈ expected rtol=4eps(real(one(T)))
    end
    out = copy(y)
    @test KadanoffBaym.muladd!(out, a, x, out) ≈ muladd.(a, x, y) rtol=4eps(real(one(T)))
  end
end

@testset "Maximum over histories with component RMS within each value" begin
  for T in (Float32, Float64, ComplexF64)
    u = [T[1, 2, 3], T[4, 5, 6]]
    @test KadanoffBaym.state_norm(u) == 6
    @test iszero(KadanoffBaym.state_norm(zero.(u)))

    u = [[reshape(T.(1:6), 2, 3) for _ in 1:5] for _ in 1:2]
    @test KadanoffBaym.state_norm(u) ≈ sqrt(sum(abs2, T.(1:6)) / 6)
    @test iszero(KadanoffBaym.state_norm(zero.(u)))
  end
  for a in (1e-150, 1e153)
    u = [[fill(a, 2, 2) for _ in 1:1000]]
    @test KadanoffBaym.state_norm(u) ≈ a
  end
  for n in (1, 100), fields in (1, 3), value in (2.0, fill(2.0, 2, 2))
    u = [[zero(value) for _ in 1:n] for _ in 1:fields]
    u[1][end] = copy(value)
    @test KadanoffBaym.state_norm(u) == 2
  end
end

@testset "Componentwise starting step" begin
  u = [1e9, 1.0]
  f = [0.0, 1e6]
  atol, rtol = 1e-8, 1e-6
  scale = atol .+ rtol .* abs.(u)
  expected = 1e-2 * LinearAlgebra.norm(u ./ scale) / LinearAlgebra.norm(f ./ scale)
  @test KadanoffBaym.initial_step([[f]], [[u]], atol, rtol) ≈ expected

  for (data, expected) in ((x -> [copy(x)],
                            1e-2 * maximum(abs, u ./ scale) / maximum(abs, f ./ scale)),
                           (x -> [[reshape(copy(x), 1, 2)]], expected))
    @test KadanoffBaym.initial_step(data(f), data(u), atol, rtol) ≈ expected
    @test KadanoffBaym.initial_step(data(1e-3f), data(1e-3u), 1e-3atol, rtol) ≈ expected
    @test KadanoffBaym.initial_step(data(100f), data(u), atol, rtol) ≈ expected / 100
  end
  @test KadanoffBaym.initial_step([[0.0]], [[1.0]], atol, rtol) == 1e-6
  @test KadanoffBaym.initial_step([[1.0]], [[0.0]], atol, rtol) == 1e-6
end

@testset "Matrix residuals use scalar tolerances" begin
  u0 = [[ComplexF64[1 2; 3 4], ComplexF64[5 6; 7 8]]]
  u1 = 2 .* u0
  err = 0.1 .* u0
  out = zero.(u0)
  KadanoffBaym.calculate_residuals!(out, err, u0, u1, 1e-8, 1e-6)
  for j in eachindex(out), i in eachindex(out[j])
    @test out[j][i] ≈ err[j][i] ./ (1e-8 .+ 1e-6 .* abs.(u1[j][i]))
  end
end

@testset "Copying extended histories preserves every stored value" begin
  for v in (1.0, ones(2, 2))
    src = [[copy(v), 2v]]
    out = zero.(src)
    push!(src[1], 3v)
    push!(out[1], zero(v))
    @test KadanoffBaym._recursivecopy!(src, src) === src
    @test src[1] == [v, 2v, 3v]
    @test KadanoffBaym._recursivecopy!(out, src) === out
    @test out[1] == [v, 2v, 3v]
    if v isa Number
      src[1][3] = 0
    else
      src[1][3] .= 0
    end
    @test out[1][3] == 3v
  end
end

@testset "Callback field slices preserve each field's storage type" begin
  fields = Vector[Float32[0, 1], Float64[0, 2]]
  out = KadanoffBaym.FieldSlice(fields, 2)
  out[1] = 0.1
  @test out[1] === Float32(0.1)
  @test fields[1][2] === Float32(0.1)
  out .= 0.1
  @test out[1] === Float32(0.1)
  @test out[2] === 0.1
  @test fields[1][1] === 0f0
  @test typeof.(collect(out)) == [Float32, Float64]

  homogeneous = [[0.0, 0.0], [0.0, 0.0]]
  KadanoffBaym.FieldSlice(homogeneous, 2) .= [0.25, -0.5]
  @test homogeneous == [[0.0, 0.25], [0.0, -0.5]]

  fields = Vector[[ones(Float32, 2, 2)], [ones(Float64, 3, 3)]]
  out = KadanoffBaym.FieldSlice(fields, 1)
  replacement = fill(0.1, 2, 2)
  out[1] = replacement
  @test out[1] == Float32.(replacement)
  @test out[1] isa Matrix{Float32}
  @test out[1] !== replacement
  before = out[2]
  out[2] .= 0.1f0
  @test out[2] === before
  @test out[2] isa Matrix{Float64}
  @test all(==(Float64(0.1f0)), out[2])
  for materialize in (copy, collect)
    values = materialize(out)
    @test typeof.(values) == [Matrix{Float32}, Matrix{Float64}]
    @test size.(values) == [(2, 2), (3, 3)]
    @test values[1] === out[1] && values[2] === out[2]
  end
end

@testset "Owned copies separate aliased input leaves" begin
  A = ones(2, 2)
  src = [[A, A]]
  copied = KadanoffBaym.recursivecopy(src)
  assigned = zero.(src)
  @test KadanoffBaym._recursivecopy!(assigned, src) === assigned
  for out in (copied, assigned)
    @test out == src
    @test out[1][1] !== A && out[1][2] !== A
    @test out[1][1] !== out[1][2]
  end
  A .= 0
  @test all(all(isone, leaf) for out in (copied, assigned) for leaf in out[1])
end
@testset "Callback assignments capture owned field values" begin
  for prototype in (ones(Float32, 2, 2), spdiagm(0 => ones(Float32, 2)))
    out = KadanoffBaym.FieldSlice([[prototype]], 1)
    work = [0.1 0.2; 0.3 0.4]
    out[1] = work
    @test out[1] === prototype
    @test out[1] == Float32.(work)
    work .= 7
    @test out[1] != work
    for transform in (transpose, adjoint, A -> view(A, 2:-1:1, 2:-1:1))
      expected = copy(transform(out[1]))
      out[1] = transform(out[1])
      @test out[1] == expected
    end
    previous = copy(out[1])
    @test_throws DimensionMismatch (out[1] = ones(2, 1))
    @test_throws DimensionMismatch (out[1] = ones(1, 4))
    @test out[1] == previous
  end

  out = KadanoffBaym.FieldSlice([[zeros(2, 2)], [zeros(2, 2)]], 1)
  work = ones(2, 2)
  out[1] = work
  work .= 3
  out[2] = work
  @test out[1] == ones(2, 2)
  @test out[2] == fill(3., 2, 2)
  out[2] = out[1]
  out[1] .*= 2
  @test out[2] == ones(2, 2)

  A, B = ones(2, 2), fill(2., 2, 2)
  out = KadanoffBaym.FieldSlice([[[A, B]]], 1)
  out[1] = [B, A]
  @test out[1] == [fill(2., 2, 2), ones(2, 2)]
  @test out[1][1] !== B && out[1][2] !== A
  out[1] = [A, A]
  @test out[1][1] !== out[1][2]
end
