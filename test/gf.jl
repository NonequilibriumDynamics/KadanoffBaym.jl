N = 10

@testset "2D GF" begin
  # Test 2d getindex setindex!
  data = zeros(ComplexF64, N, N)

  lgf = GreenFunction(copy(data), SkewHermitian)
  ggf = GreenFunction(copy(data), SkewHermitian)

  v = 30 + 30im
  lgf[2, N] = v
  ggf[N, 2] = v

  @test lgf.data[N, 2] == -adjoint(v)
  @test ggf.data[2, N] == -adjoint(v)

  @test lgf[2, N] == -adjoint(lgf.data[N, 2])
  @test ggf.data[2, N] == -adjoint(ggf[N, 2])
end

@testset "4D GF" begin
  # Test 4d getindex setindex!
  data = zeros(ComplexF64, N, N, N, N)

  lgf = GreenFunction(copy(data), SkewHermitian)
  ggf = GreenFunction(copy(data), SkewHermitian)

  v = rand(ComplexF64, N, N)
  lgf[2, N] = v
  ggf[N, 2] = v

  @test lgf.data[:, :, N, 2] == -adjoint(v)
  @test ggf.data[:, :, 2, N] == -adjoint(v)

  @test lgf[2, N] == -adjoint(lgf.data[:, :, N, 2])
  @test ggf.data[:, :, 2, N] == -adjoint(ggf[N, 2])
end

@testset "Base functions & setindex!" begin
  # Test AbstractArray-like behaviour
  data = rand(ComplexF64, N, N, N, N)
  gf = GreenFunction(copy(data), SkewHermitian)

  @test (-gf).data == (-data)
  @test (conj(gf)).data == conj(data)
  @test (real(gf)).data == real(data)
  @test (imag(gf)).data == imag(data)
end

@testset "setindex!" begin
  data = zeros(ComplexF64, N, N, N, N)
  gf = GreenFunction(copy(data), SkewHermitian)

  b = rand(ComplexF64, N, N)

  gf[1, 2] = b
  data[:, :, 1, 2] = b
  data[:, :, 2, 1] = -adjoint(b)
  @test gf.data == data

  # # this used to fail
  # gf[:, :, 2, 1] = b
  # data[:, :, 2, 1] = b
  # data[:, :, 1, 2] = -adjoint(b)

  for i in 1:N, j in 1:i
    data[:, :, i, j] = b
    if j != i
      data[:, :, j, i] = -adjoint(b)
    end
  end
  
  for i in 1:N, j in 1:i
    gf[i, j] = b
  end

  @test gf.data == data
end
