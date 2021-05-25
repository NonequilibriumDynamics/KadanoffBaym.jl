N = 10

@testset "2D GF" begin
  # Test 2d getindex setindex!
  data = zeros(ComplexF64, N, N)

  lgf = GreenFunction(copy(data), Lesser)
  ggf = GreenFunction(copy(data), Greater)

  v = 30 + 30im
  lgf[2, N] = v
  ggf[N, 2] = v

  @test lgf.data[N, 2, ..] == -adjoint(v)
  @test ggf.data[2, N, ..] == -adjoint(v)

  @test lgf[2, N] == -adjoint(lgf.data[N, 2])
  @test ggf.data[2, N, ..] == -adjoint(ggf[N, 2])
end

@testset "4D GF" begin
  # Test 4d getindex setindex!
  data = zeros(ComplexF64, N, N, N, N)

  lgf = GreenFunction(copy(data), Lesser)
  ggf = GreenFunction(copy(data), Greater)

  v = rand(ComplexF64, N, N)
  lgf[2, N] = v
  ggf[N, 2] = v

  @test lgf.data[.., N, 2] == -adjoint(v)
  @test ggf.data[.., 2, N] == -adjoint(v)

  @test lgf[2, N] == -adjoint(lgf.data[.., N, 2])
  @test ggf.data[.., 2, N] == -adjoint(ggf[N, 2])
end

@testset "Base functions & setindex!" begin
  # Test AbstractArray-like behaviour
  data = rand(ComplexF64, N, N, N, N)
  gf = GreenFunction(copy(data), Lesser)

  @test (-gf).data == (-data)
  @test (conj(gf)).data == conj(data)
  @test (real(gf)).data == real(data)
  @test (imag(gf)).data == imag(data)
end

@testset "setindex!" begin
  function setindexA(A::AbstractArray)
    for i in 1:N, j in 1:i
      A[.., i, j] = b
      if j != i
        A[.., j, i] = -adjoint(b)
      end
    end
  end

  function setindexG(G)
    for i in 1:N, j in 1:i
      G[i, j] = b
    end
  end

  data = zeros(ComplexF64, N, N, N, N)
  gf = GreenFunction(copy(data), Lesser)

  b = rand(ComplexF64, N, N)

  gf[1, 2] = b
  data[.., 1, 2] = b
  data[.., 2, 1] = -adjoint(b)
  @test gf.data == data

  setindexA(data)
  setindexG(gf)
  @test gf.data == data

  # using BenchmarkTools
  # @show @btime $setindexA($data)
  # @show @btime $setindexG($gf)
end
