using Test, BenchmarkTools
using KadanoffBaym

N = 10

@testset "2D GF" begin
  # Test 2d getindex setindex!
  data = rand(ComplexF64, N, N)
  data -= transpose(data)

  lgf = GreenFunction(copy(data), Lesser)
  ggf = GreenFunction(copy(data), Greater)

  v = 30 + 30im
  lgf[2,N] = v
  ggf[N,2] = v

  @test lgf.data[N,2,..] == -adjoint(v)
  @test ggf.data[2,N,..] == -adjoint(v)

  @test lgf[2,N] == -adjoint(lgf.data[N,2])
  @test ggf.data[2,N,..] == -adjoint(ggf[N,2])
end

@testset "4D GF" begin
  # Test 4d getindex setindex!
  data = rand(ComplexF64, N, N, N, N)
  data -= permutedims(data, [1,2,4,3])

  lgf = GreenFunction(copy(data), Lesser)
  ggf = GreenFunction(copy(data), Greater)

  v = rand(ComplexF64, N, N)
  lgf[2,N] = v
  ggf[N,2] = v

  @test lgf.data[N,2,..] == -adjoint(v)
  @test ggf.data[2,N,..] == -adjoint(v)

  @test lgf[2,N] == -adjoint(lgf.data[N,2,..])
  @test ggf.data[2,N,..] == -adjoint(ggf[N,2])
end

@testset "Base functions" begin
  # Test AbstractArray-like behaviour
  data = rand(ComplexF64, N, N, N, N)
  gf = GreenFunction(copy(data), Lesser)

  @test (-gf).data == (-data)
  @test (conj(gf)).data == conj(data)
  @test (real(gf)).data == real(data)
  @test (imag(gf)).data == imag(data)
  # @test (adjoint(gf)).data == adjoint(data)
  # @test (transpose(gf)).data == transpose(data)
  # @test (inv(gf)).data == inv(data)

  temp = rand(ComplexF64, N, N)
  gf[1,2] = temp
  data[1,2,..] = temp; data[2,1,..] = -adjoint(temp)
  @test gf.data == data

  # NOTE: using `..` for the first indices is not supported!
  # gf[:,:,1,1] = temp
  # data[:,:,1,1] = temp
  # @test gf.data == data
end

@testset "setindex! benchmarks" begin
  function setindexA(A::AbstractArray)
    for i=1:N, j=1:i
      A[i,j,..] = b
      if j != i
        A[j,i,..] = -adjoint(b)
      end
    end
  end

  function setindexG(G)
    for i=1:N, j=1:i
      G[i,j] = b
    end
  end

  b = rand(ComplexF64,N,N)
  data = rand(ComplexF64, N, N, N, N);
  gf = GreenFunction(copy(data), Lesser);

  setindexA(data)
  setindexG(gf)
  @test gf.data == data

  @show @btime $setindexA($data)
  @show @btime $setindexG($gf)
end

@testset "broadcasting" begin

end
