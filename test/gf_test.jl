using Test
using BenchmarkTools

using LinearAlgebra
using EllipsisNotation
using RecursiveArrayTools

include("../src/utils.jl")
include("../src/gf.jl")

N = 10

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

# Note: using EllipsisNotation here breaks setindex!
gf[:,:,1,1] = temp
data[:,:,1,1] = temp
@test gf.data == data


function setindexA(A::AbstractArray)
  for i=1:N, j=1:i
    b = rand(ComplexF64,N,N)
    A[i,j,..] = b
    A[j,i,..] = -adjoint(b)
  end
end

function setindexG(G)
  for i=1:N, j=1:i
    G[j,i] = rand(ComplexF64,N,N)
  end
end

@show @btime setindexA($data)
@show @btime setindexG($gf)


# data = [rand(ComplexF64, N, N) for i in 1:N, j in 1:N]
# lgf = LesserGF(copy(data))

# @test lgf[1,2,3,4] == data[1,2][3,4]

# temp = 1.0 + 2.0im
# lgf[1,2,3,4] = temp
# @test lgf.data[1,2][3,4] == temp
# @test lgf.data[2,1][3,4] == -conj(temp)

# temp = rand(ComplexF64, N, N)
# lgf[1,2] = temp
# @test lgf.data[1,2] == temp
# @test lgf.data[2,1] == -conj(temp)
