using Test

using LinearAlgebra
using EllipsisNotation
include("../src/utils.jl")
include("../src/gf.jl")

N = 10

# Test 2d getindex setindex!
data = rand(ComplexF64, N, N)
data -= transpose(data)

lgf = LesserGF(copy(data))
ggf = GreaterGF(copy(data))

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

lgf = LesserGF(copy(data))
ggf = GreaterGF(copy(data))

v = rand(ComplexF64, N, N)
lgf[2,N] = v
ggf[N,2] = v

@test lgf.data[N,2,..] == -adjoint(v)
@test ggf.data[2,N,..] == -adjoint(v)

@test lgf[2,N] == -adjoint(lgf.data[N,2,..])
@test ggf.data[2,N,..] == -adjoint(ggf[N,2])

# Test AbstractArray-like behaviour
data = rand(ComplexF64, N, N, N, N)
gf = LesserGF(copy(data))

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

using BenchmarkTools

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
