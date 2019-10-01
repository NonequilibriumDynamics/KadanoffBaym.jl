using Test

include("../src/KadanoffBaym.jl")

N = 3

data = rand(Complex{Float64}, N, N)
data -= transpose(data)

lgf = KadanoffBaym.LesserGF(data)
ggf = KadanoffBaym.GreaterGF(data)

v = 30 + 30im
lgf[2, N] = v
ggf[2, N] = v

@test lgf[N, 2] == -adjoint(v)
@test ggf[N, 2] == -adjoint(v)

@test lgf[2, N] == -adjoint(lgf[N, 2])
@test ggf[2, N] == -adjoint(ggf[N, 2])

# For special index set
@test lgf.data[2, N] != lgf.data[N, 2]
@test ggf.data[2, N] != ggf.data[N, 2]


data = rand(Complex{Float64}, N, N, N, N)
data -= permutedims(data, [1,2,4,3])

lgf = KadanoffBaym.LesserGF(data)
ggf = KadanoffBaym.GreaterGF(data)

v = rand(Complex{Float64}, N, N)
lgf[2, N, :, :] = v
ggf[2, N, :, :] = v

@test lgf[N, 2, :, :] == -adjoint(v)
@test ggf[N, 2, :, :] == -adjoint(v)

@test lgf[2, N, :, :] == -adjoint(lgf[N, 2, :, :])
@test ggf[2, N, :, :] == -adjoint(ggf[N, 2, :, :])

# For special index set
@test lgf.data[2, N, :, :] != lgf.data[N, 2, :, :]
@test ggf.data[2, N, :, :] != ggf.data[N, 2, :, :]