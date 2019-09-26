using Test

include("../src/gf.jl")

N = 3

data = rand(Complex{Float64}, N, N)
data -= transpose(data)

lgf = LesserGF(data)
ggf = GreaterGF(data)

v = 30 + 30im
lgf[2, N] = 30 + 30im
ggf[2, N] = 30 + 30im

@test lgf[N, 2] == -adjoint(v)
@test ggf[N, 2] == -adjoint(v)

@test lgf[2, N] == -adjoint(lgf[N, 2])
@test ggf[2, N] == -adjoint(ggf[N, 2])

# For special index set
@test lgf.data[2, N] != lgf.data[N, 2]
@test ggf.data[2, N] != ggf.data[N, 2]

