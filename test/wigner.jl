using FFTW

const ft = KadanoffBaym.ft

N = 100

a = rand(ComplexF64, N, N)
a .-= a'

w, _ = wigner_transform(a; fourier = false)

xs = range(-3.0, 3.0, length = N - 1) # needs to be odd and symmetric for double FFT to yield itself
ys = (x -> sin(x^2)).(xs)

@test ft(ft(xs, ys; inverse = true)...; inverse = false)[1] ≈ xs
@test ft(ft(xs, ys; inverse = true)...; inverse = false)[2] ≈ ys
