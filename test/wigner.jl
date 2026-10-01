using FFTW

const ft = KadanoffBaym.ft

N = 100

a = rand(ComplexF64, N, N)
a .-= a'

w, _ = wigner_transform(a; fourier = false)

@testset "Wigner coordinates and Fourier integral" begin
  for Nt in (2, 5, 6)
    ts = range(0.3, 1.3; length=Nt)
    x = [-im * exp(-0.7im * (t1 - t2)) for t1 in ts, t2 in ts]
    w, (τs, Ts) = wigner_transform(x; ts, fourier=false)
    Δt = step(ts)
    @test Ts == ts
    @test size(w) == (Nt, Nt)

    for T in 1:Nt, (i, τ) in enumerate(τs)
      q = round(Int, τ / (2Δt))
      expected = 1 <= T + q <= Nt && 1 <= T - q <= Nt ? x[T + q, T - q] : 0
      @test w[i, T] ≈ expected
    end

    ŵ, (ωs, Ts) = wigner_transform(x; ts)
    expected = [2Δt * sum(w[j, T] * exp(im * ω * τs[j]) for j in 1:Nt)
                for ω in ωs, T in 1:Nt]
    @test ŵ ≈ expected
    @test Ts == ts
    @test ωs ≈ 2π / (Nt * 2Δt) .* collect((-Nt ÷ 2):((Nt - 1) ÷ 2))

    G = GreenFunction(copy(x), SkewHermitian)
    @test wigner_transform(G; ts, fourier=false)[1] ≈ w
    @test wigner_transform(G; ts)[1] ≈ ŵ
  end

  @test wigner_transform(fill(-im, 1, 1); fourier=false)[1] == fill(-im, 1, 1)
  @test_throws DimensionMismatch wigner_transform(zeros(3, 4); fourier=false)
end

xs = range(-3.0, 3.0, length = N - 1) # needs to be odd and symmetric for double FFT to yield itself
ys = (x -> sin(x^2)).(xs)

@test ft(ft(xs, ys; inverse = true)...; inverse = false)[1] ≈ xs
@test ft(ft(xs, ys; inverse = true)...; inverse = false)[2] ≈ ys

@testset "Single-precision Fourier sums retain their precision" begin
  for Nt in (4, 5)
    ts = range(0.3f0; step=0.1f0, length=Nt)
    x = ComplexF32[-im * exp(-0.7f0im * (t1 - t2)) for t1 in ts, t2 in ts]
    w, (τs, _) = wigner_transform(x; ts, fourier=false)
    ŵ, (ωs, _) = wigner_transform(x; ts)
    expected = [2step(ts) * sum(w[j,T] * exp(im * ω * τs[j]) for j in 1:Nt)
                for ω in ωs, T in 1:Nt]
    @test eltype(τs) == Float32
    @test eltype(ωs) == Float32
    @test eltype(ŵ) == ComplexF32
    @test ŵ ≈ expected atol=1e-6 rtol=1e-6

    for inverse in (false, true)
      ys = x[:, 1]
      ωs, z = ft(ts, ys; inverse)
      expected = [(inverse ? step(ts) / (2Float32(π)) : step(ts)) *
                  sum(ys[j] * exp((inverse ? -im : im) * ω * ts[j]) for j in 1:Nt)
                  for ω in ωs]
      @test eltype(ωs) == Float32
      @test eltype(z) == ComplexF32
      @test z ≈ expected atol=1e-6 rtol=1e-6
    end
  end
end

@testset "Fourier grids require positive uniform spacing" begin
  for ts in (Float64[], [0.0], zeros(3), [0.0, 0.1, 0.4], [0.0, -1.0, -2.0],
             [0.0, Inf], [Inf, Inf], range(0.0; step=Inf, length=3),
             range(prevfloat(2f0); step=eps(prevfloat(2f0)), length=4))
    @test_throws AssertionError wigner_transform(ones(length(ts), length(ts)); ts)
    @test_throws AssertionError ft(ts, ones(length(ts)))
  end
  @test_throws AssertionError ft([0.0, 1.0, 2.0], ones(2))
  w, (τs, _) = wigner_transform(fill(1f0, 1, 1); ts=[0f0], fourier=false)
  @test w == fill(1f0, 1, 1)
  @test τs == [0f0]
  @test eltype(τs) == Float32
end

@testset "Uniform ranges retain their declared spacing" begin
  ts = range(0f0, 1f0; length=8192)
  ωs, z = ft(ts, ones(ComplexF32, length(ts)))
  @test eltype(ωs) == Float32
  @test eltype(z) == ComplexF32
  @test z[findfirst(iszero, ωs)] == length(ts) * step(ts)
  @test count(x -> !iszero(x), z) == 1
  @test KadanoffBaym.uniform_spacing(ts) == step(ts)
end

@testset "Integer Wigner centers omit the other input parity" begin
  A = [1.0 0.2; 0.2 1.0]
  B = [1.0 0.8; 0.8 1.0]
  @test isposdef(A) && isposdef(B)
  @test A != B
  for fourier in (false, true)
    a, axes_a = wigner_transform(A; fourier)
    b, axes_b = wigner_transform(B; fourier)
    @test a == b
    @test axes_a == axes_b
    @test size(a) == size(A)
  end
end
