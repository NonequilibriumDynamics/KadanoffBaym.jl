@testset "1-time benchmark" begin
  λ = 0.2
  atol = 1e-7
  rtol = 1e-5

  function fv!(out, ts, h1, h2, t, t′)
    out[1] = 1im * L[t, t′]
    out[2] = 1im * G[t, t′] - λ * L[t, t′]
  end

  function fd!(out, ts, h1, h2, t, t′)
    out[1] = zero(out[1])
    out[2] = zero(out[2])
  end

  function f1!(out, ts, h1, t)
    out[1] = -1.0im * J[t]
  end

  # two-time initial conditions
  G = GreenFunction(zeros(ComplexF64, 1, 1), SkewHermitian)
  L = GreenFunction(1im * ones(ComplexF64, 1, 1), SkewHermitian)

  # one-time initial conditions
  J = GreenFunction(ones(ComplexF64, 1, 1), OnePoint)

  kb = kbsolve!(fv!, fd!, [G, L], (0.0, 30.0); atol=atol, rtol=rtol, v0 = [J,], f1! =f1!)

  function sol1(t, g0, l0)
    s = sqrt(Complex(λ^2 - 4))
    return exp(-λ * t / 2) * (l0 * cosh(0.5 * t * s) + (2im * g0 + l0 * λ) * sinh(0.5 * t * s) / s)
  end

  function sol2(t, g0, l0)
    s = sqrt(Complex(λ^2 - 4))
    return exp(-λ * t / 2) * (g0 * cosh(0.5 * t * s) + (2im * l0 - g0 * λ) * sinh(0.5 * t * s) / s)
  end

  @testset begin
    @test G[:, 1] ≈ [sol1(t1, L[1, 1], G[1, 1]) for t1 in kb.t] atol = atol rtol = rtol
    @test L[:, 1] ≈ [sol2(t1, L[1, 1], G[1, 1]) for t1 in kb.t] atol = atol rtol = rtol
    @test real(J).data[:] ≈ cos.(kb.t) atol = atol rtol = rtol
  end
end

@testset "1-time Volterra benchmark" begin
  atol = 1e-6
  rtol = 1e-3

  function fv!(out, times, h1, h2, t, t′)
    I = sum(h1[s] * G[t′, s] for s in eachindex(h1))
    out[1] = (1 - I)
  end
  function fd!(out, times, h1, h2, t, t′)
    out[1] = zero(out[1])
  end

  G = GreenFunction(ones(1, 1), Symmetrical)

  kb = kbsolve!(fv!, fd!, [G], (0.0, 30.0); atol=atol, rtol=rtol)

  sol(t) = cos(t) + sin(t)

  @test G[:, 1] ≈ [sol(t1) for t1 in kb.t] atol = atol rtol = 2e0rtol
end

@testset "2-time benchmark" begin
  λ = 0.2
  atol = 1e-8
  rtol = 1e-5

  function fv!(out, ts, h1, h2, t, t′)
    out[1] = -1im * λ * cos(λ * (ts[t] - ts[t′])) * L[t, t′]
  end

  function fd!(out, ts, h1, h2, t, t′)
    out[1] = zero(out[1])
  end

  L = GreenFunction(-1im * ones(ComplexF64, 1, 1), SkewHermitian)

  kb = kbsolve!(fv!, fd!, [L], (0.0, 200.0); atol=atol, rtol=rtol)

  sol(t, t′) = -1.0im * exp(-1.0im * sin(λ * (t - t′)))

  @test L.data ≈ [sol(t1, t2) for t1 in kb.t, t2 in kb.t] atol = atol rtol = 2e0rtol
end

@testset "2-time Volterra benchmark" begin
  atol = 1e-8
  rtol = 1e-5

  function fv!(out, times, h1, h2, t, t′)
    I = sum(h1[s] * G[s, t′] for s in eachindex(h1))
    I-= sum(h2[s] * G[s, t′] for s in eachindex(h2))
    out[1] = (1 - I)
  end

  function fd!(out, times, h1, h2, t, t′)
    out[1] = zero(out[1])
  end

  function sol(t)
    return cos(t) + sin(t)
  end

  G = GreenFunction(ones(1, 1), Symmetrical)

  kb = kbsolve!(fv!, fd!, [G], (0.0, 1.0); atol=atol, rtol=rtol)

  sol_ = hcat([vcat(sol.(kb.t[i] .- kb.t[1:i]), sol.(kb.t[(1 + i):length(kb.t)] .- kb.t[i])) for i in eachindex(kb.t)]...)

  @test G.data ≈ sol_ atol = atol rtol = 2e1rtol
end

@testset "Brownian motion" begin
  atol = 1e-9
  rtol = 1e-7
  theta = 1.0
  D = 4.0
  N₀ = 3.0
  T = 4.0

  F = GreenFunction(N₀ * ones(1, 1), Symmetrical)

  function fv!(out, _, _, _, t1, t2)
    out[1] = -theta * F[t1, t2]
  end

  function fd!(out, _, _, _, t1, t2)
    out[1] = -theta * 2F[t1, t2] + D
  end

  sol = kbsolve!(fv!, fd!, [F], (0.0, T); atol=atol, rtol=rtol, dtini=1e-10)

  F_ana(t1, t2) = (N₀ - D / (2theta)) * exp(-theta * (t1 + t2)) + D / (2theta) * exp(-theta * abs(t1 - t2))
  exact = [F_ana(t1, t2) for t1 in sol.t, t2 in sol.t]

  @test F.data ≈ exact atol=1e-5 rtol=1e-4
end

@testset "Geometric Brownian motion" begin
  atol = 1e-9
  rtol = 1e-7
  mu = 1.0
  sigma = 0.5
  S₀ = 2.0
  T = 1.0

  S = GreenFunction(S₀ * ones(1, 1), Symmetrical)
  F = GreenFunction(0.0 * ones(1, 1), Symmetrical)

  function fv!(out, _, _, _, t1, t2)
    out[1] = 0
    out[2] = mu * F[t1, t2]
  end

  function fd!(out, _, _, _, t1, t2)
    out[1] = mu * S[t1, t2]
    out[2] = 2mu * F[t1, t2] + sigma^2 * (S[t1, t2]^2 + F[t1, t2])
  end

  sol = kbsolve!(fv!, fd!, [S, F], (0.0, T); atol=atol, rtol=rtol, dtini=1e-10)

  F_ana(t1, t2) = S₀^2 * exp(mu * (t1 + t2)) * (exp(sigma^2 * min(t1, t2)) - 1)
  exact = [F_ana(t1, t2) for t1 in sol.t, t2 in sol.t]

  @test F.data ≈ exact atol=1e-5 rtol=1e-4
end

@testset "Bose-Einstein condensate" begin
  atol = 1e-6
  rtol = 1e-4
  ω₀ = 1.0
  D = 1.0
  N = 1.0
  δN = 0.0

  φ = GreenFunction(zeros(ComplexF64, 1), OnePoint)
  GK = GreenFunction(zeros(ComplexF64, 1, 1), SkewHermitian)

  GK[1, 1] = -im * (2δN + 1)
  φ[1] = sqrt(2N)

  function fv!(out, ts, h1, h2, t, t′)
    out[1] = zero(out[1])
  end

  function fd!(out, ts, h1, h2, t, _)
    out[1] = -im * (2D * abs2(φ[t]))
  end

  function f1!(out, ts, h1, t)
    out[1] = -im * ω₀ * φ[t] - D * φ[t]
  end

  sol = kbsolve!(fv!, fd!, [GK,], (0.0, 1.0); atol=atol, rtol=rtol, v0=[φ,], f1! = f1!)

  # Analytical: φ(t) = φ(0) exp((-iω₀ - D)t), so |φ(t)|² = 2N exp(-2Dt)
  φ_ana(t) = sqrt(2N) * exp((-im * ω₀ - D) * t)

  @test φ[:] ≈ [φ_ana(t) for t in sol.t] atol=1e-2 rtol=1e-2
  # Condensate occupation should decay exponentially
  @test [abs2(φ[k]) / 2 for k in eachindex(sol.t)] ≈ [N * exp(-2D * t) for t in sol.t] atol=1e-2 rtol=1e-2
end
