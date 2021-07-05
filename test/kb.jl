@testset "1-time benchmark" begin
  λ = 0.2
  atol = 1e-7
  rtol = 1e-5

  function fv(out, ts, t, t′)
    out[1] = 1im * L[t, t′]
    out[2] = 1im * G[t, t′] - λ * L[t, t′]
  end

  function fd(out, ts, t)
    out[1] = 0im * L[t, t]
    out[2] = 0im * G[t, t]
  end

  G = GreenFunction(zeros(ComplexF64, 1, 1), SkewHermitian)
  L = GreenFunction(1im * ones(ComplexF64, 1, 1), SkewHermitian)

  kb = kbsolve!(fv, fd, [G, L], (0.0, 30.0); atol=atol, rtol=rtol)

  function sol1(t, g0, l0)
    s = sqrt(Complex(λ^2 - 4))
    return exp(-λ * t / 2) * (l0 * cosh(0.5 * t * s) + (2im * g0 + l0 * λ) * sinh(0.5 * t * s) / s)
  end

  function sol2(t, g0, l0)
    s = sqrt(Complex(λ^2 - 4))
    return exp(-λ * t / 2) * (g0 * cosh(0.5 * t * s) + (2im * l0 - g0 * λ) * sinh(0.5 * t * s) / s)
  end

  @testset begin
    @test G[:, 1] ≈ [sol1(t1, L[1, 1], G[1, 1]) for t1 in kb.t] atol = 1e1atol rtol = 1e1rtol
    @test L[:, 1] ≈ [sol2(t1, L[1, 1], G[1, 1]) for t1 in kb.t] atol = 1e1atol rtol = 1e1rtol
  end
end

@testset "1-time Volterra benchmark" begin
  atol = 1e-7
  rtol = 1e-5

  function kv!(out, times, t, t′, s)
    out[1] = G[t′, s]
  end
  function kd!(out, times, t, s)
    out[1] = 0.0
  end
  function fv!(out, v, times, t, t′)
    out[1] = (1 - v[1])
  end
  function fd!(out, v, times, t)
    out[1] = 0.0
  end

  G = GreenFunction(ones(1, 1), Symmetrical)

  kb = kbsolve!(fv, fd, [G], (0.0, 30.0); atol=atol, rtol=rtol, k_vert=kv, k_diag=kd)

  sol(t) = cos(t) + sin(t)

  @test G[:, 1] ≈ [sol(t1) for t1 in kb.t] atol = 1e1atol rtol = 1e1rtol
end

@testset "2-time benchmark" begin
  λ = 0.2
  atol = 1e-8
  rtol = 1e-5

  function fv(out, ts, t, t′)
    out[1] = -1im * λ * cos(λ * (ts[t] - ts[t′])) * L[t, t′]
  end

  function fd(out, ts, t)
    out[1] = 0im * L[t, t]
  end

  L = GreenFunction(-1im * ones(ComplexF64, 1, 1), SkewHermitian)

  kb = kbsolve!(fv, fd, [L], (0.0, 200.0); atol=atol, rtol=rtol)

  sol(t, t′) = -1.0im * exp(-1.0im * sin(λ * (t - t′)))

  @test L.data ≈ [sol(t1, t2) for t1 in kb.t, t2 in kb.t] atol = 1e1atol rtol = 1e1rtol
end

@testset "2-time Volterra benchmark" begin
  atol = 1e-8
  rtol = 1e-5

  function kv!(out, times, t, t′, s)
    out[1] = s >= t′ ? G[t′, s] : 0.0
  end
  function kd!(out, times, t, s)
    out[1] = 0.0
  end
  function fv!(out, v, times, t, t′)
    out[1] = (1 - v[1])
  end
  function fd!(out, v, times, t)
    out[1] = 0.0
  end

  G = GreenFunction(ones(1, 1), Symmetrical)

  kb = kbsolve!(fv, fd, [G], (0.0, 1.0); atol=atol, rtol=rtol, k_vert=kv, k_diag=kd)

  function sol(t)
    return cos(t) + sin(t)
  end

  sol_ = hcat([vcat(sol.(kb.t[i] .- kb.t[1:i]), sol.(kb.t[(1 + i):length(kb.t)] .- kb.t[i])) for i in eachindex(kb.t)]...)

  @test G.data ≈ sol_ atol = 1e1atol rtol = 5e2rtol
end
