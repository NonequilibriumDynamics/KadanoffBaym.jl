@testset "Causal piecewise Volterra integration" begin
  function solve_piecewise(initial, tol, dtini, dtmax; options...)
    T = eltype(initial)
    G = GreenFunction([cos(t - s) + sin(abs(t - s)) for t in initial, s in initial], Symmetrical)
    seen = Dict{Tuple{T, Int}, Vector{T}}()
    consistent = Ref(true)
    attempted = Set{T}()
    function callback(ts, w1, w2, i, j)
      seen[(ts[i], j)] = collect(w1)
      i == j && push!(attempted, ts[i])
      nothing
    end
    function fv!(out, ts, w1, w2, i, j)
      key = (ts[i], j)
      if haskey(seen, key)
        consistent[] &= collect(w1) == seen[key]
      end
      out[1] = 1 - sum(w1[s] * G[s, j] for s in eachindex(w1)) + sum(w2[s] * G[s, j] for s in eachindex(w2))
      nothing
    end
    fd!(out, args...) = (out[1] = 0; nothing)
    sol = kbsolve!(fv!, fd!, [G], (initial, one(T)); atol=tol, rtol=tol, dtini, dtmax, quadrature=:piecewise, callback, options...)
    error = maximum(abs(G[i, j] - cos(t - s) - sin(abs(t - s))) for (i, t) in enumerate(sol.t), (j, s) in enumerate(sol.t))
    return (; sol, G, error, consistent=consistent[], attempts=length(attempted))
  end

  errors = Float64[]
  for h in (1/8, 1/16, 1/32, 1/64)
    result = solve_piecewise([0.0], 1e3, h, h)
    push!(errors, result.error)
    @test result.consistent
  end
  @test all(6 .< errors[1:end-1] ./ errors[2:end] .< 10)

  for (initial, tol) in (([0.0], 1e-7), ([0.0, 0.03, 0.1, 0.2], 1e-9), ([0f0], 2f-5))
    result = solve_piecewise(initial, tol, 1.0, 0.03)
    @test result.sol.t[end] == 1
    @test result.consistent
    @test result.attempts > length(result.sol.t) - length(initial)
    @test result.error < 1e-4 # The supplied coarse history retains its trapezoidal startup error.
    @test all(w -> w isa Vector{eltype(initial)}, result.sol.w)
  end
  capped = solve_piecewise([0.0], 1e-7, 0.1, 0.05; kmax=12, kmax_vie=6)
  explicit = solve_piecewise([0.0], 1e-7, 0.1, 0.05; kmax=3, kmax_vie=2)
  @test capped.sol == explicit.sol
  @test capped.G.data == explicit.G.data

  for kmax_vie in 0:1, kmax in (1, 3)
    G = GreenFunction(ones(1, 1), Symmetrical)
    f!(out, args...) = (out[1] = 0; nothing)
    sol = kbsolve!(f!, f!, [G], (0.0, 0.1); dtini=0.01, dtmax=0.01, kmax, kmax_vie, quadrature=:piecewise)
    @test all(isone, G.data)
    @test sum(sol.w[end]) ≈ 0.1
  end
end

@testset "Matrix RHS buffers preserve accepted history" begin
  H = ComplexF64[0.3 0.4; 0.4 -0.3]
  G0 = ComplexF64[-im 0; 0 -2im]
  solutions = []
  for storage in (:dense, :blocks, :packed), inplace in (false, true)
    G = if storage == :packed
      KadanoffBaym.SkewHermitianArray(copy(G0))
    else
      data = storage == :blocks ? reshape([copy(G0)], 1, 1) : reshape(copy(G0), 2, 2, 1, 1)
      GreenFunction(data, SkewHermitian)
    end
    function fv!(out, ts, w1, w2, t1, t2)
      F = -im * H * G[t1, t2]
      if inplace
        out[1] .= F
      else
        out[1] = F
      end
      nothing
    end
    function fd!(out, ts, w1, w2, t1, t2)
      F = -im * (H * G[t1, t2] - G[t1, t2] * H)
      if inplace
        out[1] .= F
      else
        out[1] = F
      end
      nothing
    end
    sol = kbsolve!(fv!, fd!, [G], (0.0, 1.0); atol=1e-8, rtol=1e-8, dtini=1e-3, dtmax=0.02)
    @test G[1, 1] == G0
    @test maximum(maximum(abs.(G[i, j] - exp(-im * H * t1) * G0 * exp(im * H * t2)))
                  for (i, t1) in enumerate(sol.t), (j, t2) in enumerate(sol.t)) < 1e-7
    push!(solutions, (t=sol.t, G=[G[i, j] for i in eachindex(sol.t), j in eachindex(sol.t)]))
  end
  for sol in solutions[2:end]
    @test sol.t == solutions[1].t
    @test sol.G == solutions[1].G
  end
end

@testset "A diagonal cusp uses only physical RHS history" begin
  tol = 1e-7
  for times in ([0.0], [0.0, 0.05, 0.12, 0.2])
    G = GreenFunction([exp(-abs(t - s)) for t in times, s in times], Symmetrical)
    function fv!(out, ts, w1, w2, i, j)
      @assert i >= j
      out[1] = -G[i, j]
      nothing
    end
    fd!(out, args...) = (out[1] = 0; nothing)
    sol = kbsolve!(fv!, fd!, [G], (times, 1.0); atol=tol, rtol=tol)
    @test last(sol.t) == 1.0
    @test length(sol.t) < 100
    @test maximum(abs.(G.data .- [exp(-abs(t - s)) for t in sol.t, s in sol.t])) < 3tol
  end
end

@testset "Free level startup and rejected endpoint" begin
  for dtini in (1e-6, 1e-2, 1.0)
    G = GreenFunction(fill(-1.0im, 1, 1), SkewHermitian)
    fv!(out, ts, w1, w2, t1, t2) = (out[1] = -im * G[t1, t2]; nothing)
    fd!(out, ts, w1, w2, t1, t2) = (out[1] = 0; nothing)
    sol = kbsolve!(fv!, fd!, [G], (0.0, 1.0); atol=1e-8, rtol=1e-8, dtini)

    @test sol.t[end] == 1.0
    @test all(>(0), diff(sol.t))
    @test maximum(abs.(G.data .- [-im * exp(-im * (t1 - t2)) for t1 in sol.t, t2 in sol.t])) < 1e-6
  end
end

@testset "One-time variables control accuracy" begin
  errors = Float64[]
  for tol in (1e-6, 1e-9)
    G = GreenFunction(ones(1, 1), Symmetrical)
    v = GreenFunction(ones(ComplexF64, 1), OnePoint)
    f0!(out, ts, w1, w2, t1, t2) = (out[1] = 0; nothing)
    f1!(out, ts, w, t) = (out[1] = -10im * v[t]; nothing)
    sol = kbsolve!(f0!, f0!, [G], (0.0, 1.0); v0=[v], f1!, atol=tol, rtol=tol, dtini=1e-3, dtmax=0.1)

    @test sol.t[end] == 1.0
    push!(errors, maximum(abs.(v.data .- exp.(-10im .* sol.t))))
    @test errors[end] < 100tol
  end
  @test errors[2] < errors[1] / 100
end

@testset "Time-grid types and continuation" begin
  for (t0, tmax) in ((0, 1), (0f0, 0.1), (0f0, 0.1f0), ([0.0, 0.03, 0.1, 0.2], 1.0), (0.0, 0.0))
    times = t0 isa Real ? [t0] : t0
    G = GreenFunction([-im * exp(-im * (t1 - t2)) for t1 in times, t2 in times], SkewHermitian)
    fv!(out, ts, w1, w2, t1, t2) = (out[1] = -im * G[t1, t2]; nothing)
    fd!(out, ts, w1, w2, t1, t2) = (out[1] = 0; nothing)
    sol = kbsolve!(fv!, fd!, [G], (t0, tmax); atol=1e-7, rtol=1e-7, dtmax=0.05)

    @test sol.t[end] == tmax
    @test all(>(0), diff(sol.t))
    @test G[:, 1] ≈ -im .* exp.(-im .* (sol.t .- sol.t[1])) atol=1e-5 rtol=1e-5
  end
end

@testset "Volterra continuation preserves integration accuracy" begin
  for tol in (1e-6, 1e-9)
    G = GreenFunction(ones(1, 1), Symmetrical)
    v = GreenFunction([1.0], OnePoint)
    f0!(out, ts, w1, w2, t1, t2) = (out[1] = 0; nothing)
    f1!(out, ts, w, t) = (out[1] = 1 - sum(w[i] * v[i] for i in eachindex(w)); nothing)

    first = kbsolve!(f0!, f0!, [G], (0.0, 1.0); v0=[v], f1!, atol=tol, rtol=tol)
    second = kbsolve!(f0!, f0!, [G], (first.t, 2.0); v0=[v], f1!, atol=tol, rtol=tol)
    @test second.t[end] == 2.0
    @test maximum(abs.(v.data .- cos.(second.t) .- sin.(second.t))) < 20tol
  end
end

@testset "Time steps advance representable times" begin
  for offset in (1e12, -1e12, 1f4)
    G = GreenFunction(fill(-1.0im, 1, 1), SkewHermitian)
    fv!(out, ts, w1, w2, i, j) = (out[1] = -im * G[i, j]; nothing)
    fd!(out, args...) = (out[1] = 0; nothing)
    tmax = offset + oftype(offset, 0.1)
    sol = kbsolve!(fv!, fd!, [G], (offset, tmax); atol=1e-6, rtol=1e-6)
    @test last(sol.t) == tmax
    @test all(>(0), diff(sol.t))
    @test all(w -> all(isfinite, w), sol.w)
    @test G[length(sol.t), 1] ≈ -im * exp(-im * (tmax - offset)) atol=1e-5 rtol=1e-5
  end

  G = GreenFunction(ones(1, 1), Symmetrical)
  f0!(out, args...) = (out[1] = 0; nothing)
  sol = kbsolve!(f0!, f0!, [G], (0.0, 1.0); dtini=1e-20)
  @test last(sol.t) == 1.0
  @test all(>(0), diff(sol.t))
  @test all(isone, G.data)

  G = GreenFunction(ones(1, 1), Symmetrical)
  sol = kbsolve!(f0!, f0!, [G], (0.0, Inf); stop=ts -> length(ts) == 8)
  @test length(sol.t) == 8
  @test all(isfinite, sol.t)
  @test all(>(0), diff(sol.t))

  for options in ((dtini=1e-10,), (dtmax=1e-10,), (dtmax=0.75 * (nextfloat(1e12) - 1e12),))
    G = GreenFunction(ones(1, 1), Symmetrical)
    @test_throws ErrorException kbsolve!(f0!, f0!, [G], (1e12, 1e12 + 1); options...)
  end
  G = GreenFunction(fill(-1.0im, 1, 1), SkewHermitian)
  f_reject!(out, ts, w1, w2, i, j) = (out[1] = -im * G[i, j]; nothing)
  attempts = Ref(0)
  stop(ts) = (attempts[] += 1; attempts[] > 8)
  @test_throws ErrorException kbsolve!(f_reject!, f0!, [G], (1e12, 1e12 + 0.1); atol=2e-9, rtol=2e-9, stop)
  @test attempts[] < 8
  G = GreenFunction(ones(2, 2), Symmetrical)
  @test_throws AssertionError kbsolve!(f0!, f0!, [G], ([0.0, 0.0], 1.0))
end

@testset "Each function retains its own storage capacity" begin
  for capacities in ((1, 8), (8, 1), (8, 8))
    G = [GreenFunction(fill(Float64(i), n, n), Symmetrical) for (i, n) in enumerate(capacities)]
    v = GreenFunction(fill(3.0, 1), OnePoint)
    f0!(out, args...) = (out .= 0; nothing)
    sol = kbsolve!(f0!, f0!, G, (0.0, 0.1); v0=[v], f1! =f0!, dtini=0.01, dtmax=0.01)
    @test last(sol.t) == 0.1
    @test all(g -> size(g) == (length(sol.t), length(sol.t)), G)
    @test all(i -> all(==(i), G[i].data), eachindex(G))
    @test size(v) == (length(sol.t),)
    @test all(==(3), v.data)
  end
end

@testset "Coupled correctors use the same predicted state" begin
  G = GreenFunction(fill(2.0, 1, 1), Symmetrical)
  v = GreenFunction([3.0], OnePoint)
  fv!(out, ts, w1, w2, t1, t2) = (out[1] = 0; nothing)
  fd!(out, ts, w1, w2, t1, t2) = (out[1] = v[t1]; nothing)
  f1!(out, ts, w, t) = (out[1] = G[t, t]; nothing)
  h = 0.1
  sol = kbsolve!(fv!, fd!, [G], (0.0, h); v0=[v], f1!, dtini=h, atol=1.0, rtol=0.0)

  @test sol.t == [0.0, h]
  @test G[end, end] ≈ 2 + 3h + 2h^2 / 2
  @test v[end] ≈ 3 + 2h + 3h^2 / 2
end

@testset "Maximum step and maximum order" begin
  G = GreenFunction(ones(1, 1), Symmetrical)
  f0!(out, ts, w1, w2, t1, t2) = (out[1] = 0; nothing)
  sol = kbsolve!(f0!, f0!, [G], (0.0, 0.3); dtini=0.5, dtmax=0.1)
  @test sol.t[end] == 0.3
  @test maximum(diff(sol.t)) <= 0.1 * (1 + 8eps())

  G = GreenFunction(ones(1, 1), Symmetrical)
  fv!(out, ts, w1, w2, t1, t2) = (out[1] = G[t1, t2]; nothing)
  fd!(out, ts, w1, w2, t1, t2) = (out[1] = 2G[t1, t2]; nothing)
  sol = kbsolve!(fv!, fd!, [G], (0.0, 2.0); atol=1e-4, rtol=1e-4, dtini=0.001, dtmax=0.1, kmax=3, stop=ts -> length(ts) > 80)
  @test sol.t[end] == 2.0
  @test length(sol.t) < 40
end

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
  v(t) = 1 / 2 + cos(sqrt(2) * t) / 2 - sin(sqrt(2) * t) / sqrt(2)
  exact = [cos(t - s) + v(min(t, s)) * sin(abs(t - s)) for t in kb.t, s in kb.t]
  @test G.data ≈ exact atol=atol rtol=5rtol
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

@testset "Fermionic dimer" begin
  atol = 1e-8
  rtol = 1e-6

  dim = 2
  GL = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)
  GG = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)

  N_0 = 1.0
  GL[1, 1] = +im * diagm([N_0, 0.0])
  GG[1, 1] = -im * I(2) + GL[1, 1]

  H = ComplexF64[1.0/20 1.0; 1.0 -1.0/20]

  function fv!(out, _, _, _, t1, t2)
    out[1] = -1.0im * H * GL[t1, t2]
    out[2] = -1.0im * H * GG[t1, t2]
  end

  function fd!(out, _, _, _, t1, t2)
    fv!(out, nothing, nothing, nothing, t1, t2)
    out[1] .-= adjoint(out[1])
    out[2] .-= adjoint(out[2])
  end

  ana(i1, i2, t1, t2) = (exp(-1.0im * H * t1) * GL[:, :, 1, 1] * exp(1.0im * H * t2))[i1, i2]

  sol = kbsolve!(fv!, fd!, [GL, GG], (0.0, 5.0); atol=atol, rtol=rtol)

  exact = [ana(i, j, t1, t2) for i in 1:2, j in 1:2, t1 in sol.t, t2 in sol.t]
  @test GL.data ≈ exact atol=1e-3 rtol=1e-3
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

  sol = kbsolve!(fv!, fd!, [F], (0.0, T); atol=atol, rtol=rtol, dtini=1e-6)

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

  sol = kbsolve!(fv!, fd!, [S, F], (0.0, T); atol=atol, rtol=rtol, dtini=1e-6)

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

@testset "Initial step includes vertical evolution" begin
  for θ in ([0.0], [0.0, 0.05, 0.2]), ω in (1e-6, 1.0, 1e6)
    G = GreenFunction([1 + abs(t - s) for t in θ, s in θ], Symmetrical)
    fv!(out, ts, w1, w2, i, j) = (out[1] = ω; nothing)
    fd!(out, args...) = (out[1] = 0; nothing)
    sol = kbsolve!(fv!, fd!, [G], (θ / ω, 1 / ω); atol=1e-6, rtol=1e-6, stop=ts -> length(ts) == length(θ) + 1)
    u = vcat(1 .+ last(θ) .- θ, [1.0])
    h = maximum(u ./ (1 .+ u)) / maximum(1 ./ (1 .+ u[1:end-1])) / 100
    @test ω * (sol.t[end] - sol.t[end-1]) ≈ h
    @test G[end, end-1] ≈ 1 + h
  end
end

@testset "Infinite error can reject a finite trial" begin
  cache = KadanoffBaym.VCABMCache{Float64}(3, [[0.0]])
  for h in (1.0, 1e-160)
    times = [0.0, h]
    f!(out) = (out[1][1] = 1e300 * h; nothing)
    KadanoffBaym.predict!(cache, times)
    KadanoffBaym.correct!(cache, f!)
    KadanoffBaym.adjust!(cache, times, f!, 3, 1e-10, 0.0)
    @test all(isfinite, only(cache.u_next))
    if h == 1
      @test cache.error_k == Inf
      @test cache.steps == 0
      @test cache.u_prev == [[0.0]]
    else
      @test cache.error_k < 1
      @test cache.steps == 1
      @test only(only(cache.u_prev)) ≈ (1e300 * h) * h / 2
    end
  end
end

@testset "Undefined trial errors cannot be accepted" begin
  for onepoint in (false, true), overflow in (false, true)
    G = GreenFunction(ones(1, 1), Symmetrical)
    v = GreenFunction(ones(1), OnePoint)
    rhs(y, t) = overflow ? -1e200 * y : (t > 0.5 ? NaN : -y)
    fv!(out, ts, w1, w2, i, j) = (out[1] = onepoint ? 0 : rhs(G[i, j], ts[i]); nothing)
    fd!(out, args...) = (out[1] = 0; nothing)
    f1!(out, ts, w, i) = (out[1] = rhs(v[i], ts[i]); nothing)
    @test_throws r"Local error is NaN at t = 1.0" kbsolve!(fv!, fd!, [G], (0.0, 1.0);
      v0=onepoint ? [v] : [], f1!, atol=1e-8, rtol=1e-8, dtini=1.0)
  end
end

@testset "Piecewise matrix memory integrals" begin
  A = [0.05 0.1; 0.1 -0.05]
  B = -[1.0 0.2; 0.2 0.6]
  C = [0.8 -0.3; -0.3 1.2]
  F₀ = im * [1.0 0.2; 0.2 0.5]
  # F′ = AF + BYC, Y′ = F gives G(t, s) = F(t - s) on the lower triangle.
  M = [kron(Matrix{Float64}(I, 2, 2), A) kron(transpose(C), B);
    Matrix{Float64}(I, 4, 4) zeros(4, 4)]
  F(t) = reshape((exp(t * M) * [vec(F₀); zeros(ComplexF64, 4)])[1:4], 2, 2)
  errors = Float64[]

  for h in (1/8, 1/16, 1/32, Inf)
    G = GreenFunction(reshape(copy(F₀), 2, 2, 1, 1), SkewHermitian)
    function fv!(out, ts, w1, w2, i, j)
      integral = sum(w1[s] * G[s, j] for s in eachindex(w1)) -
        sum(w2[s] * G[s, j] for s in eachindex(w2))
      out[1] .= A * G[i, j] + B * integral * C
      return nothing
    end
    fd!(out, args...) = (out[1] .= 0; nothing)

    if isfinite(h)
      sol = kbsolve!(fv!, fd!, [G], (0.0, 1.0);
        quadrature=:piecewise, atol=1e3, rtol=1e3, dtini=h, dtmax=h)
    else
      first = kbsolve!(fv!, fd!, [G], (0.0, 0.4);
        quadrature=:piecewise, atol=1e-7, rtol=1e-7, dtini=0.8)
      n = length(first.t)
      sol = kbsolve!(fv!, fd!, [G], (first.t, 1.0);
        quadrature=:piecewise, atol=1e-7, rtol=1e-7, dtini=0.6)
      @test sol.t[2] - sol.t[1] < 0.01
      @test sol.t[n+1] - sol.t[n] < 0.01
      @test G[1, length(sol.t)] == -adjoint(G[length(sol.t), 1])
    end
    error = maximum(norm(G[i, j] - F(sol.t[i] - sol.t[j]))
      for i in eachindex(sol.t) for j in 1:i)
    isfinite(h) ? push!(errors, error) : (@test error < 5e-5)
  end
  @test all(7 .< errors[1:2] ./ errors[2:3] .< 9)
end

@testset "Piecewise continuation preserves constant integrals" begin
  G = GreenFunction(ones(1, 1), Symmetrical)
  defect = Ref(0.0)
  function fv!(out, ts, w1, w2, i, j)
    out[1] = sum(w1) - (ts[i] - ts[1])
    defect[] = max(defect[], abs(out[1]))
    return nothing
  end
  fd!(out, args...) = (out[1] = 0; nothing)
  times = [0.0]
  for endpoint in (0.1, 0.13, 0.14)
    sol = kbsolve!(fv!, fd!, [G], (times, endpoint);
      quadrature=:piecewise, dtini=0.01, dtmax=0.01, atol=1e-8, rtol=1e-8,
      stop=ts -> length(ts) > 100)
    @test sol.t[end] == endpoint
    @test defect[] < 100eps()
    @test maximum(abs.(G.data .- 1)) < 100eps()
    @test all(abs(sum(w) - (t - sol.t[1])) < 100eps() for (t, w) in zip(sol.t, sol.w))
    times = sol.t
  end
end

@testset "Piecewise trials after strongly graded history" begin
  times = [0.0, 1e-16, 0.1]
  G = GreenFunction([exp(-abs(t - s)) for t in times, s in times], Symmetrical)
  seen = Dict{Tuple{Float64, Int}, Vector{Float64}}()
  consistent = Ref(true)
  attempted = Set{Float64}()
  function callback(ts, w1, w2, i, j)
    seen[(ts[i], j)] = collect(w1)
    i == j && push!(attempted, ts[i])
    return nothing
  end
  function fv!(out, ts, w1, w2, i, j)
    key = (ts[i], j)
    haskey(seen, key) && (consistent[] &= collect(w1) == seen[key])
    out[1] = -G[i, j] + sum(w1) - (ts[i] - ts[1])
    return nothing
  end
  fd!(out, args...) = (out[1] = 0; nothing)
  sol = kbsolve!(fv!, fd!, [G], (times, 0.3);
    quadrature=:piecewise, atol=1e-9, rtol=1e-9, dtini=1.0, callback,
    stop=ts -> length(ts) > 100)
  @test sol.t[end] == 0.3
  @test sol.t[4] - sol.t[3] < 0.01
  @test length(attempted) > length(sol.t) - length(times)
  @test consistent[]
  @test maximum(abs.(G.data .- [exp(-abs(t - s)) for t in sol.t, s in sol.t])) < 3e-8
end

@testset "Nonlinear piecewise memory integral" begin
  errors = Float64[]
  for h in (1/8, 1/16, 1/32, 1/64)
    G = GreenFunction(ones(1, 1), Symmetrical)
    function fv!(out, ts, w1, w2, i, j)
      integral = ((1 + ts[j])^3 + (1 + ts[i] - ts[j])^3 - 2) / 3
      out[1] = 1 + sum(w1[s] * G[s, j]^2 for s in eachindex(w1)) - integral
      return nothing
    end
    fd!(out, args...) = (out[1] = 0; nothing)
    sol = kbsolve!(fv!, fd!, [G], (0.0, 1.0);
      quadrature=:piecewise, atol=1e3, rtol=1e3, dtini=h, dtmax=h)
    push!(errors, maximum(abs(G[i, j] - 1 - abs(t - s))
      for (i, t) in enumerate(sol.t), (j, s) in enumerate(sol.t)))
  end
  @test all(6 .< errors[1:end-1] ./ errors[2:end] .< 10)
end

@testset "Inactive fields do not dilute adaptive error control" begin
  function with_inactive_fields(n)
    G = [GreenFunction(fill(a == 1 ? -1.0im : 0.0im, 1, 1), SkewHermitian) for a in 1:n]
    function fv!(out, t, w1, w2, i, j)
      out[1] = -im * G[1][i, j]
      for a in 2:n
        out[a] = 0
      end
    end
    fd!(out, args...) = fill!(out, 0)
    sol = kbsolve!(fv!, fd!, G, (0.0, 1.0); atol=1e-7, rtol=1e-7)
    return sol.t, G
  end
  t, G = with_inactive_fields(1)
  padded_t, padded_G = with_inactive_fields(16)
  @test t == padded_t
  @test G[1].data == padded_G[1].data
  @test all(g -> iszero(g.data), padded_G[2:end])
end

@testset "RHS callbacks may reuse a matrix workspace between calls" begin
  A = [1.0 0.25; 0.25 2.0]
  for initial in ([0.0], [0.0, 0.25, 0.5])
    G = GreenFunction([min(t, s) * A for t in initial, s in initial], Symmetrical)
    v = GreenFunction([2t * A for t in initial], OnePoint)
    work = zero(A)
    fv!(out, ts, w1, w2, i, j) = (out[1] = zero(A); nothing)
    fd!(out, ts, w1, w2, i, j) = (work .= A; out[1] = work; nothing)
    f1!(out, ts, w, i) = (work .= 2A; out[1] = work; nothing)
    sol = kbsolve!(fv!, fd!, [G], (initial, 1.0); v0=[v], f1!,
                   dtini=0.125, dtmax=0.125, atol=1.0, rtol=0.0)
    @test sol.t == [initial; collect(last(initial)+0.125:0.125:1.0)]
    @test all(isapprox(G[i, j], min(t, s) * A; atol=1e-14, rtol=0) for (i, t) in enumerate(sol.t), (j, s) in enumerate(sol.t))
    @test all(isapprox(v[i], 2t * A; atol=1e-14, rtol=0) for (i, t) in enumerate(sol.t))
  end
end

@testset "Each time-row derivative is captured before workspace reuse" begin
  A = [1.0 0.25; 0.25 2.0]
  G = GreenFunction(reshape([zero(A)], 1, 1), Symmetrical)
  work = [zero(A) for _ in 1:Threads.maxthreadid()]
  function fv!(out, ts, w1, w2, i, j)
    buffer = work[Threads.threadid()]
    buffer .= ts[j] .* A
    out[1] = buffer
    nothing
  end
  function fd!(out, ts, w1, w2, i, j)
    buffer = work[Threads.threadid()]
    buffer .= (2ts[i]) .* A
    out[1] = buffer
    nothing
  end
  sol = kbsolve!(fv!, fd!, [G], (0.0, 1.0); dtini=0.125, dtmax=0.125, atol=1.0, rtol=0.0)
  @test sol.t == collect(0.0:0.125:1.0)
  @test all(isapprox(G[i, j], t * s * A; atol=1e-14, rtol=0) for (i, t) in enumerate(sol.t), (j, s) in enumerate(sol.t))
end

@testset "Automatic startup uses the coupled field norm" begin
  function linear_startup(scale, dummy)
    G = GreenFunction(fill(scale, 1, 1), Symmetrical)
    v = GreenFunction([scale], OnePoint)
    fv!(out, args...) = (out[1] = scale; nothing)
    fd!(out, args...) = (out[1] = 2scale; nothing)
    f1!(out, args...) = (out[1] = 0; nothing)
    sol = kbsolve!(fv!, fd!, [G], (0.0, 1.0); v0=dummy ? [v] : [], f1!, atol=scale * 1e-8, rtol=1e-8)
    return sol.t, G.data ./ scale
  end
  times, _ = linear_startup(1.0, false)
  @test times[2] == 0.005
  for scale in (0.125, 1.0, 8.0), dummy in (false, true)
    ts, values = linear_startup(scale, dummy)
    @test ts == times
    @test values ≈ [1 + t + s for t in ts, s in ts] atol=1e-14 rtol=0
  end

  grids = []
  for onepoint in (false, true)
    G = GreenFunction(ones(1, 1), Symmetrical)
    v = GreenFunction([1.0], OnePoint)
    fv!(out, args...) = (out[1] = 0; nothing)
    fd!(out, args...) = (out[1] = onepoint ? 0 : 1; nothing)
    f1!(out, args...) = (out[1] = onepoint ? 1 : 0; nothing)
    sol = kbsolve!(fv!, fd!, [G], (0.0, 1.0); v0=[v], f1!, atol=1e-8, rtol=1e-8)
    push!(grids, sol.t)
    @test sol.t[2] == 0.01
    @test G.data ≈ [1 + (onepoint ? 0 : min(t, s)) for t in sol.t, s in sol.t] atol=1e-14 rtol=0
    @test v.data ≈ 1 .+ (onepoint ? sol.t : zero(sol.t)) atol=1e-14 rtol=0
  end
  @test grids[1] == grids[2]
end
