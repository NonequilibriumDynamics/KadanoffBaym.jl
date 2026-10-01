module QuadratureTests
using KadanoffBaym, LinearAlgebra, Test

const PanelQuadrature = KadanoffBaym.PanelQuadrature

@testset "Volterra quadrature" begin
  @testset "Integration interval and sample domain are independent" begin
    T = Rational{BigInt}
    ts = T[2, 1, 0]
    for (a, b) in ((0, 1), (1, 2), (1//3, 4//3), (1, 0), (2, 2))
      a, b = T(a), T(b)
      w = KadanoffBaym.integration_weights(ts, a, b)
      for p in 0:2
        @test sum(w .* ts.^p) == (b^(p+1) - a^(p+1)) / (p+1)
      end
      @test w == KadanoffBaym.integration_weights(ts, a, (a+b)/2) +
                 KadanoffBaym.integration_weights(ts, (a+b)/2, b)
    end
    @test KadanoffBaym.integration_weights(ts, T(0), T(1)) == T[-1//12, 2//3, 5//12]
    @test KadanoffBaym.integration_weights(T[2], T(0), T(1)) == T[1]
  end

  @testset "Polynomial moments" begin
    for T in (Float32, Float64), scale in T.((1e-8, 1.0, 1e8))
      for x in (T.(0:6), T.([0, 0.2, 0.7, 1.5, 2.5, 4, 6]))
        ts = scale .* x
        ws = reverse!(KadanoffBaym.integration_weights(reverse(ts), ts[end-1], ts[end]))
        for p in 0:6
          exact = (Float64(x[end])^(p+1) - Float64(x[end-1])^(p+1)) / (p+1)
          @test sum(ws .* x.^p) / scale ≈ exact rtol=200eps(T)
        end
      end
    end
  end

  @testset "Constant and linear rules" begin
    ts = [0.0, 0.2, 0.7, 1.5]
    ws = KadanoffBaym.initialize_weights(ts)
    for i in eachindex(ts)
      @test sum(ws[i]) ≈ ts[i] - ts[1]
      @test sum(ws[i] .* ts[1:i]) ≈ (ts[i]^2 - ts[1]^2) / 2
    end
    @test KadanoffBaym.update_weights([0.0], [2.0, 5.0], 0) == [0.0, 3.0]
    @test KadanoffBaym.update_weights([0.0], [2.0, 5.0], 1) == [1.5, 1.5]
    @test KadanoffBaym.update_weights(ws[end], [ts; 2.0], 3)[1] == ws[end][1]
  end

  @testset "Changes of time units and origin" begin
    x = Float64.(0:6)
    w = reverse!(KadanoffBaym.integration_weights(reverse(x), x[end-1], x[end]))
    for scale in (1e-50, 1e50)
      @test reverse!(KadanoffBaym.integration_weights(scale .* reverse(x), scale*x[end-1], scale*x[end])) ./ scale ≈ w
    end
    @test reverse!(KadanoffBaym.integration_weights(1e12 .+ reverse(x), 1e12+x[end-1], 1e12+x[end])) ≈ w
  end

  @testset "Short final panels" begin
    for (T, h) in ((Float32, 1f-12), (Float64, 1e-70))
      ts = T[-5, -4, -3, -2, -1, 0, h]
      ws = reverse!(KadanoffBaym.integration_weights(reverse(ts), ts[end-1], ts[end]))
      x = Rational{BigInt}.(ts)
      reference = [t^(j - 1) for j in 1:7, t in x] \ [x[end]^j / j for j in 1:7]
      @test all(isfinite, ws)
      @test all(isapprox.(ws, T.(reference); rtol=32eps(T), atol=zero(T)))
      @test sum(ws) ≈ h
      @test sum(ws .* ts) ≈ h^2 / 2
    end
  end

  @testset "Clustered initial-history nodes stay distinct" begin
    for (T, h) in ((Float32, 1f-12), (Float64, 1e-70))
      ts = T[-5, -4, -3, -2, -1, 0, h]
      ws = KadanoffBaym.initialize_weights(ts, 6)[end]
      x = Rational{BigInt}.(ts)
      reference = [t^(j - 1) for j in 1:7, t in x] \ [(x[end]^j - x[1]^j) / j for j in 1:7]
      @test all(isfinite, ws)
      @test all(isapprox.(ws, T.(reference); rtol=2048eps(T), atol=zero(T)))
    end
  end

  @testset "Initial history integrates polynomial moments" begin
    ts = [-0.7, -0.42, -0.15, 0.1, 0.36, 0.55, 0.92, 1.4]
    for k in 1:6
      ws = KadanoffBaym.initialize_weights(ts, k)
      for i in eachindex(ts), p in 0:min(k, i - 1)
        exact = (ts[i]^(p + 1) - ts[1]^(p + 1)) / (p + 1)
        @test sum(ws[i] .* ts[1:i].^p) ≈ exact atol=2e-14 rtol=5e-12
      end
    end
  end
end

@testset "Startup quadrature does not retain a first-panel bias" begin
  G = GreenFunction(zeros(1, 1), Symmetrical)
  f!(out, ts, w1, w2, t1, t2) = (out[1] = sum(w1 .* ts[1:t1].^2); nothing)
  tol = 1e-10
  sol = kbsolve!(f!, f!, [G], (0.0, 1.0); atol=tol, rtol=tol, dtini=0.1, dtmax=0.2)
  @test abs(G[end, 1] - 1 / 12) <= 5tol
  for i in eachindex(sol.t), p in 0:min(2, i - 1)
    @test sum(sol.w[i] .* sol.t[1:i].^p) ≈ sol.t[i]^(p+1) / (p+1) atol=1e-15 rtol=1e-12
  end
end

@testset "Lowest Adams order retains trapezoidal quadrature" begin
  G = GreenFunction(zeros(1, 1), Symmetrical)
  v = GreenFunction([0.0], OnePoint)
  f0!(out, args...) = (out[1] = 0; nothing)
  f1!(out, ts, w, t) = (out[1] = sum(w .* ts[1:t]); nothing)
  tol = 1e-6
  sol = kbsolve!(f0!, f0!, [G], (0.0, 1.0); v0=[v], f1!, kmax=1,
                 atol=tol, rtol=tol, dtini=0.01, dtmax=0.1)
  @test abs(v[end] - 1 / 6) < 2tol
  @test sum(sol.w[end] .* sol.t) ≈ 1 / 2
end

@testset "Causal quadratic panel exactness and stability" begin
  T = Rational{BigInt}
  for a in T.([1//100, 1//5, 2//3]), r in T.([1//1000, 1//10, 1, 2, 5])
    b = a * r
    x = [-a, zero(T), b]
    w = reverse!(KadanoffBaym.integration_weights(reverse(x), x[end-1], x[end]))
    for p in 0:2
      @test sum(w .* x.^p) == b^(p + 1) / (p + 1)
    end
    @test sum(abs, w) == b * (1 + r^2 / (3(1 + r)))
    @test sum(abs, w) <= 43b / 18
    @test b^4 / 4 - sum(w .* x.^3) == -b^3 * (b + 2a) / 12
  end
end

@testset "Piecewise weights preserve causal branch quadrature" begin
  t = Rational{BigInt}.([-1//2, -1//5, 0, 1//3, 4//5, 9//10, 91//100, 911//1000, 13//10])
  ws = KadanoffBaym.initialize_weights(t, 2, :piecewise)
  split = Vector{Vector{eltype(t)}}()
  for i in 3:length(t)
    KadanoffBaym.piecewise_weights!(split, t, ws[i])
    for j in 2:(i - 1)
      w = split[j]
      direct = [KadanoffBaym.initialize_weights(t[1:j], 2, :piecewise)[end]; zeros(eltype(t), i - j)]
      direct[j:i] .+= KadanoffBaym.initialize_weights(t[j:i], 2, :piecewise)[end]
      @test collect(w) == direct
      @test sum(w .* abs.(t[1:i] .- t[j])) == ((t[j] - t[1])^2 + (t[i] - t[j])^2) / 2
    end
  end

  for n in (8, 16, 32, 64)
    t = Rational{BigInt}.(0:n) ./ n
    j = n ÷ 2 + 1
    ws = KadanoffBaym.initialize_weights(t, 2, :piecewise)
    w = KadanoffBaym.piecewise_weights!(Vector{Vector{eltype(t)}}(), t, ws[end])[j]
    f = [1 + abs(s - t[j]) + (s <= t[j] ? 1 : 2) * (s - t[j])^2 for s in t]
    @test sum(w .* f) - 11//8 == 1//(2n^3)
  end

  for T in (Float32, Float64)
    t = T[0, 0.1, 0.3, 0.7, 1.0]
    ws = KadanoffBaym.initialize_weights(t, 2, :piecewise)
    split = KadanoffBaym.piecewise_weights!(Vector{Vector{T}}(), t, ws[5])
    w = split[3]
    dense = collect(w)
    @test w[2:4] == dense[2:4]
    @test w .* t == dense .* t
    @test dot(w, t) ≈ dot(dense, t)
    @test_throws BoundsError w[0]
    @test_throws BoundsError w[length(w) + 1]
    @test split[1] == split[5] == ws[5]
    @test split[1] !== ws[5]
  end
end

@testset "Causal panel selection on arbitrary grids" begin
  T = Rational{BigInt}
  for t in (T.([0, 1//10, 11//100, 12//100, 1, 11//10, 10]),
            T.([0, 1//10000, 1//10, 2//10, 20001//100000, 3//10, 1]))
    for n in 2:length(t)
      l, q, δ = KadanoffBaym.piecewise_panel(t[1:n])
      h = t[n] - t[n - 1]
      x = [t[l], t[n - 1], t[n]]
      @test sum(q) == h
      @test sum(q .* x) == (t[n]^2 - t[n - 1]^2) / 2
      @test sum(abs, q) <= 7h / 6
      @test sum(δ) == 0
      if !iszero(δ[1])
        @test sum(q .* x.^2) == (t[n]^3 - t[n - 1]^3) / 3
      end
    end
  end

  for T in (Float32, Float64)
    t = T[0, 0.01, 0.02, 0.1, nextfloat(T(0.1)), 0.11, 0.12]
    w = KadanoffBaym.initialize_weights(t, 2, :piecewise)[end]
    @test sum(abs, w) <= 7t[end] / 6
    @test sum(w) ≈ t[end] rtol=4eps(T)
    @test sum(w .* t) ≈ t[end]^2 / 2 rtol=4eps(T)
  end

  errors = Float64[]
  for n in (16, 32, 64)
    t = sort!([collect(range(0, 1; length=n + 1)); nextfloat(0.5); nextfloat(nextfloat(0.5))])
    w = KadanoffBaym.initialize_weights(t, 2, :piecewise)[end]
    push!(errors, abs(sum(w .* exp.(t)) - expm1(1)))
  end
  @test all(7 .< errors[1:2] ./ errors[2:3] .< 9)
end

@testset "Piecewise weights depend only on the current grid" begin
  t = Rational{BigInt}.([0, 1//10, 11//100, 12//100])
  split = Vector{Vector{eltype(t)}}()
  for endpoint in (1, 1//5, 1//2, 1//5)
    trial = [t; endpoint]
    base = KadanoffBaym.initialize_weights(trial, 2, :piecewise)[end]
    saved = copy(base)
    KadanoffBaym.piecewise_weights!(split, trial, base)
    @test base == saved
    @test split == KadanoffBaym.piecewise_weights!(Vector{Vector{eltype(t)}}(), trial, base)
    for j in 2:length(t)
      w = split[j]
      direct = [KadanoffBaym.initialize_weights(trial[1:j], 2, :piecewise)[end]; zeros(eltype(t), length(trial) - j)]
      direct[j:end] .+= KadanoffBaym.initialize_weights(trial[j:end], 2, :piecewise)[end]
      @test collect(w) == direct
    end
  end
  base = KadanoffBaym.initialize_weights(t[1:3], 2, :piecewise)[end]
  KadanoffBaym.piecewise_weights!(split, t, base)
  @test length(split) == 3
  @test all(w -> length(w) == 3, split)
  @test split[1] == split[end] == base
end

@testset "Panel quadrature moments and interval laws" begin
  T = Rational{BigInt}
  blocks = T.([-1, -1//3, 0, 1//20, 2])
  t = sort!([blocks; (blocks[1:end-1] + blocks[2:end]) / 2])
  for (panels, degree) in (([i:i+1 for i in 1:length(t)-1], 1),
                           ([i:i+2 for i in 1:2:length(t)-2], 2))
    Q = PanelQuadrature(t, panels)
    for a in eachindex(t), b in eachindex(t)
      for p in 0:degree
        value = Q(k -> t[k]^p, a, b; sample_domain=eachindex(t), init=zero(T))
        @test value == (t[b]^(p+1) - t[a]^(p+1)) / (p+1)
      end
      K = k -> t[k]^3 + 2t[k] + 1
      Iab = Q(K, a, b; sample_domain=eachindex(t), init=zero(T))
      @test Iab == -Q(K, b, a; sample_domain=eachindex(t), init=zero(T))
      for c in eachindex(t)
        Ibc = Q(K, b, c; sample_domain=eachindex(t), init=zero(T))
        Iac = Q(K, a, c; sample_domain=eachindex(t), init=zero(T))
        @test Iab + Ibc == Iac
      end
    end
    if degree == 2
      for a in 1:2:length(t), b in 1:2:length(t)
        @test Q(k -> t[k]^3, a, b; sample_domain=eachindex(t), init=zero(T)) ==
              (t[b]^4 - t[a]^4) / 4
      end
    end
  end

  # Three unequally spaced samples still integrate their quadratic interpolant.
  t = T[0, 1//5, 1]
  Q = PanelQuadrature(t, [1:3])
  for a in eachindex(t), b in eachindex(t), p in 0:2
    @test Q(k -> t[k]^p, a, b; sample_domain=1:3, init=zero(T)) ==
          (t[b]^(p+1) - t[a]^(p+1)) / (p+1)
  end

  t = T[-2, -1, 0, 1//10, 1, 2, 4]
  Q = PanelQuadrature(t, (1:2, 2:5, 5:7))
  for (a,b) in ((1,7), (2,5), (3,4), (5,6), (6,7), (4,6), (7,2), (5,5)), p in 0:1
    @test Q(k -> t[k]^p, a, b; sample_domain=eachindex(t), init=zero(T)) ==
          (t[b]^(p+1) - t[a]^(p+1)) / (p+1)
  end
end

@testset "Matrix integrals and zero intervals" begin
  t = Rational{BigInt}.([0, 1//2, 1])
  A, B = [1 2; 3 4], [0 1; -1 2]
  initial = Rational{BigInt}.([2 -1; 4 3])
  for panels in ([1:2, 2:3], [1:3])
    Q = PanelQuadrature(t, panels)
    for (a, b) in ((1,3), (3,1), (1,2))
      result = Q(k -> A + t[k]*B, a, b; sample_domain=1:3, init=initial)
      @test result == initial + (t[b]-t[a])*A + (t[b]^2-t[a]^2)*B/2
      @test initial == [2 -1; 4 3]
    end
    @test Q(k -> error("A zero interval must not sample its kernel"), 2, 2; init=initial) == initial
    @test initial == [2 -1; 4 3]
  end

  Q = PanelQuadrature([0.0,1.0], [1:2])
  nested = [ones(2,2)]
  @test Q(k -> [fill(2.0,2,2)], 1, 2; init=nested) == [fill(3.0,2,2)]
  @test nested == [ones(2,2)]
  empty = Q(k -> error("No sample is needed"), 1, 1; init=nested)
  empty[1][1,1] = 7
  @test nested == [ones(2,2)]
end

@testset "Branch support and endpoint traces" begin
  t = Rational{BigInt}.([0, 1//4, 1//2, 3//4, 1])
  for (panels, cuts) in (([i:i+1 for i in 1:4], 2:4), ([1:3,3:5], 3:3))
    Q = PanelQuadrature(t, panels)
    for j in cuts
      # The two closures supply different traces at the same boundary node.
      left = Q(k -> 3//2, 1, j; init=0//1)
      right = Q(k -> -5//4, j, length(t); init=0//1)
      @test left + right == 3(t[j]-t[1])/2 - 5(t[end]-t[j])/4
      @test Q(k -> 3//2, j, 1; init=0//1) == -left
    end
  end

  Q = PanelQuadrature(t, [1:3,3:5])
  for (a,b) in ((1,2), (2,3), (2,1), (3,2))
    seen = Int[]
    K = k -> (push!(seen,k); one(eltype(t)))
    @test_throws ArgumentError Q(K, a, b; init=0//1)
    @test all(k -> k in min(a,b):max(a,b), seen)
  end
  for (a,b,outside) in ((1,2,3), (2,3,1))
    seen = Int[]
    K = k -> (push!(seen,k); one(eltype(t)))
    @test Q(K, a, b; sample_domain=1:3, init=0//1) == t[b]-t[a]
    @test outside in seen
  end
end

@testset "Only current history nodes are available" begin
  t = [0.0, 0.5, 1.0, 1.5, 2.0]
  Q = PanelQuadrature(t, [1:3,3:5])
  midpoint = PanelQuadrature(view(t,1:2), Q.panels, Q.weights, Q.prefix)
  seen = Int[]
  K = k -> (push!(seen,k); 1.0)
  @test_throws ArgumentError midpoint(K, 1, 2; sample_domain=1:3, init=0.0)
  @test isempty(seen)
  @test_throws BoundsError midpoint(K, 1, 3; sample_domain=1:3, init=0.0)
  @test_throws BoundsError midpoint(K, 0, 1; init=0.0)
  @test isempty(seen)

  endpoint = PanelQuadrature(view(t,1:3), Q.panels, Q.weights, Q.prefix)
  @test endpoint(K, 1, 2; sample_domain=1:3, init=0.0) ≈ 0.5
  @test 3 in seen
  @test endpoint(k -> 1.0, 1, 3; init=0.0) ≈ 1.0

  initial = PanelQuadrature(view(t,1:1), Q.panels, Q.weights, Q.prefix)
  @test initial(k -> error("No sample is needed"), 1, 1; init=2.0) == 2.0
end

@testset "Small intervals do not subtract cumulative weights" begin
  h = 1e-20
  t = [-1.0, 0.0, h]
  Q = PanelQuadrature(t, [1:2,2:3])
  @test Q(k -> 1.0, 2, 3; init=0.0) ≈ h rtol=4eps() atol=0

  t = [-1.0, -0.5, 0.0, h/2, h]
  Q = PanelQuadrature(t, [1:3,3:5])
  for (a,b) in ((3,4), (4,5), (3,5))
    @test Q(k -> 1.0, a, b; sample_domain=3:5, init=0.0) ≈ t[b]-t[a] rtol=4eps() atol=0
  end
end

end # module
