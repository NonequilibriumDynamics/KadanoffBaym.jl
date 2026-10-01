# Independent Newton divided differences, normalized at their endpoint.
function endpoint_differences(f, times, k)
  nodes = BigFloat.(reverse(times))
  d = f.(nodes)
  scale = one(first(nodes))
  ϕ = BigFloat[]
  for r in 1:min(k, length(nodes))
    push!(ϕ, scale * d[1])
    if r < min(k, length(nodes))
      d = [(d[i] - d[i + 1]) / (nodes[i] - nodes[i + r]) for i in 1:(length(d) - 1)]
      scale *= nodes[1] - nodes[r + 1]
    end
  end
  return ϕ
end

@testset "First Adams step" begin
  for h in (0.01, 0.1, 0.7), λ in (-1.0, 2.0)
    u₀ = 0.5
    cache = KadanoffBaym.VCABMCache{Float64}(3, [[u₀]])
    cache.ϕ_n[1][1] .= λ * u₀

    @test only(only(KadanoffBaym.predict!(cache, [0.0, h]))) ≈ u₀ * (1 + h * λ)
    @test only(only(KadanoffBaym.correct!(cache, out -> KadanoffBaym._recursivecopy!(out, λ .* cache.u_next)))) ≈ u₀ * (1 + h * λ + (h * λ)^2 / 2)
  end
end

@testset "Accepted Adams derivatives do not retain RHS workspaces" begin
  cache = KadanoffBaym.VCABMCache{Float64}(3, [[fill(Float64(i), 2, 2)] for i in 1:2])
  work = [zeros(2, 2) for _ in 1:2]
  rhs! = function (out)
    for i in 1:2
      work[i] .= -cache.u_next[i][1]
      KadanoffBaym.FieldSlice(out, 1)[i] = work[i]
    end
    return out
  end
  times = [0.0, 0.125]
  KadanoffBaym._recursivecopy!(cache.ϕ_n[1], -1 .* cache.u_prev)
  KadanoffBaym.predict!(cache, times)
  KadanoffBaym.correct!(cache, rhs!)
  derivative = -1 .* cache.u_next
  KadanoffBaym.adjust!(cache, times, rhs!, 3, 1.0, 0.0)
  for i in 1:2
    @test cache.steps == 1
    @test cache.ϕ_n[1][i] == derivative[i]
    work[i] .= 123
    @test cache.ϕ_n[1][i] == derivative[i]
  end
end

@testset "Adams coefficients preserve time precision" begin
  setprecision(BigFloat, 256) do
    cache = KadanoffBaym.VCABMCache{BigFloat}(4, [BigFloat[0]])
    KadanoffBaym.g_coeffs!(cache, BigFloat.(0:4), 5)
    @test cache.g ≈ BigFloat.([1, 1//2, 5//12, 3//8, 251//720]) rtol=10eps(BigFloat)
  end
end

@testset "Adams coefficients through the error estimate" begin
  h = 0.125
  for k in 1:6
    cache = KadanoffBaym.VCABMCache{Float64}(6, [[0.0]])
    cache.k = k
    cache.steps = k - 1
    KadanoffBaym.predict!(cache, collect(0:k) .* h)
    @test cache.g[1:(k + 1)] ≈ h .* [1, 1//2, 5//12, 3//8, 251//720, 95//288, 19087//60480][1:(k + 1)]
  end
end

@testset "Adams polynomial exactness on a nonuniform grid" begin
  times = [0.0, 0.031, 0.095, 0.17, 0.19, 0.28, 0.45, 0.53, 0.68, 0.74, 0.90, 1.05, 1.13, 1.29]
  for k in 2:8, m in 0:(k - 1)
    f(t) = t^m
    F(t) = t^(m + 1) / (m + 1)
    cache = KadanoffBaym.VCABMCache{Float64}(8, [[F(times[end - 1])]])
    cache.k = k
    cache.steps = length(times) - 2

    for (ϕ, value) in zip(cache.ϕ_n, endpoint_differences(f, times[1:end-1], length(cache.ϕ_n)))
      ϕ[1][1] = value
    end

    prediction = only(only(KadanoffBaym.predict!(cache, times)))
    @test prediction ≈ F(times[end]) atol=2e-13 rtol=2e-13
    @test only(only(KadanoffBaym.correct!(cache, out -> KadanoffBaym._recursivecopy!(out, [[f(times[end])]])))) ≈ F(times[end]) atol=2e-13 rtol=2e-13
  end
end

@testset "Adams local error estimate" begin
  for k in 2:8
    errors = BigFloat[]
    ratios = BigFloat[]
    for h in (big"0.001", big"0.0001")
      times = collect((-k - 1):1) .* h
      cache = KadanoffBaym.VCABMCache{BigFloat}(k, [[exp(times[end - 1])]])
      cache.k = k
      cache.steps = length(times) - 2
      for (ϕ, value) in zip(cache.ϕ_n, endpoint_differences(exp, times[1:end-1], length(cache.ϕ_n)))
        ϕ[1][1] = value
      end
      KadanoffBaym.predict!(cache, times)
      correction = only(only(KadanoffBaym.correct!(cache, out -> KadanoffBaym._recursivecopy!(out, cache.u_next))))
      KadanoffBaym.adjust!(cache, times, out -> KadanoffBaym._recursivecopy!(out, cache.u_next), k, big"1", big"0")
      push!(errors, abs(correction - exp(times[end])))
      push!(ratios, errors[end] / cache.error_k)
    end
    @test errors[1] / errors[2] ≈ big(10)^(k + 1) rtol=0.02
    @test abs(ratios[2] - 1) < abs(ratios[1] - 1) / 5
    @test ratios[2] ≈ 1 rtol=0.002
  end
end

@testset "Embedded Adams defects include the endpoint residual" begin
  for k in 3:7, power in (1, 2), nonuniform in (false, true)
    u(t) = power == 1 ? exp(t) : inv(1 - t)
    f(y) = y^power
    x = BigFloat.((-k - 1):1)
    times = big"0.1" .* (nonuniform ? x .+ big"0.1" .* sin.(x) : x)
    h = times[end] - times[end - 1]
    cache = KadanoffBaym.VCABMCache{BigFloat}(k, [[u(times[end - 1])]])
    cache.k = k
    cache.steps = length(times) - 2
    for (ϕ, value) in zip(cache.ϕ_n, endpoint_differences(t -> f(u(t)), times[1:end-1], length(cache.ϕ_n)))
      ϕ[1][1] = value
    end
    prediction = only(only(KadanoffBaym.predict!(cache, times)))
    correction = only(only(KadanoffBaym.correct!(cache, out -> KadanoffBaym._recursivecopy!(out, [[f(prediction)]]))))
    f_p, f_c = f(prediction), f(correction)
    cache.f_pred[1] .= f_p
    KadanoffBaym._recursivecopy!(cache.ϕ_np1[1], [[f_c]])
    KadanoffBaym.ϕ_np1!(cache, k + 1)

    A = function (j, endpoint_rhs)
      nodes = times[(end - j + 1):end]
      ξ = (nodes .- times[end - 1]) ./ h
      weights = [ξi^p for p in 0:(j - 1), ξi in ξ] \ [h / (p + 1) for p in 0:(j - 1)]
      values = [f.(u.(nodes[1:(end - 1)])); endpoint_rhs]
      return u(times[end - 1]) + sum(weights .* values)
    end
    @test A(k, f_p) ≈ correction rtol=big"1e-60"
    for j in (k - 2):k
      estimate = KadanoffBaym.adams_error!(cache, j, big"1", big"0")
      @test estimate ≈ abs(A(j + 1, f_c) - A(j, f_p)) atol=big"1e-65" rtol=big"1e-50"
    end
  end
end

@testset "Higher-order error coefficients use the current grid" begin
  T = Rational{BigInt}
  cache = KadanoffBaym.VCABMCache{T}(9, [T[0]])
  KadanoffBaym.g_coeffs!(cache, T.(0:12), 10)
  @test cache.g[5:10] - cache.g[4:9] == [-19//720, -3//160, -863//60480, -275//24192, -33953//3628800, -8183//1036800]

  times = BigFloat.([0.0, 0.031, 0.095, 0.17, 0.19, 0.28, 0.45, 0.53, 0.68, 0.74, 0.90, 1.05, 1.13, 1.29])
  for k in 3:8
    f(t) = t^(k + 1)
    F(t) = t^(k + 2) / (k + 2)
    cache = KadanoffBaym.VCABMCache{BigFloat}(k + 1, [[F(times[end - 1])]])
    cache.k = k
    cache.steps = length(times) - 2
    for (ϕ, value) in zip(cache.ϕ_n, endpoint_differences(f, times[1:end-1], length(cache.ϕ_n)))
      ϕ[1][1] = value
    end
    candidate = deepcopy(cache)
    candidate.k = k + 1

    KadanoffBaym.predict!(cache, times)
    KadanoffBaym._recursivecopy!(cache.ϕ_np1[1], [[f(times[end])]])
    KadanoffBaym.ϕ_np1!(cache, k + 2)
    estimate = abs((cache.g[k + 2] - cache.g[k + 1]) * only(only(cache.ϕ_np1[k + 2])))
    KadanoffBaym.predict!(candidate, times)
    correction = only(only(KadanoffBaym.correct!(candidate, out -> KadanoffBaym._recursivecopy!(out, [[f(times[end])]]))))
    @test estimate ≈ abs(F(times[end]) - correction) rtol=big"1e-50"
  end
end

@testset "Order cap preserves the quadrature error" begin
  times = collect(0.0:0.1:0.6)
  f(t) = t^3
  F(t) = t^4 / 4
  cache = KadanoffBaym.VCABMCache{Float64}(3, [[F(times[end - 1])]])
  cache.k = 3
  cache.steps = length(times) - 2
  for (ϕ, value) in zip(cache.ϕ_n, endpoint_differences(f, times[1:end-1], length(cache.ϕ_n)))
    ϕ[1][1] = value
  end

  KadanoffBaym.predict!(cache, times)
  correction = only(only(KadanoffBaym.correct!(cache, out -> KadanoffBaym._recursivecopy!(out, [[f(times[end])]]))))
  KadanoffBaym.adjust!(cache, times, out -> KadanoffBaym._recursivecopy!(out, [[f(times[end])]]), 3, 1.0, 0.0)
  @test cache.k == 3
  @test cache.error_k ≈ abs(F(times[end]) - correction) atol=1e-16 rtol=1e-10
end

@testset "A rejected Adams step preserves its accepted history" begin
  times = collect(0.0:0.1:0.6)
  cache = KadanoffBaym.VCABMCache{Float64}(4, [[exp(times[end - 1])]])
  cache.k = 4
  cache.steps = length(times) - 2
  for (ϕ, value) in zip(cache.ϕ_n, endpoint_differences(exp, times[1:end-1], length(cache.ϕ_n)))
    ϕ[1][1] = value
  end
  accepted = deepcopy(cache)

  KadanoffBaym.predict!(cache, times)
  KadanoffBaym.correct!(cache, out -> KadanoffBaym._recursivecopy!(out, cache.u_next))
  KadanoffBaym.adjust!(cache, times, out -> KadanoffBaym._recursivecopy!(out, cache.u_next), 4, 1e-14, 0.0)
  @test cache.error_k > 1
  @test cache.k == accepted.k
  @test cache.steps == accepted.steps
  @test cache.u_prev == accepted.u_prev
  @test cache.ϕ_n[1] == accepted.ϕ_n[1]
  @test cache.ϕ_n == accepted.ϕ_n

  times[end] = times[end - 1] + 0.01
  @test KadanoffBaym.predict!(cache, times) == KadanoffBaym.predict!(accepted, times)
  @test KadanoffBaym.correct!(cache, out -> KadanoffBaym._recursivecopy!(out, cache.u_next)) == KadanoffBaym.correct!(accepted, out -> KadanoffBaym._recursivecopy!(out, accepted.u_next))
end

@testset "Every Adams field controls acceptance" begin
  times = collect(0.0:0.1:0.6)
  for active in 1:2
    cache = KadanoffBaym.VCABMCache{Float64}(4, [[i == active ? exp(times[end - 1]) : 0.0] for i in 1:2])
    cache.k = 4
    cache.steps = length(times) - 2
    for i in 1:2
      f(t) = i == active ? exp(t) : zero(t)
      for (ϕ, value) in zip(cache.ϕ_n, endpoint_differences(f, times[1:end-1], length(cache.ϕ_n)))
        ϕ[i][1] = value
      end
    end
    KadanoffBaym.predict!(cache, times)
    rhs!(out) = KadanoffBaym._recursivecopy!(out, cache.u_next)
    KadanoffBaym.correct!(cache, rhs!)
    KadanoffBaym.adjust!(cache, times, rhs!, 4, 1e-14, 0.0)
    @test cache.error_k > 1
    @test [only(field) for field in cache.u_prev] == [i == active ? exp(times[end - 1]) : 0.0 for i in 1:2]
  end
end

@testset "Triangular Adams history reproduces mixed polynomials" begin
  T = Rational{BigInt}
  divided_difference(x, f) = sum(f[j] / prod(x[j] - x[m] for m in eachindex(x) if m != j; init=one(T)) for j in eachindex(x))
  for k in (2, 3, 6), startup in (false, true)
    n = startup ? k : k + 2
    times = T[i//10 + i^2//1000 for i in 0:n]
    degree = min(k + 1, n)
    for p in 0:degree, q in 0:(degree - p)
      f(x, y) = x^p * y^q
      integral(y) = y^q * (times[end]^(p + 1) - times[n]^(p + 1)) / (p + 1)
      cache = KadanoffBaym.VCABMCache{T}(k + 1, [zeros(T, n + 1)])
      cache.k = k
      cache.steps = n - 1
      for col in 1:n
        cache.ϕ_n[1][1][col] = f(times[n], times[col])
        for r in 1:min(k, n - 1)
          if col > n - r
            cache.ϕ_n[r + 1][1][col] = -10000 - r - col
          else
            x = times[n:-1:(n - r)]
            cache.ϕ_n[r + 1][1][col] = divided_difference(x, f.(x, times[col])) * prod(times[n] - times[n - m] for m in 1:r; init=one(T))
          end
        end
      end

      KadanoffBaym.predict!(cache, times; n2=length(cache.u_prev))
      if p + q <= k - 1
        for i in 0:(k - 1), col in 1:n
          x = times[n:-1:(n - i)]
          expected = divided_difference(x, f.(x, times[col])) * prod(times[end] - times[n - m] for m in 0:(i - 1); init=one(T))
          @test cache.ϕstar_n[i + 1][1][col] == expected
        end
      end
      f_np1 = [zeros(T, n + 1)]
      f_np1[1][1:n] .= f.(times[end], times[1:n])
      stars = [copy(ϕ[1]) for ϕ in cache.ϕstar_n]
      KadanoffBaym.correct!(cache, out -> KadanoffBaym._recursivecopy!(out, f_np1))
      if p + q <= k - 1
        @test cache.u_next[1][1:n] == integral.(times[1:n])
      end
      KadanoffBaym._recursivecopy!(cache.f_pred, cache.ϕ_np1[1])
      KadanoffBaym.ϕ_np1!(cache, min(k + 2, cache.steps + 2))
      KadanoffBaym.complete_triangle!(cache.ϕ_np1, times, k + 1, cache.steps + 1, cache.u_erro, 1:length(cache.u_prev))
      corrected = deepcopy(cache.ϕ_np1[1:(degree + 1)])
      for i in 0:degree, col in 1:n
        x = times[end:-1:(end - i)]
        expected = divided_difference(x, f.(x, times[col])) * prod(times[end] - times[end - m] for m in 1:i; init=one(T))
        @test cache.ϕ_np1[i + 1][1][col] == expected
      end
      if p <= k
        for col in 1:n
          higher = cache.g[1] * cache.ϕ_np1[1][1][col] + sum((cache.g[r] - cache.g[r - 1]) * cache.ϕ_np1[r][1][col] for r in 2:(k + 1))
          @test higher == integral(times[col])
        end
      end
      cache.f_pred[1][1:n] .+= 1//7
      A_c(m, col) = cache.g[1] * cache.ϕ_np1[1][1][col] + sum((cache.g[r] - cache.g[r - 1]) * cache.ϕ_np1[r][1][col] for r in 2:m)
      A_p(m, col) = cache.g[m] * cache.f_pred[1][col] + sum((cache.g[r] - cache.g[m]) * stars[r][col] for r in 1:m)
      for j in 1:k
        expected = [A_c(j + 1, col) - A_p(j, col) for col in 1:n]
        KadanoffBaym.adams_error!(cache, j, one(T), zero(T); n2=length(cache.u_prev))
        @test Float64.(cache.u_erro[1][1:n]) == Float64.(expected)
      end
      KadanoffBaym.adjust!(cache, times, out -> KadanoffBaym._recursivecopy!(out, f_np1), k + 1, one(T), zero(T); n2=length(cache.u_prev))
      @test cache.steps == n
      @test cache.ϕ_n[1:(degree + 1)] == corrected
    end
  end
end

@testset "Triangular extrapolation preserves constant columns" begin
  times = cumsum(5.0 .^ (0:9))
  n, k = length(times) - 1, 6
  cache = KadanoffBaym.VCABMCache{Float64}(k + 1, [zeros(n + 1)])
  cache.k = k
  cache.steps = n - 1
  cache.ϕ_n[1][1] .= 2
  for r in 1:k
    cache.ϕ_n[r + 1][1] .= cache.ϕ_n[r][1] .- inv(r + 1)
  end
  KadanoffBaym.predict!(cache, times; n2=length(cache.u_prev))
  @test all(ϕ -> all(==(ϕ[1][1]), ϕ[1][1:n]), cache.ϕstar_n[1:k])
  f!(out) = (out[1] .= 3; out)
  KadanoffBaym.correct!(cache, f!)
  KadanoffBaym.adjust!(cache, times, f!, k + 1, 1e100, 0.0; n2=length(cache.u_prev))
  @test all(ϕ -> all(==(ϕ[1][1]), ϕ[1][1:n]), cache.ϕ_n[1:(k + 2)])
end

@testset "Restarted Adams history begins at the first accepted step" begin
  for initial_times in ([0.0], [-3.0, -0.7, 0.0])
    h = 0.1
    times = [initial_times; h]
    n = length(initial_times)
    cache = KadanoffBaym.VCABMCache{Float64}(4, [exp.(initial_times)])
    cache.ϕ_n[1][1] .= -cache.u_prev[1]
    cache.ϕ_n[1][1][n] = 0
    fv!(out, i, j) = (out[1][j] = -cache.u_prev[1][j])
    KadanoffBaym.extend!(cache, times, fv!, length(cache.u_prev))
    @test cache.steps == 0
    KadanoffBaym.predict!(cache, times; n2=length(cache.u_prev))
    f! = function (out)
      out[1] .= -cache.u_next[1]
      out[1][end] = 0
      return out
    end
    KadanoffBaym.correct!(cache, f!)
    @test cache.u_next[1][1:n] ≈ exp.(initial_times) .* (1 - h + h^2 / 2)
    @test cache.u_next[1][end] == 1
    KadanoffBaym.adjust!(cache, times, f!, 4, 1.0, 0.0; n2=length(cache.u_prev))
    @test cache.steps == 1
    @test cache.k == 2
  end
end

@testset "An isolated column controls acceptance and order" begin
  for n in (1, 4, 100), fields in (1, 3), value in (1.0, ones(2, 2))
    h = 0.125
    u = [[copy(value) for _ in 1:n] for _ in 1:fields]
    cache = KadanoffBaym.VCABMCache{Float64}(3, u)
    rhs = function (y)
      f = zero.(y)
      f[1][end] = copy(y[1][end])
      return f
    end
    KadanoffBaym._recursivecopy!(cache.ϕ_n[1], rhs(cache.u_prev))
    KadanoffBaym.predict!(cache, [0.0, h])
    KadanoffBaym.correct!(cache, out -> KadanoffBaym._recursivecopy!(out, rhs(cache.u_next)))
    KadanoffBaym.adjust!(cache, [0.0, h], out -> KadanoffBaym._recursivecopy!(out, rhs(cache.u_next)), 3, h^2/4, 0.0)
    # Euler/trapezoid defect h²/2 divided by atol=h²/4 must reject the step.
    @test cache.error_k ≈ 2
    @test cache.steps == 0
    @test cache.k == 1
    @test all(all(isone, x) for field in cache.u_prev for x in field)
  end
end
