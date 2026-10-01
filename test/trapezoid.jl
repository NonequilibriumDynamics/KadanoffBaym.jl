@testset "Common trapezoid reference" begin
  @testset "Free matrix columns and moving diagonal" begin
    ts = [0.0, 0.03, 0.08, 0.125, 0.2, 0.3]
    H = ComplexF64[0.05 1; 1 -0.05]
    G₀ = ComplexF64[im 0; 0 0]
    G = GreenFunction(reshape(copy(G₀), 2, 2, 1, 1), SkewHermitian)
    fv!(out, ts, w1, w2, i, j) = (out[1] .= -im * H * G[i, j]; nothing)
    fd!(out, ts, w1, w2, i, j) = (out[1] .= -im * (H * G[i, i] - G[i, i] * H); nothing)
    reference_trapezoid!(fv!, fd!, [G], ts; atol=1e-13, maxiters=100)

    I₂ = Matrix{ComplexF64}(I, 2, 2)
    L = -im * (kron(I₂, H) - kron(transpose(H), I₂))
    expected = Matrix{Matrix{ComplexF64}}(undef, length(ts), length(ts))
    expected[1, 1] = G₀
    for i in 2:length(ts)
      h = ts[i] - ts[i-1]
      P = (I + im * h / 2 * H) \ (I - im * h / 2 * H)
      for j in 1:(i-1)
        expected[i, j] = P * expected[i-1, j]
      end
      expected[i, i] = reshape((I - h / 2 * L) \ ((I + h / 2 * L) * vec(expected[i-1, i-1])), 2, 2)
    end
    @test maximum(norm(G[i, j] - expected[i, j]) for i in eachindex(ts) for j in 1:i) < 2e-11
    @test maximum(norm(G[j, i] + adjoint(G[i, j])) for i in eachindex(ts) for j in 1:i) < 2e-13
    @test all(isapprox(tr(G[i, i]), tr(G₀); atol=2e-13) for i in eachindex(ts))
  end

  @testset "Noncommuting memory equals augmented trapezoid" begin
    ts = [0.0, 0.01, 0.1, nextfloat(0.1), 0.16, 0.25, 0.4]
    A = [0.05 0.1; 0.1 -0.05]
    B = -[1.0 0.2; 0.2 0.6]
    C = [0.8 -0.3; -0.3 1.2]
    G₀ = im * [1.0 0.2; 0.2 0.5]
    L = [kron(Matrix{Float64}(I, 2, 2), A) kron(transpose(C), B);
      Matrix{Float64}(I, 4, 4) zeros(4, 4)]
    expected = Matrix{Matrix{ComplexF64}}(undef, length(ts), length(ts))
    for j in eachindex(ts)
      z = [vec(G₀); zeros(ComplexF64, 4)]
      expected[j, j] = copy(G₀)
      for i in (j+1):length(ts)
        h = ts[i] - ts[i-1]
        z = (I - h / 2 * L) \ ((I + h / 2 * L) * z)
        expected[i, j] = reshape(z[1:4], 2, 2)
      end
    end

    solutions = []
    for restart in (false, true)
      G = GreenFunction(reshape(copy(G₀), 2, 2, 1, 1), SkewHermitian)
      function fv!(out, ts, w1, w2, i, j)
        memory = sum(w1[s] * G[s, j] for s in eachindex(w1)) -
          sum(w2[s] * G[s, j] for s in eachindex(w2))
        out[1] .= A * G[i, j] + B * memory * C
        return nothing
      end
      fd!(out, args...) = (out[1] .= 0; nothing)
      if restart
        reference_trapezoid!(fv!, fd!, [G], ts[1:4]; atol=1e-13, maxiters=100)
      end
      reference_trapezoid!(fv!, fd!, [G], ts; start=restart ? 4 : 1, atol=1e-13, maxiters=100)
      @test maximum(norm(G[i, j] - expected[i, j]) for i in eachindex(ts) for j in 1:i) < 2e-11
      push!(solutions, copy(G.data))
    end
    @test solutions[1] ≈ solutions[2] atol=2e-13 rtol=0
  end

  @testset "Callback couples diagonal and one-time state" begin
    ts = collect(range(0.0, 0.5; length=17))
    G = GreenFunction(ones(1, 1), Symmetrical)
    v = GreenFunction(ones(1), OnePoint)
    Σ = GreenFunction(fill(2.0, length(ts), length(ts)), Symmetrical)
    callback(ts, w1, w2, i, j) = (Σ[i, j] = G[i, j] + v[i]; nothing)
    fv!(out, ts, w1, w2, i, j) = (out[1] = Σ[i, j] - G[i, j]; nothing)
    fd!(out, ts, w1, w2, i, j) = (out[1] = Σ[i, i] - G[i, i]; nothing)
    f1!(out, ts, w, i) = (out[1] = Σ[i, i] - v[i]; nothing)
    reference_trapezoid!(fv!, fd!, [G], ts; f1!, v=[v], callback, atol=1e-13, maxiters=100)

    expected = [1.0; cumprod((1 .+ diff(ts) / 2) ./ (1 .- diff(ts) / 2))]
    @test v.data ≈ expected atol=2e-11 rtol=0
    @test G.data ≈ [expected[max(i, j)] for i in eachindex(ts), j in eachindex(ts)] atol=2e-11 rtol=0
    @test maximum(abs(Σ[i, j] - G[i, j] - v[i]) for i in eachindex(ts) for j in 1:i) < 2e-13
  end

  @testset "Nonlinear memory couples both times and the diagonal" begin
    E(t, s) = t >= s ? 1 + t + 2s + t*s + t^2 + s^2 + t^3/3 + s^3/5 : E(s, t)
    # Polynomial coefficients in the integration coordinate on each smooth branch.
    p(t) = [1 + t + t^2 + t^3/3, 2 + t, 1, 1/5]
    r(s) = [1 + 2s + s^2 + s^3/5, 1 + s, 1, 1/3]
    product_integral(a, b, x, y) = sum(a[i] * b[j] * (y^(i+j-1) - x^(i+j-1)) / (i+j-1)
                                      for i in eachindex(a), j in eachindex(b))
    Iexact(t, s) = product_integral(p(t), p(s), 0, s) + product_integral(p(t), r(s), s, t)
    Jexact(t) = product_integral(p(t), p(t), 0, t)

    errors = Float64[]
    for n in (8, 16, 32, 64)
      ts = collect(range(0.0, 1.0; length=n+1))
      G = GreenFunction(ones(1, 1), Symmetrical)
      function fv!(out, ts, w1, w2, i, j)
        t, s = ts[i], ts[j]
        I = sum(w1[k] * G[i, k] * G[k, j] for k in eachindex(w1))
        J = sum(w2[k] * G[j, k]^2 for k in eachindex(w2))
        out[1] = 1 + s + 2t + t^2 + 0.15 * (I - Iexact(t, s)) - 0.1 * (J - Jexact(s))
        return nothing
      end
      function fd!(out, ts, w1, w2, i, j)
        t = ts[i]
        J = sum(w1[k] * G[i, k]^2 for k in eachindex(w1))
        out[1] = 3 + 6t + 8t^2/5 + 0.2 * (J - Jexact(t))
        return nothing
      end
      reference_trapezoid!(fv!, fd!, [G], ts; atol=1e-13, maxiters=100)
      push!(errors, maximum(abs.(G.data .- [E(t, s) for t in ts, s in ts])))
    end
    @test all(3.7 .< errors[1:end-1] ./ errors[2:end] .< 4.3)
  end

  @testset "An unconverged implicit step is rejected" begin
    G = GreenFunction(ones(1, 1), Symmetrical)
    fv!(out, ts, w1, w2, i, j) = (out[1] = -100G[i, j]; nothing)
    fd!(out, args...) = (out[1] = 0; nothing)
    @test_throws ErrorException reference_trapezoid!(fv!, fd!, [G], [0.0, 0.1]; atol=1e-13, maxiters=4)
  end
end
