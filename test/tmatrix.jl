module HubbardTMExample
using Main.KadanoffBaym, LinearAlgebra
import Main.Hubbard2BExample: integrate1, integrate2
source = read(joinpath(@__DIR__, "../docs/src/examples/FermiHubbardTM.md"), String)
for block in eachmatch(r"```julia\n(.*?)\n```"s, source)
  code = block.captures[1]
  occursin("function T_matrix!", code) && include_string(@__MODULE__, code, "FermiHubbardTM.md")
end
end

function test_tmatrix(example)
  function data_from_green_functions(GL_u, GG_u, GL_d, GG_d)
    return (; GL_u, GG_u, GL_d, GG_d,
      TL=zero(GL_u), TG=zero(GL_u), ΦL=zero(GL_u), ΦG=zero(GL_u),
      ΣL_u=zero(GL_u), ΣG_u=zero(GL_u), ΣL_d=zero(GL_u), ΣG_d=zero(GL_u))
  end
  function weights(times)
    w = [zeros(i) for i in eachindex(times)]
    for i in 2:length(times)
      w[i][1:i-1] .= w[i-1]
      w[i][i-1:i] .+= (times[i] - times[i-1]) / 2
    end
    return w
  end

  @testset "Frozen bubbles give the exact discrete pair propagator" begin
    times = [0.0, 0.05, 0.1, 0.2]
    w = weights(times)
    p, q = 0.2, 0.5
    for U in (t -> 0.7, identity, t -> 400.0)
      gf(x) = GreenFunction(fill(ComplexF64(x), 1, 1, length(times), length(times)), SkewHermitian)
      data = data_from_green_functions(gf(im*p), gf(-im*(1-p)), gf(im*q), gf(-im*(1-q)))
      for i in eachindex(times), j in 1:i
        example.T_matrix!((; U), data, times, w[i], w[j], i, j)
        propagator = prod((1 - im * (1-p-q) * (times[k]-times[k-1]) * U(times[k-1]) / 2) /
          (1 + im * (1-p-q) * (times[k]-times[k-1]) * U(times[k]) / 2) for k in (j+1):i; init=1.0+0.0im)
        @test data.TL[i, j][1] ≈ im * p*q * propagator atol=2e-13 rtol=2e-13
        @test data.TG[i, j][1] ≈ im * (1-p)*(1-q) * propagator atol=2e-13 rtol=2e-13
      end
      if iszero(U(first(times)))
        for i in eachindex(times), Σ in (data.ΣL_u, data.ΣG_u, data.ΣL_d, data.ΣG_d)
          @test iszero(Σ[i, 1])
        end
      end
    end
  end

  @testset "Noncommuting matrix endpoints satisfy the full integral equations" begin
    times = [0.0, 0.03, 0.08, 0.15]
    w = weights(times)
    H_u = ComplexF64[0.2 0.3+0.1im; 0.3-0.1im -0.1]
    H_d = ComplexF64[-0.1 0.2-0.2im; 0.2+0.2im 0.4]
    ρ_u = [0.2 0.08; 0.08 0.65]
    ρ_d = [0.55 -0.1; -0.1 0.3]
    function green(H, ρ, lesser)
      G = GreenFunction(zeros(ComplexF64, 2, 2, length(times), length(times)), SkewHermitian)
      for i in eachindex(times), j in 1:i
        G[i, j] = exp(-im * H * times[i]) * (lesser ? im * ρ : -im * (I - ρ)) * exp(im * H * times[j])
      end
      return G
    end
    for U in (t -> 0.7, t -> 0.4 + t)
      data = data_from_green_functions(green(H_u, ρ_u, true), green(H_u, ρ_u, false),
        green(H_d, ρ_d, true), green(H_d, ρ_d, false))
      for i in eachindex(times), j in 1:i
        example.T_matrix!((; U), data, times, w[i], w[j], i, j)
      end
      (; TL, TG, ΦL, ΦG) = data
      n = length(times)
      @test norm(ΦL[n, n] * ΦG[n, n] - ΦG[n, n] * ΦL[n, n]) > 1e-4
      for i in eachindex(times), j in 1:i, (T, Φ) in ((TL, ΦL), (TG, ΦG))
        residual = T[i, j] - Φ[i, j] +
          sum(w[i][k] * U(times[k]) * (ΦG[i, k] - ΦL[i, k]) * T[k, j] for k in 1:i) +
          sum(w[j][k] * U(times[k]) * Φ[i, k] * (TL[k, j] - TG[k, j]) for k in 1:j)
        @test norm(residual) < 2e-13
      end
    end
  end
end

test_tmatrix(HubbardTMExample)
