module Hubbard2BExample
using Main.KadanoffBaym, LinearAlgebra
source = read(joinpath(@__DIR__, "../docs/src/examples/FermiHubbard2B.md"), String)
for block in eachmatch(r"```julia\n(.*?)\n```"s, source)
  code = block.captures[1]
  if occursin(r"function (second_Born!|integrate1|integrate2|fv!|fd!)\(", code)
    include_string(@__MODULE__, code, "FermiHubbard2B.md")
  end
end
end

function test_hubbard_example(example)
  function initial_data(nup, ndown, n=1)
    gf(value) = GreenFunction(fill(ComplexF64(value), 1, 1, n, n), SkewHermitian)
    GL_u, GG_u = gf(im * nup), gf(-im * (1 - nup))
    GL_d, GG_d = gf(im * ndown), gf(-im * (1 - ndown))
    return (; GL_u, GG_u, GL_d, GG_d,
      ΣL_u=zero(GL_u), ΣG_u=zero(GG_u), ΣL_d=zero(GL_d), ΣG_d=zero(GG_d))
  end

  @testset "Hubbard atomic limit" begin
    for (U, phase) in ((t -> 0.7, t -> 0.7t),
                       (t -> 0.5 + t^2, t -> 0.5t + t^3 / 3))
      data = initial_data(0.3, 1.0)
      model = (; U, H_u=zeros(ComplexF64, 1, 1), H_d=zeros(ComplexF64, 1, 1))
      sol = kbsolve!((args...) -> example.fv!(model, data, args...),
                    (args...) -> example.fd!(model, data, args...),
                    [data.GL_u, data.GG_u, data.GL_d, data.GG_d], (0.0, 0.4);
                    callback=(args...) -> example.second_Born!(model, data, args...),
                    atol=1e-9, rtol=1e-9, dtini=0.01, dtmax=0.05)
      @test maximum(abs(data.GL_u[i, j][1] - 0.3im * cis(-(phase(t) - phase(s))))
        for (i, t) in enumerate(sol.t), (j, s) in enumerate(sol.t)) < 2e-8
      @test maximum(abs(data.GG_u[i, j][1] + 0.7im * cis(-(phase(t) - phase(s))))
        for (i, t) in enumerate(sol.t), (j, s) in enumerate(sol.t)) < 2e-8
      @test maximum(abs(data.GL_d[i, i][1] - im) for i in eachindex(sol.t)) < 1e-12
      @test iszero(data.GG_d.data)
      n = length(sol.t)
      @test iszero(view(data.ΣL_u.data, :, :, 1:n, 1:n))
      @test iszero(view(data.ΣG_u.data, :, :, 1:n, 1:n))
    end
  end

  @testset "Hartree uses the opposite spin at the differentiated time" begin
    data = initial_data(0.2, 0.8, 2)
    times = [0.0, 0.7]
    model = (U=t -> 0.5 + t^2, H_u=zeros(ComplexF64, 1, 1), H_d=zeros(ComplexF64, 1, 1))
    out = [zeros(ComplexF64, 1, 1) for _ in 1:4]
    example.fv!(model, data, out, times, zeros(2), zeros(1), 2, 1)
    for (value, G, occupation) in zip(out, (data.GL_u, data.GG_u, data.GL_d, data.GG_d), (0.8, 0.8, 0.2, 0.2))
      @test value ≈ -im * model.U(times[2]) * occupation * G[2, 1]
    end
  end

  @testset "Second Born has two time-local interaction vertices" begin
    times = [0.0, 0.7]
    p, q = 0.2, 0.8
    for U in (t -> 0.5 + t^2, identity)
      data = initial_data(p, q, 2)
      example.second_Born!((; U), data, times, nothing, nothing, 2, 1)
      UU = U(times[2]) * U(times[1])
      @test data.ΣL_u[2, 1][1] ≈ im * UU * p * q * (1 - q)
      @test data.ΣL_d[2, 1][1] ≈ im * UU * q * p * (1 - p)
      @test data.ΣG_u[2, 1][1] ≈ -im * UU * (1 - p) * (1 - q) * q
      @test data.ΣG_d[2, 1][1] ≈ -im * UU * (1 - q) * (1 - p) * p
    end
  end
end

test_hubbard_example(Hubbard2BExample)
