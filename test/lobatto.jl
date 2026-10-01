module LobattoReferenceTests
using KadanoffBaym, LinearAlgebra, Test
include(joinpath(@__DIR__, "..", "benchmarks", "quantum.jl"))

function manufactured(n; irregular=false, μ=.2)
  g = GreenFunction(ones(1,1), Symmetrical)
  v = GreenFunction([1.], OnePoint)
  function fv!(out, t, w1, w2, i, j)
    @assert i >= j
    memory = sum(w1[k]*g[i,k]*g[k,j] for k in eachindex(w1)) +
             sum(w2[k]*g[i,k]*g[k,j] for k in eachindex(w2))
    exact = exp(t[i]+t[j])*(expm1(2t[i])+expm1(2t[j]))/2
    out[1] = g[i,j] + .1*(v[i]*v[j]-g[i,j]) + μ*(memory-exact)
  end
  function fd!(out, t, w1, w2, i, j)
    fv!(out,t,w1,w2,i,j)
    out .*= 2
  end
  function f1!(out,t,w,i)
    out[1] = v[i] + .1*(g[i,i]-v[i]^2) + μ*(sum(w[k]*v[k] for k in eachindex(w))-expm1(t[i]))
  end
  blocks = collect(range(0.,1.;length=n+1))
  irregular && (blocks .^= 1.5)
  s = reference_lobatto!(fv!,fd!,[g],blocks; f1!, v=[v],atol=1e-14)
  return (full=maximum(abs(g[i,j]-exp(s.t[i]+s.t[j])) for i in eachindex(s.t) for j in 1:i),
          one=maximum(abs(v[i]-exp(s.t[i])) for i in eachindex(s.t)))
end

@testset "Joint block laws" begin
  g = GreenFunction(ones(1,1), Symmetrical)
  fv!(o,t,w1,w2,i,j) = (o[1] = sum(w2[k]*t[k]^2 for k in eachindex(w2))-t[j]^3/3)
  fd!(o,args...) = (o[1] = 0.)
  sol = reference_lobatto!(fv!,fd!,[g],[0.,.1,.6,.65,1.])
  @test maximum(abs,g.data.-1) <= 5eps()
  for j in eachindex(sol.t), q in 0:2
    @test sum(sol.w[j][k]*sol.t[k]^q for k in eachindex(sol.w[j])) ≈ sol.t[j]^(q+1)/(q+1) atol=8eps()
  end
  p = GreenFunction(ones(1,1),Symmetrical)
  fp!(o,t,w1,w2,i,j) = (o[1]=3(t[i]+t[j])^2)
  dp!(o,t,w1,w2,i,j) = (o[1]=6(2t[i])^2)
  s = reference_lobatto!(fp!,dp!,[p],[0.,.03,.3,.35,1.];atol=1e-14)
  @test maximum(abs(p[i,j]-1-(s.t[i]+s.t[j])^3) for i in eachindex(s.t) for j in 1:i) < 1e-13
  @test_throws ErrorException reference_lobatto!(fp!,dp!,[GreenFunction(ones(1,1),Symmetrical)],[0.,1.];maxiters=1)
  for irregular in (false,true)
    errors = [manufactured(n;irregular) for n in (8,16,32)]
    for field in (:full,:one)
      ratios=[getproperty(errors[i],field)/getproperty(errors[i+1],field) for i in 1:2]
      @test all(>(13),ratios)
    end
  end
end

@testset "A cusp violates smooth block quadrature despite diagonal compatibility" begin
  for n in (2, 4, 8, 16)
    G = GreenFunction(fill(1.0im, 1, 1), SkewHermitian)
    function fv!(out, t, w1, w2, i, j)
      s = t[j]
      out[1] = im * (sum(w1[k] * abs(t[k] - s) for k in eachindex(w1)) - (s^2 + (t[i] - s)^2) / 2)
    end
    function fd!(out, args...)
      fv!(out, args...)
      out[1] -= out[1]'
    end
    sol = reference_lobatto!(fv!, fd!, [G], range(0.0, 1.0; length=n+1))
    @test G[end, end] - im ≈ -im / (18n^2) atol=5e-15 rtol=1e-12

    G = GreenFunction(fill(1.0im, 1, 1), SkewHermitian)
    reference_trapezoid!(fv!, fd!, [G], sol.t)
    @test maximum(abs, G.data .- im) < 1e-13
  end
end

@testset "Shared kernel equations with separate branch traces" begin
  for rule in (reference_trapezoid!, reference_lobatto!), jump in (false, true)
    G = GreenFunction(fill(1.0im, 1, 1), SkewHermitian)
    function fv!(out, ts, integral, i, j)
      s, t = ts[j], ts[i]
      # Each branch has a smooth polynomial continuation across its limit.
      left = k -> jump ? 2.0 : s-ts[k]
      right = k -> jump ? -1.0 : ts[k]-s
      memory = integral(left, 1, j; sample_domain=eachindex(ts), init=0.0)
      memory = integral(right, j, i; sample_domain=eachindex(ts), init=memory)
      exact = jump ? 2s-(t-s) : (s^2+(t-s)^2)/2
      out[1] = im*(memory-exact)
    end
    function fd!(out, args...)
      fv!(out, args...)
      out[1] -= out[1]'
    end
    equations = (; fv!, fd!, f1! = nothing, callback=(args...) -> nothing)
    rule(equations, [G], [0., .03, .3, .35, 1.]; atol=1e-14)
    @test maximum(abs, G.data .- im) < 2e-14
  end

  # Merely splitting bounds does not authorize sampling across a branch boundary.
  limited!(out, ts, integral, i, j) = (out[1] = integral(k -> ts[k], 1, j; init=0.0))
  stationary!(out, args...) = (out[1] = 0.0)
  equations = (; fv! = limited!, fd! = stationary!, f1! = nothing, callback=(args...) -> nothing)
  G = GreenFunction(ones(1,1), Symmetrical)
  @test_throws ArgumentError reference_lobatto!(equations, [G], [0., 1.])
end

@testset "Lobatto oscillator endpoints and stored stages" begin
  for η in (0.1, 1.0, 2.0, 3.0)
    G = GreenFunction(fill(1.0im, 1, 1), SkewHermitian)
    z = im * η
    fv!(out, t, w1, w2, i, j) = (out[1] = z * G[i, j])
    fd!(out, args...) = (out[1] = 0im)
    reference_lobatto!(fv!, fd!, [G], [0.0, 1.0]; atol=1e-14, maxiters=1000)
    R = (1 + z/2 + z^2/12) / (1 - z/2 + z^2/12)
    r = (1 - z^2/24) / (1 - z/2 + z^2/12)
    @test G[3, 1] / im ≈ R atol=1e-13
    @test G[2, 1] / im ≈ r atol=1e-13
    @test G[3, 2] / im ≈ r atol=1e-13
    @test abs(G[3, 1]) ≈ 1 atol=1e-13
    @test abs(G[2, 1]) < 1
  end
end

@testset "The corner has a fourth-order birth defect" begin
  G = GreenFunction(fill(1.0im, 1, 1), SkewHermitian)
  fv!(out, t, w1, w2, i, j) = (out[1] = t[i]^3)
  fd!(out, args...) = (out[1] = 0im)
  h = 0.4
  reference_lobatto!(fv!, fd!, [G], [0.0, h]; atol=1e-15)
  exact = im + (h^4 - (h/2)^4) / 4
  @test exact - G[3, 2] ≈ -h^4/64 atol=3e-16
end

@testset "Lobatto quantum convergence" begin
  @testset "The same equations through weighted callbacks" begin
    for factory in (quantum_dimer, quantum_born), method in (:trapezoid, :lobatto)
      canonical = quantum_solve(factory, method, 8)
      p = factory()
      rule = method == :trapezoid ? reference_trapezoid! : reference_lobatto!
      grid = method == :trapezoid ? canonical.t : canonical.t[1:2:end]
      sol = rule(p.fv!, p.fd!, p.G, grid; p.callback, atol=1e-13)
      @test sol.t == canonical.t
      @test quantum_reference_error(p.G, canonical.G) < 5e-15
    end
  end
  for factory in (quantum_dimer, quantum_born)
    coarse, medium, fine = [quantum_solve(factory, :lobatto, n) for n in (8, 16, 32)]
    if factory == quantum_dimer
      errors = [quantum_exact_error(s.G, s.t) for s in (coarse, medium, fine)]
      @test all(14 .< errors[1:2] ./ errors[2:3] .< 18)
      @test last(errors) < 7e-7
    else
      # Independent trapezoidal discretization, extrapolated on the common grid.
      r1, r2 = [quantum_solve(factory, :trapezoid, n) for n in (64, 128)]
      reference = [GreenFunction((4f.data[:,:,1:2:end,1:2:end]-c.data)/3, SkewHermitian)
                   for (c,f) in zip(r1.G,r2.G)]
      errors = [quantum_reference_error(s.G, reference) for s in (coarse, medium, fine)]
      @test all(14 .< errors[1:2] ./ errors[2:3] .< 18)
      @test last(errors) < 5e-7
    end
    for s in (coarse, medium, fine)
      inv = quantum_invariants(s.G, s.t)
      @test maximum(values(inv)) < 2e-12
    end
  end
end
end # module
