include("../src/KadanoffBaym.jl")
# using .KadanoffBaym

using Test: @test, @testset
using RecursiveArrayTools: ArrayPartition
using OrdinaryDiffEq: ODEProblem, solve, ABM43

λ = 0.2

# Define your Green functions at t0
Greater = KadanoffBaym.GreaterGF(zeros(ComplexF64,1,1))
Lesser = KadanoffBaym.LesserGF(1im * ones(ComplexF64,1,1))

# Pack them in an ArrayPartition
u0 = ArrayPartition(Greater, Lesser);

# Remember that `u` here is also an ArrayPartition-like element
function f(u, p, t, t′)
  u1, u2 = u.x[1], u.x[2]
  return ArrayPartition(1im * u2[t,t′], 1im * u1[t,t′] - λ * u2[t,t′])
end

# ODE problem is defined by the rhs, initial condition and time span
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)

# Algorithm to timestep is the (Kadanoff-Baym) ABM43 (only really this one exists)
alg = KadanoffBaym.KB{ABM43}()

# This algorithm requires a fixed time
dt = 0.001

sol = solve(prob, alg, dt)

function sol1(t, u1_0, u2_0, λ)
  s = sqrt(Complex(λ^2 - 4))
  return exp(-λ * t / 2) * (u2_0 * cosh(0.5 * t * s) + (2im * u1_0 + u2_0 * λ) * sinh(0.5 * t * s) / s)
end

function sol2(t, u1_0, u2_0, λ)
  s = sqrt(Complex(λ^2 - 4))
  return exp(-λ * t / 2) * (u1_0 * cosh(0.5 * t * s) + (2im * u2_0 - u1_0 * λ) * sinh(0.5 * t * s) / s)
end

times = first.(sol.t[:,1])

@testset begin
  for (i, t) in Iterators.enumerate(times)
    @test sol.u.x[1][:,1][i] ≈ sol1(t, Lesser[1,1], Greater[1,1], λ)
    @test sol.u.x[2][:,1][i] ≈ sol2(t, Lesser[1,1], Greater[1,1], λ)
  end
end
