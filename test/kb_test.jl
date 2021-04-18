using Test

include("../src/KadanoffBaym.jl")
using .KadanoffBaym

λ = 0.2

# Define your Green functions at t0
ggf = GreenFunction(zeros(ComplexF64,1,1,1,1), Greater)
lgf = GreenFunction(1im * ones(ComplexF64,1,1,1,1), Lesser)

# Pack them in a VectorOfArray
u0 = VectorOfArray([ggf, lgf]);
@assert eltype(u0[1,1,..]) !== Any

# Remember that `u` here is also an ArrayPartition-like element
function f(u, p, t, t′)
  return VectorOfArray([1im * u[2][t,t′], 1im * u[1][t,t′] - λ * u[2][t,t′]])
end

# ODE problem is defined by the rhs, initial condition and time span
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)

# Algorithm to timestep is the (Kadanoff-Baym) ABM43 (only really this one exists)
alg = KB{ABM43}()

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
    @test sol.u[1][:,1][i] ≈ sol1(t, lgf[1,1,1,1], ggf[1,1,1,1], λ)
    @test sol.u[2][:,1][i] ≈ sol2(t, lgf[1,1,1,1], ggf[1,1,1,1], λ)
  end
end
