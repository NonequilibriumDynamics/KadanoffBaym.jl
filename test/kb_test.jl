using Test

include("../src/KadanoffBaym.jl")
using RecursiveArrayTools
using OrdinaryDiffEq

# Define your Green functions at t0
Lesser0 = KadanoffBaym.LesserGF(ones(Float64,1,1))
Greater0 = KadanoffBaym.GreaterGF(ones(ComplexF64,1,1))

# Pack them in an ArrayPartition
u0 = ArrayPartition(Lesser0, Greater0);

# Remember that `u` here is also an ArrayPartition-like element
function f(u, p, t, t′)
  return ArrayPartition(u.x[1][t,t′], u.x[2][t,t′])
end

# ODE problem is defined by the rhs, initial condition and time span
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)

# Algorithm to timestep is the (Kadanoff-Baym) ABM43 (only really this one exists)
alg = KadanoffBaym.KB{ABM43}()

# This algorithm requires a fixed time
dt = 0.001

sol = solve(prob, alg, dt)

@test sol.u.x[1][1,:][end] ≈ exp(1)
@test sol.u.x[1][:,1][end] ≈ -exp(1)
@test sol.u.x[1][end] ≈ exp(2)
