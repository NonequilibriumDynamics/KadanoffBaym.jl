using Parameters
using MuladdMacro

using Reexport
@reexport using OrdinaryDiffEq

struct KadanoffBaym <: OrdinaryDiffEq.OrdinaryDiffEqAlgorithm end
export KadanoffBaym

OrdinaryDiffEq.alg_order(alg::KadanoffBaym) = 4

# OrdinaryDiffEq.@cache
mutable struct KadanoffBaym4Cache{rateType} <: OrdinaryDiffEq.OrdinaryDiffEqConstantCache
  k2::rateType
  k3::rateType
  k4::rateType
  step::Int
end

# OrdinaryDiffEq.@cache
mutable struct KadanoffBaym43Cache{rateType} <: OrdinaryDiffEq.OrdinaryDiffEqConstantCache
  k2::rateType
  k3::rateType
  k4::rateType
  step::Int
end