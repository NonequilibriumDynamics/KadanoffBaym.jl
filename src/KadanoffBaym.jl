# __precompile__(false)

module KadanoffBaym

using LinearAlgebra
using EllipsisNotation
using RecursiveArrayTools
# using MuladdMacro
using Parameters
using DataStructures
using OrdinaryDiffEq

using Reexport
@reexport using DiffEqBase
import DiffEqBase.__init

include("utils.jl")
include("gf.jl")

include("common.jl")
include("core.jl")

end