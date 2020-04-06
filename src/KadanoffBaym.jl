module KadanoffBaym

using LinearAlgebra
using MuladdMacro
using Parameters
using Reexport
@reexport using EllipsisNotation
# @reexport using RecursiveArrayTools

# import DiffEqBase: __init, solve!
# @reexport using OrdinaryDiffEq

include("utils.jl")
export trapz

include("gf.jl")
export GreenFunction, Lesser, Greater

# include("common.jl")
# include("core.jl")
# export KB

include("vcabm.jl")
include("kb.jl")
export kbsolve

end # module