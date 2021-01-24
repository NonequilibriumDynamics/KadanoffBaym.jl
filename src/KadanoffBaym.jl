module KadanoffBaym

using Base: @propagate_inbounds, front

using LinearAlgebra
using Parameters: @unpack
using EllipsisNotation
using Requires

export GreenFunction, Classical, Lesser, Greater, MixedLesser, MixedGreater
export kbsolve

include("utils.jl")
include("gf.jl")
include("vcabm.jl")
include("volterra.jl")
include("kb.jl")

function __init__()
  @require FFTW="7a1cc6ca-52ef-59f5-83cd-3a7055c09341" begin
    @require Interpolations="a98d9a8b-a2ab-59e6-89dd-64a1c18fca59" begin
      using .FFTW, .Interpolations
      include("wigner.jl")
      export wigner_transform, wigner_transform_itp
    end
  end
end

end # module
