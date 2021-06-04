module KadanoffBaym

using LinearAlgebra
using EllipsisNotation
using UnPack
using Requires
using RecursiveArrayTools

export GreenFunction, SymmetricSymmetry, SkewHermitianSymmetry, OneTimeSymmetry
export kbsolve

include("utils.jl")
include("gf.jl")
include("vcabm.jl")
include("kb.jl")

@init @require FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341" begin
  @require Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59" begin
    using .FFTW, .Interpolations
    include("wigner.jl")
    export wigner_transform, wigner_transform_itp
  end
end

end # module
