module KadanoffBaym

using Requires
using QuadGK
import OrdinaryDiffEq

export GreenFunction, Symmetrical, SkewHermitian
export kbsolve!

include("utils.jl")
include("gf.jl")
include("vie.jl")
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
