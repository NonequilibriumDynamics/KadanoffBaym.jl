using KadanoffBaym
using Test

@testset verbose=true "KadanoffBaym.jl" begin
  @testset "GreenFunction" begin
    include("gf.jl")
  end

  @testset "Solver" begin
    include("kb.jl")
  end
end
