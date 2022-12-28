using KadanoffBaym
using LinearAlgebra
using Test

@testset verbose=true "KadanoffBaym.jl" begin
  @testset "GreenFunction" begin
    include("gf.jl")
  end

  @testset "Solver" begin
    include("kb.jl")
  end

  @testset "Langreth" begin
    include("langreth.jl")
  end
end
