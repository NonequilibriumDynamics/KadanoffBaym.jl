using KadanoffBaym
using Test

@testset "All tests" begin
  @testset "Green functions" begin
    include("gf.jl")
  end

  @testset "Kadanoff-Baym solver" begin
    include("kb.jl")
  end
end
