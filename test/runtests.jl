using KadanoffBaym
using LinearAlgebra
using SparseArrays
using Test

@testset verbose=true "KadanoffBaym.jl" begin
  @testset "GreenFunction" begin
    include("gf.jl")
  end

  @testset "Solver" begin
    include("quadrature.jl")
    include("utils.jl")
    include("vcabm.jl")
    include("kb.jl")
  end

  @testset "Langreth" begin
    include("langreth.jl")
  end

  @testset "Quantum examples" begin
    include("examples.jl")
    include("tmatrix.jl")
  end

  @testset "Wigner" begin
    include("wigner.jl")
  end
end
