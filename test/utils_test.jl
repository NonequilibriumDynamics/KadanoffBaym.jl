using Test
using KadanoffBaym

@testset "trapz" begin
  a = rand(ComplexF64, 33)
  b = rand(ComplexF64, 33)
  s1 = trapz(a,b)
  s2 = trapz(a,b,1,length(a))
  s3 = trapz(a,b,length(a),1)
  @test s1 == s2
  @test s1 == -s3
end