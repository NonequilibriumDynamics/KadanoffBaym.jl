abstract type AbstractTimeOrderedGreenFunction end

"""
"""
struct TimeOrderedGreenFunction <: AbstractTimeOrderedGreenFunction
  L   # Lesser
  G   # Greater  
end

Base.:*(a::Number, g::TimeOrderedGreenFunction) = TimeOrderedGreenFunction(a * lesser(g), a * greater(g))
Base.:+(g1::TimeOrderedGreenFunction, g2::TimeOrderedGreenFunction) = TimeOrderedGreenFunction(lesser(g1) + lesser(g2), greater(g1) + greater(g2))

"""
"""
struct TimeOrderedConvolution{TA<:AbstractTimeOrderedGreenFunction, TB<:AbstractTimeOrderedGreenFunction} <: AbstractTimeOrderedGreenFunction
  A::TA
  B::TB
  dts::UpperTriangular
end

"""
"""
function conv(A::AbstractTimeOrderedGreenFunction, B::AbstractTimeOrderedGreenFunction, dts)
  c = TimeOrderedConvolution(A, B, dts)
  return TimeOrderedGreenFunction(lesser(c), greater(c))
end

# Langreth's rules
greater(g::TimeOrderedGreenFunction) = g.G
lesser(g::TimeOrderedGreenFunction) = g.L
advanced(g::TimeOrderedGreenFunction) = UpperTriangular(lesser(g) - greater(g))
retarded(g::TimeOrderedGreenFunction) = adjoint(advanced(g))

greater(g::TimeOrderedConvolution) = (retarded(g.A) .* adjoint(g.dts)) * greater(g.B) + greater(g.A) * (g.dts .* advanced(g.B)) |> skew_hermitify!
lesser(g::TimeOrderedConvolution) = (retarded(g.A) .* adjoint(g.dts)) * lesser(g.B) + lesser(g.A) * (g.dts .* advanced(g.B)) |> skew_hermitify!
