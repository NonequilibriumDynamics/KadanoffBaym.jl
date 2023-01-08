abstract type AbstractTimeOrderedGreenFunction end

"""
    TimeOrderedGreenFunction(L::AbstractMatrix, G::AbstractMatrix)

A simple time-ordered Green function structure for a hassle-free computation of the Langreth rules.

# Parameters
  - `L::AbstractMatrix`: The *lesser* component
  - `G::AbstractMatrix`: The *greater* component
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
  L::TA
  R::TB
  ws::UpperTriangular
end

"""
    conv(L::AbstractTimeOrderedGreenFunction, R::AbstractTimeOrderedGreenFunction, ws::UpperTriangular)

Calculates a time-convolution between time-ordered Green functions through the Langreth rules.

# Parameters
  - `L::AbstractTimeOrderedGreenFunction`: The left time-ordered Green function
  - `R::AbstractTimeOrderedGreenFunction`: The right time-ordered Green function
  - `ws::UpperTriangular`: An upper-triangular weight matrix containing the integration weights
"""
function conv(L::AbstractTimeOrderedGreenFunction, R::AbstractTimeOrderedGreenFunction, ws::UpperTriangular)
  c = TimeOrderedConvolution(L, R, ws)
  return TimeOrderedGreenFunction(lesser(c), greater(c))
end

# Langreth's rules
greater(g::TimeOrderedGreenFunction) = g.G
lesser(g::TimeOrderedGreenFunction) = g.L
advanced(g::TimeOrderedGreenFunction) = UpperTriangular(lesser(g) - greater(g))
retarded(g::TimeOrderedGreenFunction) = adjoint(advanced(g))

greater(c::TimeOrderedConvolution) = (retarded(c.L) .* adjoint(c.ws)) * greater(c.R) + greater(c.L) * (c.ws .* advanced(c.R)) |> skew_hermitify!
lesser(c::TimeOrderedConvolution) = (retarded(c.L) .* adjoint(c.ws)) * lesser(c.R) + lesser(c.L) * (c.ws .* advanced(c.R)) |> skew_hermitify!
