abstract type AbstractTimeOrderedGreenFunction end

"""
    TimeOrderedGreenFunction(L::AbstractMatrix, G::AbstractMatrix)

A simple time-ordered Green function structure for a hassle-free computation of the Langreth rules.

# Parameters
  - `L::AbstractMatrix`: The *lesser* component
  - `G::AbstractMatrix`: The *greater* component
"""
struct TimeOrderedGreenFunction{T} <: AbstractTimeOrderedGreenFunction
  L::T   # Lesser
  G::T   # Greater
end

Base.:*(a::Number, g::TimeOrderedGreenFunction) = TimeOrderedGreenFunction(a * lesser(g), a * greater(g))
Base.:+(g1::TimeOrderedGreenFunction, g2::TimeOrderedGreenFunction) = TimeOrderedGreenFunction(lesser(g1) + lesser(g2), greater(g1) + greater(g2))

"""
    conv(L::AbstractTimeOrderedGreenFunction, R::AbstractTimeOrderedGreenFunction, ws::UpperTriangular)

Calculates a time-convolution between time-ordered Green functions through the Langreth rules.

# Parameters
  - `L::AbstractTimeOrderedGreenFunction`: The left time-ordered Green function
  - `R::AbstractTimeOrderedGreenFunction`: The right time-ordered Green function
  - `ws::UpperTriangular`: An upper-triangular weight matrix containing the integration weights
"""
function conv(L::AbstractTimeOrderedGreenFunction, R::AbstractTimeOrderedGreenFunction, ws::UpperTriangular)
  Lᴿ, Rᴬ = retarded(L), advanced(R)
  # Structured array operations require scalar zero/one units, absent for matrix blocks.
  Lᴿ = eltype(Lᴿ) <: AbstractMatrix ? Matrix(Lᴿ) : Lᴿ
  Rᴬ = eltype(Rᴬ) <: AbstractMatrix ? Matrix(Rᴬ) : Rᴬ
  Lᴿ = Lᴿ .* transpose(ws)
  Rᴬ = ws .* Rᴬ
  return TimeOrderedGreenFunction(
    Lᴿ * lesser(R) + lesser(L) * Rᴬ,
    Lᴿ * greater(R) + greater(L) * Rᴬ,
  )
end

"""
    greater(g::AbstractTimeOrderedGreenFunction)

Extract the *greater* Keldysh component ``G^>(t,t')``.
"""
greater(g::TimeOrderedGreenFunction) = g.G

"""
    lesser(g::AbstractTimeOrderedGreenFunction)

Extract the *lesser* Keldysh component ``G^<(t,t')``.
"""
lesser(g::TimeOrderedGreenFunction) = g.L

"""
    advanced(g::AbstractTimeOrderedGreenFunction)

Compute the *advanced* Green function ``G^A(t,t') = \\theta(t'-t)(G^< - G^>)``.
"""
advanced(g::TimeOrderedGreenFunction) = UpperTriangular(lesser(g) - greater(g))

"""
    retarded(g::AbstractTimeOrderedGreenFunction)

Compute the *retarded* Green function ``G^R(t,t') = \\theta(t-t')(G^> - G^<)``.
"""
retarded(g::TimeOrderedGreenFunction) = LowerTriangular(greater(g) - lesser(g))
