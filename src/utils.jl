# ODE norm: Section II.4 (4.11)
@inline norm(u, y=nothing) = LinearAlgebra.norm(u) / sqrt(total_length(u))
@inline total_length(u::Number) = length(u)
@inline total_length(u::AbstractArray{<:Number}) = length(u)
@inline total_length(u::AbstractArray{<:AbstractArray}) = sum(total_length, u)
@inline total_length(u::VectorOfArray) = sum(total_length, u.u)

# Error estimation and norm: Section II.4 Eq. (4.11)
@inline function calculate_residuals!(out::AbstractArray, ũ::AbstractArray, u₀::AbstractArray, u₁::AbstractArray, atol, rtol, norm)
  @. out = calculate_residuals!(out, ũ, u₀, u₁, atol, rtol, norm)
  return out
end
@inline function calculate_residuals!(out::AbstractArray{<:Number}, ũ::AbstractArray{<:Number}, u₀::AbstractArray{<:Number}, u₁::AbstractArray{<:Number}, atol, rtol, norm)
  @. out = calculate_residuals(ũ, u₀, u₁, atol, rtol, norm)
  return out
end
@inline function calculate_residuals(ũ::Number, u₀::Number, u₁::Number, atol::Real, rtol::Real, norm)
  return ũ / (atol + max(norm(u₀), norm(u₁)) * rtol)
end

# Starting Step Size: Section II.4
function initial_step(f0, u0, atol, rtol)
  sc = atol + rtol * norm(u0)
  d0 = norm(u0 ./ sc)
  d1 = norm(f0 ./ sc)

  return dt0 = min(d0, d1) < 1e-5 ? 1e-6 : 1e-2 * d0 / d1
end

# Returns a `Tuple` consisting of all but the last 2 components of `t`.
@inline front2(v1, v2) = ()
@inline front2(v, t...) = (v, front2(t...)...)

# Returns a `Tuple` consisting of the last 2 components of `t`.
@inline last2(v1, v2) = (v1, v2)
@inline last2(v, t...) = last2(t...)

function skew_hermitify!(x)
  for i in 1:size(x, 1)
    for j in 1:(i - 1)
      x[j,i] = -conj(x[i,j])
    end

    x[i,i] = eltype(x) <: Real ? 0.0 : im * imag(x[i,i])
  end
  return x
end
