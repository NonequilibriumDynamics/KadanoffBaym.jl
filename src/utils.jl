# Component RMS within each value; maximum over fields and time coordinates.
@inline norm(u) = LinearAlgebra.norm(u) / sqrt(total_length(u))
@inline state_norm(u) = maximum(norm, Iterators.flatten(u))
@inline total_length(u::Number) = length(u)
@inline total_length(u::AbstractArray{<:Number}) = length(u)
@inline total_length(u::AbstractArray{<:AbstractArray}) = sum(total_length, u)

# Error estimation and norm: Section II.4 Eq. (4.11)
@inline function calculate_residuals!(out::AbstractArray, ũ, u₀, u₁, atol, rtol)
  for i in eachindex(out, ũ, u₀, u₁)
    calculate_residuals!(out[i], ũ[i], u₀[i], u₁[i], atol, rtol)
  end
  return out
end
@inline function calculate_residuals!(out::AbstractArray{<:Number}, ũ::AbstractArray{<:Number}, u₀::AbstractArray{<:Number}, u₁::AbstractArray{<:Number}, atol, rtol)
  @. out = calculate_residuals(ũ, u₀, u₁, atol, rtol)
  return out
end
@inline function calculate_residuals(ũ::Number, u₀::Number, u₁::Number, atol::Real, rtol::Real)
  return ũ / (atol + max(norm(u₀), norm(u₁)) * rtol)
end

# Starting Step Size: Section II.4
function initial_step(f0, u0, atol, rtol)
  scaled = zero.(u0)
  calculate_residuals!(scaled, u0, u0, u0, atol, rtol)
  d0 = state_norm(scaled)
  calculate_residuals!(scaled, f0, u0, u0, atol, rtol)
  d1 = state_norm(scaled)

  return min(d0, d1) < 1e-5 ? 1e-6 : 1e-2 * d0 / d1
end

# Returns a `Tuple` consisting of all but the last 2 components of `t`.
@inline front2(v1, v2) = ()
@inline front2(v, t...) = (v, front2(t...)...)

# Returns a `Tuple` consisting of the last 2 components of `t`.
@inline last2(v1, v2) = (v1, v2)
@inline last2(v, t...) = last2(t...)

@inline recursivecopy(x::Number) = copy(x)
@inline recursivecopy(x::AbstractArray{<:Number}) = copy(x)
@inline recursivecopy(x::AbstractArray) = map(recursivecopy, x)

@inline _recursivecopy!(out::Number, src::Number) = src
@inline _recursivecopy!(out::AbstractArray{<:Number}, src::AbstractArray{<:Number}) = out === src ? out : copyto!(out, src)
@inline function _recursivecopy!(out::AbstractArray, src::AbstractArray)
  out === src && return out
  for i in eachindex(out, src)
    out[i] = _recursivecopy!(out[i], src[i])
  end
  return out
end

@inline muladd!(out::Number, a, x, y) = muladd(a, x, y)
@inline function muladd!(out::Array{<:Number}, a::Number, x::Array{<:Number}, y::Union{Array{<:Number},Number})
  indices = y isa Number ? eachindex(out, x) : eachindex(out, x, y)
  @inbounds for i in indices
    out[i] = muladd(a, x[i], y isa Number ? y : y[i])
  end
  return out
end
@inline function muladd!(out::AbstractArray{<:Number}, a, x, y)
  @. out = muladd(a, x, y)
  return out
end
@inline function muladd!(out, a, x, y)
  for i in eachindex(out, x)
    out[i] = muladd!(out[i], a, x[i], y isa Number ? y : y[i])
  end
  return out
end
