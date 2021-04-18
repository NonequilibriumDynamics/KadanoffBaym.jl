abstract type GreenFunctionType end

"""
"""
struct Classical <: GreenFunctionType end

"""
Defined as
    G^<(t,t') = -i < a^{\\dagger}(t') a(t) > for t ≺ t'
"""
struct Lesser <: GreenFunctionType end

"""
Defined as
    G^>(t,t') = -i < a(t)a^{\\dagger}(t')  > for t′ ≺ t
"""
struct Greater <: GreenFunctionType end

"""
Defined as
    G^>(iτ,t') = -i < a(iτ)a^{\\dagger}(t')  > for t′ ≺ iτ
"""
struct MixedLesser <: GreenFunctionType end

"""
Defined as
    G^<(t,iτ) = -i < a(t)a^{\\dagger}(iτ)  > for iτ ≺ t
"""
struct MixedGreater <: GreenFunctionType end

"""
    GreenFunction

A 2-time Green function with array indexing respecting the symmetries of the 
Green function.

_Note_: The time-dimensions **must** be the last dimensions of the `data` 
array. When indexing with just 2 indices, the time-arguments will be chosen.

# Example
```
time_dim = 10
spin_dim = 2
gf = GreenFunction(zeros(spin_dim, spin_dim, time_dim, time_dim), Lesser)
```
"""
mutable struct GreenFunction{T,N,A,U<:GreenFunctionType} <: AbstractArray{T,N}
  data::A
end

function GreenFunction(G::AbstractArray, U::Type{<:GreenFunctionType})
  if U <: Union{Classical, Greater, Lesser}
    @assert ==(last2(size(G))...) "Time dimension ($(last2(size(G)))) must be a square"
  end
  GreenFunction{eltype(G), ndims(G), typeof(G), U}(G)
end

@inline Base.size(G::GreenFunction, I...) = size(G.data, I...)
@inline Base.length(G::GreenFunction) = length(G.data)
@inline Base.ndims(G::GreenFunction) = ndims(G.data)
@inline Base.axes(G::GreenFunction, d) = axes(G.data, d)

Base.copy(G::GreenFunction) = oftype(G, copy(G.data))
Base.eltype(::GreenFunction{T}) where {T} = T

@inline Base.getindex(::GreenFunction{T,S,<:Union{Classical,Greater,Lesser}}, I) where {T,S} = error("Single indexing not allowed")
Base.@propagate_inbounds Base.getindex(G::GreenFunction{T,N,A,<:Union{Classical,Greater,Lesser}}, I...) where {T,N,A} = G.data[..,I...]
Base.@propagate_inbounds Base.getindex(G::GreenFunction{T,N,A,<:Union{MixedGreater,MixedLesser}}, I...) where {T,N,A} = G.data[..,I...]

@inline Base.setindex!(::GreenFunction{T,S,<:Union{Classical,Greater,Lesser}}, v, I) where {T,S} = error("Single indexing not allowed")
Base.@propagate_inbounds function Base.setindex!(G::GreenFunction{T,N,A,<:Union{Greater,Lesser}}, v, I...) where {T,N,A}
  ts = last2(I)
  jj = front2(I)

  if ==(ts...)
    G.data[..,jj...,ts...] = v
  else 
    G.data[..,jj...,ts...] = v
    G.data[..,jj...,reverse(ts)...] = -adjoint(v)
  end
end
Base.@propagate_inbounds function Base.setindex!(G::GreenFunction{T,N,A,Classical}, v, I...) where {T,N,A}
  ts = last2(I)
  jj = front2(I)

  if ==(ts...)
    G.data[..,jj...,ts...] = v
  else 
    G.data[..,jj...,ts...] = v
    G.data[..,jj...,reverse(ts)...] = v
  end
end
Base.@propagate_inbounds function Base.setindex!(G::GreenFunction{T,N,A,<:Union{MixedGreater,MixedLesser}}, v, I...) where {T,N,A}
  G.data[..,I] = v
end

for g in (:GreenFunction, )
  for f in (:-, :conj, :real, :imag, :adjoint, :transpose, :zero)
    @eval (Base.$f)(G::$g{T,N,A,U}) where {T,N,A,U} = $g(Base.$f(G.data), U)
  end

  for f in (:+, :-, :/, :\, :*)
    if f != :/  
      @eval (Base.$f)(a::Number, G::$g{T,N,A,U}) where {T,N,A,U} = $g(Base.$f(a, G.data), U)
    end
    if f != :\
        @eval (Base.$f)(G::$g{T,N,A,U}, b::Number) where {T,N,A,U} = $g(Base.$f(G.data, b), U)
    end
  end

  for f in (:+, :-, :\, :/, :*)
      @eval (Base.$f)(G1::$g{T,N,A,U}, G2::$g{T,N,A,U}) where {T,N,A,U} = $g(Base.$f(G1.data, G2.data), U)
  end

  @eval Base.:+(G::$g{T,N,A,U}, Gs::$g{T,N,A,U}...) where {T,N,A,U} = Base.+(G.data, getfield.(Gs,:data)...)
end

function Base.show(io::IO, x::GreenFunction)
  # if get(io, :compact, false) || get(io, :typeinfo, nothing) == GreenFunction
    Base.show_default(IOContext(io, :limit => true), x)
  # else
  #   # dump(IOContext(io, :limit => true), p, maxdepth=1)
  #   for field in fieldnames(typeof(x))
  #     if field === :data
  #       print(io, "\ndata: ")
  #       Base.show(io, MIME"text/plain"(), x.data)
  #     else
  #       Base.show(io, getfield(x, field))
  #     end
  #   end
  # end
end

Base.resize!(A::GreenFunction, t::Int) = A.data = _resize!(A.data, t)

function _resize!(a::Array{T,N}, t::Int) where {T,N}
  a′ = Array{T,N}(undef, front2(size(a))..., t, t)

  k =  min(last(size(a)), t)
  for t in 1:k, t′ in 1:k
      a′[..,t′,t] = a[..,t′,t]
  end

  return a′
end

"""
    SkewHermitianArray

Provides a different (but not necessarily more efficient) data storage for 
elastic skew-Hermitian data
"""
struct SkewHermitianArray{T,N} <: AbstractArray{T,N}
  data::Vector{Vector{T}}

  function SkewHermitianArray(a::T) where {T<:Union{Number, AbstractArray}}
    new{T,2+ndims(a)}([[a]])
  end
end

@inline Base.size(a::SkewHermitianArray{<:Number}) = (length(a.data), length(a.data))
@inline Base.size(a::SkewHermitianArray{<:AbstractArray}) = (size(a.data[1][1])..., length(a.data), length(a.data))

Base.getindex(::SkewHermitianArray, I) = error("Single indexing not allowed")
@inline function Base.getindex(a::SkewHermitianArray{T,N}, i::Int, j::Int) where {T,N}
  return (i >= j) ? a.data[i-j+1][j] : -adjoint(a.data[j-i+1][i])
end

@inline function Base.setindex!(a::SkewHermitianArray, v, i::Int, j::Int)
  if i >= j
    a.data[i-j+1][j] = v
  else
    a.data[j-i+1][i] = -adjoint(v)
  end
end

function _resize!(a::SkewHermitianArray{T}, t::Int) where {T}
  l = length(a)

  resize!(a.data, t)

  for k in 1:min(l, t)
    resize!(a.data[k], t-k+1)
  end
  for k in min(l,t)+1:t
    a.data[k] = Array{T}(undef, t-k+1)
  end
end
