abstract type GreenFunctionType end

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
"""
struct Classical <: GreenFunctionType end

"""
    GreenFunction

A 2-time Green function with array indexing operations respecting the 
symmetries of the Green function.

_Note_: The time-dimensions **must** be the last dimensions of the `data` 
array. When indexing with just 2 indices, the time-arguments will be chosen.

# Example
```
time_dim = 10
spin_dim = 2
gf = GreenFunction(zeros(spin_dim, spin_dim, time_dim, time_dim), Lesser)
```
"""
mutable struct GreenFunction{T,S<:AbstractArray{<:T},U<:GreenFunctionType}
  data::S
end

function GreenFunction(A::AbstractArray, U::Type{<:GreenFunctionType})
  if U <: GreaterOrLesser
    @assert ==(last2(size(A))...) "Time dimension ($(last2(size(A)))) must be a square"
  end
  GF_type(typeof(A), U)(A)
end

function GF_type(::Type{S}, U) where {T<:Number,S<:AbstractArray{T}}
  GreenFunction{T,S,U}
end

# function GF_type(::Type{S}, U) where {T<:AbstractArray,S<:AbstractArray{T}}
#   GreenFunction{AbstractArray,S,U}
# end

Base.copy(A::GreenFunction) = oftype(A, copy(A.data))
Base.eltype(::GreenFunction{T,S,U}) where {T,S,U} = eltype(T)

@inline Base.size(A::GreenFunction,I...) = size(A.data,I...)
@inline Base.length(A::GreenFunction) = length(A.data)
# @inline Base.ndims(A::GreenFunction{T,S,U}) where {T,S,U} = ndims(S)
# @inline Base.axes(A::GreenFunction, d) = axes(A.data, d)

const GreaterOrLesser{T,S} = GreenFunction{T,S,<:Union{Greater,Lesser}}
@inline function Base.getindex(A::GreaterOrLesser, I::Union{Int64,Colon})
  error("Single indexing not allowed")
end
@propagate_inbounds function Base.getindex(A::GreaterOrLesser, I::Vararg{Union{Int64,Colon},2})
  Base.getindex(A.data, .., I...) 
end
@propagate_inbounds function Base.getindex(A::GreaterOrLesser, I...)
  Base.getindex(A.data, I...)
end
@propagate_inbounds function Base.setindex!(A::GreaterOrLesser, v, I::Union{Int64,Colon})
  error("Single indexing not allowed")
end
@propagate_inbounds function Base.setindex!(A::GreaterOrLesser, v, F::Vararg{Union{Int64,Colon}, 2})
  __setindex!(A, v, (..,), F)
end
@propagate_inbounds function Base.setindex!(A::GreaterOrLesser, v, I...)
  __setindex!(A, v, front_last2(I)...)
end
@propagate_inbounds function __setindex!(A::GreaterOrLesser, v, F, L::Tuple{F1,F2}) where {F1,F2}
  if ==(L...)
    setindex!(A.data, v, F..., L...)
  else 
    setindex!(A.data, v, F..., L...)
    setindex!(A.data, -adjoint(v), F..., reverse(L)...)
  end
end

const MixedGOL{T,S} = GreenFunction{T,S,<:Union{MixedGreater,MixedLesser}}
@propagate_inbounds function Base.getindex(A::MixedGOL, I::Int64)
  Base.getindex(A.data, .., I)
end
@propagate_inbounds function Base.setindex!(A::MixedGOL, v, I::Int64)
  Base.setindex!(A.data, v, .., I)
end

const Classical_{T,S} = GreenFunction{T,S,Classical}
@inline function Base.getindex(A::Classical_, I::Union{Int64,Colon})
  error("Single indexing not allowed")
end
@propagate_inbounds function Base.getindex(A::Classical_, I::Vararg{Union{Int64,Colon},2})
  Base.getindex(A.data, .., I...) 
end
@propagate_inbounds function Base.getindex(A::Classical_, I...)
  Base.getindex(A.data, I...)
end
@propagate_inbounds function Base.setindex!(A::Classical_, v, I::Union{Int64,Colon})
  error("Single indexing not allowed")
end
@propagate_inbounds function Base.setindex!(A::Classical_, v, F::Vararg{Union{Int64,Colon}, 2})
  __setindex!(A, v, (..,), F)
end
@propagate_inbounds function Base.setindex!(A::Classical_, v, I...)
  __setindex!(A, v, front_last2(I)...)
end
@propagate_inbounds function __setindex!(A::Classical_, v, F, L::Tuple{F1,F2}) where {F1,F2}
  if ==(L...)
    setindex!(A.data, v, F..., L...)
  else 
    setindex!(A.data, v, F..., L...)
    setindex!(A.data, v, F..., reverse(L)...)
  end
end

# Disable broadcasting
Base.dotview(A::GreenFunction, I...) = error("Not supported")

# NOTE: This is absolutely fundamental to make sure that getelement of VectorOfArray returns non-eltype-Any arrays
# RecursiveArrayTools.VectorOfArray(vec::AbstractVector{GreenFunction}, dims::NTuple{N}) where {N} = error("eltype of the GreenFunctions must be the same")
# RecursiveArrayTools.VectorOfArray(vec::AbstractVector{GreenFunction{T,S}}, dims::NTuple{N}) where {T, S, N} = VectorOfArray{T, N, typeof(vec)}(vec)

for g in (:GreenFunction, )
  for f in (:-, :conj, :real, :imag, :adjoint, :transpose, :inv, :zero)
    @eval (Base.$f)(A::$g{T,S,U}) where {T,S,U} = $g(Base.$f(A.data), U)
  end

  for f in (:+, :-, :/, :\, :*)
    if f != :/  
      @eval (Base.$f)(A::Number, B::$g{T,S,U}) where {T,S,U} = $g(Base.$f(A, B.data), U)
    end
    if f != :\
        @eval (Base.$f)(A::$g{T,S,U}, B::Number) where {T,S,U} = $g(Base.$f(A.data, B), U)
    end
  end

  for f in (:+, :-, :\, :/)
      @eval (Base.$f)(A::$g{T,S,U}, B::$g{T,S,U}) where {T,S,U} = $g(Base.$f(A.data, B.data), U)
  end

  @eval Base.:*(A::$g{T,S,U}, B::$g{T,S,<:Union{Greater,Lesser}}) where{T,S,U<:Union{Greater,Lesser}} = $g(Base.:*(A.data, B.data), U)

  @eval Base.:+(A::$g{T,S,U}, B::$g{T,S,U}...) where {T,S,U} = Base.+(A.data, getfield.(B,:data)...)
end

function Base.show(io::IO, x::GreenFunction)
  if get(io, :compact, false) || get(io, :typeinfo, nothing) == GreenFunction
    Base.show_default(IOContext(io, :limit => true), x)
  else
    # dump(IOContext(io, :limit => true), p, maxdepth=1)
    for field in fieldnames(typeof(x))
      if field === :data
        print(io, "\ndata: ")
        Base.show(io, MIME"text/plain"(), x.data)
      else
        Base.show(io, getfield(x, field))
      end
    end
  end
end

# Base.resize!(A::GreaterOrLesser, t::Int) = Base.resize!(A, t, t)
# function Base.resize!(A::GreaterOrLesser, t::Vararg{Int,2})

Base.resize!(A::GreenFunction, t::Int) = Base.resize!(A, t, t)
function Base.resize!(A::GreenFunction, t::Vararg{Int,2})

  # if eltype(A) <: AbstractArray
  #   newdata = fill(eltype(A)(undef,t...,size(first(A))...))
  # else
  newdata = typeof(A)(zeros(eltype(A),front2(size(A))...,t...))
  # replace!(newdata.data, 0.0=>NaN)
  # end

  T = min(last(size(A)), last(t))

  for t=1:T, t′=1:T
#     newdata.data[t,t′] = A.data[t,t′]
    newdata.data[..,t,t′] = A.data[..,t,t′]
  end

  A.data = newdata.data
  return A
end

Base.resize!(A::MixedGOL, t::Int) = Base.resize!(A.data, front(size(A))..., t)

"""
    UnstructuredGreenFunction

  This 2-time GreenFunction stores its data in a lower-triangular matrix
configuration. This allows for a flexible resizing of the time-dimension, but 
array indexing might be slower due to the data not being aligned.
"""
struct UnstructuredGreenFunction{T,N,U} <: AbstractArray{T,N}
  data::Vector{Vector{T}}
  
  function UnstructuredGreenFunction(a::T, U::Type{<:GreenFunctionType}) where {T<:Union{Number,AbstractArray}}
    new{T,2+ndims(a),U}([[a]])
  end
end


function Base.getindex(a::UnstructuredGreenFunction{T,N,U}, i::Int, j::Int) where {T,N,U<:Union{Greater,Lesser}}
  return (i >= j) ? a.data[i-j+1][j] : -adjoint(a.data[j-i+1][i])
end

function Base.setindex!(a::UnstructuredGreenFunction{T,N,U}, v, i::Int, j::Int) where {T,N,U<:Union{Greater,Lesser}}
  if i >= j
    Base.setindex!(a.data[i-j+1],v,j)
  else
    Base.setindex!(a.data[j-i+1],-adjoint(v),i)
  end
end

@inline Base.size(a::UnstructuredGreenFunction) = (length(a.data), length(first(a.data)), size(first(first(a.data)))...)

function Base.resize!(a::UnstructuredGreenFunction, i::Int, j::Int)
  col = size(a, 2)
  
  resize!(a.data, j)
  for k in 1:min(col, j)
    resize!(a.data[k], i-k+1)
  end
  for k in min(col,j)+1:i
    a.data[k] = Array{eltype(a)}(undef, i-k+1)
  end
end
