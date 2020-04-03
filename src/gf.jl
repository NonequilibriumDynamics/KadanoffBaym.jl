abstract type GreenFunctionType end

# """
# Defined as
#     G^<(t,t') = -i < a^{\dagger}(t') a(t) > for t <= t'
# """
struct Lesser <: GreenFunctionType end

# """
# Defined as
#     G^>(t,t') = -i < a(t)a^{\dagger}(t')  > for t > t'
# """
struct Greater <: GreenFunctionType end

"""
    GreenFunction

  A 2-time Green function with array indexing operations respecting the 
symmetries of the Green function.
"""
struct GreenFunction{T,S<:AbstractArray{<:T},U}
  data::S

  function GreenFunction{T,S,U}(A) where {T,S<:AbstractArray{<:T},U<:GreenFunctionType}
    @assert ndims(A) == 2 || ndims(A) == 4
    new{T,S,U}(A)
  end
end

GF_type(::Type{S}, U) where {T<:Number,S<:AbstractArray{T}} = GreenFunction{T,S,U}
GF_type(::Type{S}, U) where {T<:AbstractArray,S<:AbstractArray{T}} = GreenFunction{AbstractArray,S,U}

function GreenFunction(A::AbstractArray, U::Type{<:GreenFunctionType})
  LinearAlgebra.checksquare(A)
  return GF_type(typeof(A), U)(A)
end

Base.convert(T::Type{<:GreenFunction}, m::AbstractArray) = T(m)
Base.copy(A::GreenFunction) = typeof(A)(copy(A.data))
Base.eltype(::GreenFunction{T,S,U}) where {T,S,U} = eltype(T)
Base.size(A::GreenFunction) = size(A.data)
Base.getindex(A::GreenFunction, I::Int64) = error("Single indexing not allowed")

@inline Base.getindex(A::GreenFunction, I::Vararg{Union{Int64,Colon},2}) = Base.getindex(A.data, I..., ..)
@inline Base.getindex(A::GreenFunction, I...) = Base.getindex(A.data, I...)

@inline Base.setindex!(A::GreenFunction, v, I::Int64) = error("Single indexing not supported")
@inline Base.setindex!(A::GreenFunction{T,S,<:Union{Greater,Lesser}}, v, F::Vararg{Union{Int64, Colon}, 2}) where {T,S,F1,F2} = __setindex!(A, v, F, ..)
@inline Base.setindex!(A::GreenFunction{T,S,<:Union{Greater,Lesser}}, v, I...) where {T,S,F1,F2} = __setindex!(A, v, front2_last(I)...)
@inline function __setindex!(A::GreenFunction{T,S,<:Union{Greater,Lesser}}, v, F::Tuple{F1,F2}, I...) where {T,S,F1,F2}
  if ==(F...)
    setindex!(A.data, v, F..., I...)
  else 
    setindex!(A.data, v, F..., I...)
    setindex!(A.data, -adjoint(v), reverse(F)..., I...)
  end
end

# NOTE: This is absolutely fundamental to make sure that getelement of VectorOfArray returns non-eltype-Any arrays
RecursiveArrayTools.VectorOfArray(vec::AbstractVector{GreenFunction}, dims::NTuple{N}) where {N} = error("eltype of the GreenFunctions must be the same")
RecursiveArrayTools.VectorOfArray(vec::AbstractVector{GreenFunction{T,S}}, dims::NTuple{N}) where {T, S, N} = VectorOfArray{T, N, typeof(vec)}(vec)

for g in (:GreenFunction, )
  for f in (:-, :conj, :real, :imag, :adjoint, :transpose, :inv)
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

function resize(A::GreenFunction, t::Vararg{Int,2})
  if eltype(A) <: AbstractArray
    newdata = fill(eltype(A)(undef,t...,size(first(A))...))
  else
    newdata = typeof(A)(zeros(eltype(A),t...,tail2(size(A))...))
  end

  T, T′ = min.(first2(size(A)), t)

  for t=1:T, t′=1:T′
    @views newdata[t,t′] = A[t,t′]
  end

  return newdata
end

"""
    UnstructuredGreenFunction

  This 2-time GreenFunction stores its data in a lower-triangular matrix
configuration. This allows for a flexible resizing of the time-dimension, but 
array indexing might be slower due to the data not being aligned.
"""
struct UnstructuredGreenFunction{T,N,U} <: AbstractArray{T,N}
  data::Vector{Vector{T}}
  
  function UnstructuredGreenFunction(a::T, U::GreenFunctionType) where T<:Union{Number,AbstractArray}
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
