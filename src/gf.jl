abstract type GreenFunction end

# """
# Defined as
#     G^<(t,t') = -i < a^{\dagger}(t') a(t) > for t <= t'
# """
struct LesserGF{T,S<:AbstractArray{<:T}} <: GreenFunction
  data::S

  function LesserGF{T,S}(data) where {T, S<:AbstractArray{<:T}}
    # Base.require_one_based_indexing(data)
    new{T,S}(data)
  end
end

function LesserGF(A::AbstractArray)
  LinearAlgebra.checksquare(A)
  return lesserGF_type(typeof(A))(A)
end

LesserGF(::Type{T}, method, dims::Int...) where {T} = LesserGF(T(method, dims))
LesserGF(::Type{T}, dims::Int...) where {T} = LesserGF(T, undef, dims...)

function lesserGF_type(::Type{T}) where {S<:Number, T<:AbstractArray{S}}
  return LesserGF{S, T}
end
function lesserGF_type(::Type{T}) where {S<:AbstractArray, T<:AbstractArray{S}}
  return LesserGF{AbstractArray, T}
end

# """
# Defined as
#     G^>(t,t') = -i < a(t)a^{\dagger}(t')  > for t > t'
# """
struct GreaterGF{T,S<:AbstractArray{<:T}} <: GreenFunction
  data::S

  function GreaterGF{T,S}(data) where {T, S<:AbstractArray{<:T}}
    # Base.require_one_based_indexing(data)
    new{T,S}(data)
  end
end

function GreaterGF(A::AbstractArray)
  LinearAlgebra.checksquare(A)
  return greaterGF_type(typeof(A))(A)
end

GreaterGF(::Type{T}, method, dims::Int...) where {T} = GreaterGF(T(method, dims))
GreaterGF(::Type{T}, dims::Int...) where {T} = GreaterGF(T, undef, dims...)

function greaterGF_type(::Type{T}) where {S<:Number, T<:AbstractArray{S}}
  return GreaterGF{S, T}
end
function greaterGF_type(::Type{T}) where {S<:AbstractArray, T<:AbstractArray{S}}
  return GreaterGF{AbstractArray, T}
end

const LesserOrGreater{T,S} = Union{LesserGF{T,S}, GreaterGF{T,S}}

Base.eltype(::Type{<:LesserOrGreater{T,S}}) where {T,S} = eltype(S)
Base.convert(T::Type{<:GreenFunction}, m::AbstractArray) = T(m)

Base.size(A::LesserOrGreater) = size(A.data)
Base.length(A::LesserOrGreater) = length(A.data)
Base.size(A::LesserOrGreater{AbstractArray, S}) where S = tuplejoin(size(A.data), size(first(A.data)))
Base.length(A::LesserOrGreater{AbstractArray, S}) where S = length(A.data) * length(first(A.data))

# Base.firstindex(A::LesserOrGreater) = firstindex(A.data)
# Base.lastindex(A::LesserOrGreater) = lastindex(A.data)

Base.getindex(A::LesserOrGreater, F::Vararg{Union{Int64, Colon}, 2}) = Base.getindex(A.data, F..., ..)
Base.getindex(A::LesserOrGreater, I...) = Base.getindex(A.data, I...)
Base.getindex(A::LesserOrGreater{AbstractArray, S}, F::Vararg{Union{Int64, Colon}, 2}) where S = Base.getindex(A.data, F...)
Base.getindex(A::LesserOrGreater{AbstractArray, S}, I...) where S = Base.getindex(Base.getindex(A.data, front2(I)...), tail2(I)...)

Base.setindex!(A::LesserOrGreater, v, F::Vararg{Union{Int64, Colon}, 2}) = __setindex!(A, v, F, (..,))
Base.setindex!(A::LesserOrGreater, v, I...) = _setindex!(A, v, front2_last(I)...)

# _setindex!(A::LesserGF, v, L::NTuple{2, Int64}, F...) = begin @assert <=(L...) "t>t′"; __setindex!(A, v, L, F...) end
# _setindex!(A::GreaterGF, v, L::NTuple{2, Int64}, F...) = begin @assert >=(L...) "t<t′"; __setindex!(A, v, L, F...) end
_setindex!(A::LesserOrGreater, v, F::NTuple{2, Union{Int64, Colon}}, L::Tuple) = __setindex!(A, v, F, L)

@inline function __setindex!(A::LesserOrGreater, v, F::Tuple{T,U}, L::Tuple) where {T,U}
  if ==(F...)
    setindex!(A.data, v, F..., L...)
  else 
    setindex!(A.data, v, F..., L...)
    setindex!(A.data, -adjoint(v), reverse(F)..., L...)
  end
end

@inline function __setindex!(A::LesserOrGreater{AbstractArray, S}, v, F::Tuple{T,U}, L::Tuple) where {S,T,U}
  if ==(F...)
    setindex!(A.data[F...], v , L...)
  else 
    setindex!(A.data[F...], v , L...)
    setindex!(A.data[reverse(F)...], -adjoint(v) , L...)
  end
end

# Base.iterate(A::LesserOrGreater, state=1) = iterate(A.data, state)
Base.copy(A::LesserOrGreater) = typeof(A)(copy(A.data))

function symmetrize!(A::LesserOrGreater)
  T, T′ = first2(size(A))
  if typeof(A) <: LesserGF
    for t′ in 1:1:T′, t in 1:1:t′
      A[t,t′] = A[t,t′]
    end
  else
    for t in 1:1:T, t′ in 1:1:t
      A[t,t′] = A[t,t′]
    end
  end
  A
end

# struct RetardedGF <: GreenFunction end
# struct AdvancedGF <: GreenFunction end
# struct TimeOrderedGF <: GreenFunction end
# struct AntiTimeOrderedGF <: GreenFunction end

for g in (:LesserGF, :GreaterGF)
  for f in (:-, :conj, :real, :imag, :adjoint, :transpose, :inv)
      @eval (Base.$f)(A::$g) = $g(Base.$f(A.data))
  end

  # for f in (:tr)
  #     @eval (LinearAlgebra.$f)(A::$g) = LinearAlgebra.$f(A.data)
  # end

  for f in (:+, :-, :/, :\, :*)
    if f != :/  
        @eval (Base.$f)(A::Number, B::$g) = $g(Base.$f(A, B.data))
    end
    if f != :\
        @eval (Base.$f)(A::$g, B::Number) = $g(Base.$f(A.data, B))
    end
  end

  for f in (:+, :-, :*, :\, :/)
    # for g′ in (:LesserGF, :GreaterGF)
      @eval (Base.$f)(A::$g, B::$g) = $g(Base.$f(A.data, B.data))
    # end
  end

  @eval Base.:+(A::$g, B::$g...) = Base.+(A.data, getfield.(B,:data)...)
end

function Base.show(io::IO, x::LesserOrGreater)
  if get(io, :compact, false) || get(io, :typeinfo, nothing) == LesserOrGreater
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

function resize(A::LesserOrGreater, t::NTuple{2,Int})
  if eltype(A) <: AbstractArray
    newdata = fill(eltype(A)(undef,t...,size(first(A))...))
  else
    newdata = typeof(A)(zeros(eltype(A),t...,tail2(size(A))...))
  end

  T, T′ = min.(first2(size(A)), t)

  for t in 1:1:T
    for t′ in 1:1:T′
      newdata[t,t′] = A[t,t′]
    end
  end

  return newdata
end
