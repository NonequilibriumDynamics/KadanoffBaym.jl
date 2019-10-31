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

Base.eltype(::LesserOrGreater{T,S}) where {T,S} = eltype(S)
Base.eltype(::Type{<:LesserOrGreater{T,S}}) where {T,S} = eltype(S)
Base.convert(T::Type{<:GreenFunction}, m::AbstractArray) = T(m)

Base.size(A::LesserOrGreater) = size(A.data)
Base.length(A::LesserOrGreater) = length(A.data)

Base.firstindex(A::LesserOrGreater) = firstindex(A.data)
Base.lastindex(A::LesserOrGreater) = lastindex(A.data)

Base.getindex(A::LesserOrGreater, I...) = Base.getindex(A.data, I...) #get(A.data, I, zero(eltype(A)))
Base.setindex!(A::LesserOrGreater, v, I...) = _setindex!(A, v, last2(I), front2(I)...)

_setindex!(A::LesserGF, v, L::NTuple{2, Int64}, F...) = begin @assert <=(L...) "t>t′"; __setindex!(A, v, L, F...) end
_setindex!(A::GreaterGF, v, L::NTuple{2, Int64}, F...) = begin @assert >=(L...) "t<t′"; __setindex!(A, v, L, F...) end
_setindex!(A::LesserOrGreater, v, L::Tuple{T, U}, F...) where {T, U} = __setindex!(A, v, L, F...)

function __setindex!(A::LesserOrGreater, v, L::Tuple{T,U}, F...) where {T,U}
  if ==(L...)
    setindex!(A.data, v, F..., L...)
  else 
    setindex!(A.data, v, F..., L...)
    setindex!(A.data, -adjoint(v), F..., reverse(L)...)
  end
end

Base.iterate(A::LesserOrGreater, state=1) = iterate(A.data, state)

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
        print(io, "data: ")
        Base.show(io, MIME"text/plain"(), x.data)
      else
        Base.show(io, getfield(x, field))
      end
    end
  end
end
