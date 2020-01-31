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


struct GreenFunction{T, S<: AbstractArray{<:T}, U}
  data::S

  function GreenFunction{T,S,U}(data) where {T, S<:AbstractArray{<:T}, U}
    new{T,S,U}(recursivecopy(data))
  end
end

function GreenFunction(A::AbstractArray, U::Type{<:GreenFunctionType})
  LinearAlgebra.checksquare(A)
  return GF_type(typeof(A), U)(A)
end

function GF_type(::Type{T}, U) where {S<:Number, T<:AbstractArray{S}}
  GreenFunction{S, T, U}
end

function GF_type(::Type{T}, U) where {S<:AbstractArray, T<:AbstractArray{S}}
  return GreenFunction{AbstractArray, T, U}
end

Base.eltype(::GreenFunction{T,S,U}) where {T,S,U} = eltype(S)
Base.convert(T::Type{<:GreenFunction}, m::AbstractArray) = T(m)

Base.size(A::GreenFunction) = size(A.data)
Base.length(A::GreenFunction) = length(A.data)

Base.firstindex(A::GreenFunction) = firstindex(A.data)
Base.lastindex(A::GreenFunction) = lastindex(A.data)

Base.getindex(A::GreenFunction, I::Int64) = error("Single indexing not allowed")
Base.getindex(A::GreenFunction, F::Vararg{Union{Int64, Colon}, 2}) = Base.getindex(A.data, F..., ..)
Base.getindex(A::GreenFunction, I...) = Base.getindex(A.data, I...)

Base.setindex!(A::GreenFunction, I::Int64) = error("Single indexing not allowed")
Base.setindex!(A::GreenFunction, v, F::Vararg{Union{Int64, Colon}, 2}) = __setindex!(A, v, F, (..,))
Base.setindex!(A::GreenFunction, v, I...) = _setindex!(A, v, front2_last(I)...)

# _setindex!(A::LesserGF, v, L::NTuple{2, Int64}, F...) = begin @assert <=(L...) "t>t′"; __setindex!(A, v, L, F...) end
# _setindex!(A::GreaterGF, v, L::NTuple{2, Int64}, F...) = begin @assert >=(L...) "t<t′"; __setindex!(A, v, L, F...) end
_setindex!(A::GreenFunction, v, F::NTuple{2, Union{Int64, Colon}}, L::Tuple) = __setindex!(A, v, F, L)

@inline function __setindex!(A::GreenFunction{T,S,<:Union{Greater,Lesser}}, v, F::Tuple{F1,F2}, L::Tuple) where {T,S,F1,F2}
  if ==(F...)
    setindex!(A.data, v, F..., L...)
  else 
    setindex!(A.data, v, F..., L...)
    setindex!(A.data, -adjoint(v), reverse(F)..., L...)
  end
end

Base.copy(A::GreenFunction) = typeof(A)(recursivecopy(A.data))

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

function symmetrize!(A::GreenFunction{I,S,U}) where {I,S,U}
  T, T′ = first2(size(A))
  if U === Lesser
    for t′ in 1:1:T′, t in 1:1:t′
      A[t,t′] = A[t,t′]
    end
  elseif U === Greater
    for t in 1:1:T, t′ in 1:1:t
      A[t,t′] = A[t,t′]
    end
  else
    error("Type not supported")
  end
  A
end

function resize(A::GreenFunction, t::NTuple{2,Int})
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
