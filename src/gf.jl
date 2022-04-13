abstract type AbstractSymmetry end
abstract type AbstractGreenFunction{T,N} <: AbstractArray{T,N} end

"""
    Symmetrical

Defined as

``G(t,t') = G(t',t)^\\top``
"""
struct Symmetrical <: AbstractSymmetry end

"""
    SkewHermitian

Defined as

``G(t,t') = -G(t',t)^\\dagger``
"""
struct SkewHermitian <: AbstractSymmetry end

"""
    GreenFunction(g::AbstractArray, s::AbstractSymmetry)

A container interface for `g` with array indexing respecting some symmetry rule `s`.
Because of that, `g` must be square in its last 2 dimensions, which can be resized 
with [`resize!`](@ref).

The array `g` is not restricted to being contiguous. For example, `g` can have
`Matrix{T}`, `Array{T,4}`, `Matrix{SparseMatrixCSC{T}}`, etc as its type.

# Notes
The GreenFunction *does not* own `g`. Proper care must be taken when using multiple
GreenFunctions since using the same array will result in unexpected behaviour
```julia-repl
julia> data = zeros(2,2)
julia> g1 = GreenFunction(data, Symmetrical)
julia> g2 = GreenFunction(data, Symmetrical)
julia> g1[1,1] = 3
julia> @show g2[1,1]
julia> g1.data === g2.data # they share the same data
```

Indexing with less indices than the dimension of `g` results in a 
"take-all-to-the-left" indexing
```julia-repl
julia> gf[i,j] == gf[:,:,...,:,i,j]
julia> gf[i,j,k] == gf[:,:,...,:,i,j,k]
```

Custom symmetries can be implemented via multiple dispatch
```julia-repl
julia> struct MySymmetry <: KadanoffBaym.AbstractSymmetry end
julia> @inline KadanoffBaym.symmetry(::GreenFunction{T,N,A,MySymmetry}) where {T,N,A} = conj
```

# Examples
`GreenFunction` simply takes some data `g` and embeds the symmetry `s` in its indexing
```julia-repl
julia> time_dim = 3
julia> spin_dim = 2
julia> data = zeros(spin_dim, spin_dim, time_dim, time_dim)
julia> gf = GreenFunction(data, Symmetrical)
julia> gf[2,1] = rand(spin_dim, spin_dim)
julia> @show gf[1,2]
julia> @show KadanoffBaym.symmetry(gf)(gf[2,1])
```
"""
mutable struct GreenFunction{T,N,A,U<:AbstractSymmetry} <: AbstractGreenFunction{T,N}
  data::A
end

function GreenFunction(G::AbstractArray, U::Type{<:AbstractSymmetry})
  if U <: Union{Symmetrical,SkewHermitian}
    @assert ==(last2(size(G))...) "Time dimension ($(last2(size(G)))) must be a square"
  end
  return GreenFunction{eltype(G),ndims(G),typeof(G),U}(G)
end

@inline symmetry(::GreenFunction{T,N,A,Symmetrical}) where {T,N,A} = transpose
@inline symmetry(::GreenFunction{T,N,A,SkewHermitian}) where {T,N,A} = (-) ∘ adjoint

@inline Base.size(G::GreenFunction, I...) = size(G.data, I...)
@inline Base.length(G::GreenFunction) = length(G.data)
@inline Base.ndims(G::GreenFunction) = ndims(G.data)
@inline Base.axes(G::GreenFunction, d) = axes(G.data, d)

Base.copy(G::GreenFunction) = oftype(G, copy(G.data))
Base.eltype(::GreenFunction{T}) where {T} = T

@inline Base.getindex(::GreenFunction{T,N,A,U}, I) where {T,N,A,U} = error("Single indexing not allowed")
Base.@propagate_inbounds Base.getindex(G::GreenFunction{T,N,A,U}, I::Vararg{T1,N1}) where {T,N,A,U,T1,N1} = G.data[ntuple(i -> Colon(), N-N1)..., I...]

@inline Base.setindex!(::GreenFunction{T,N,A,U}, v, I) where {T,N,A,U} = error("Single indexing not allowed")
Base.@propagate_inbounds function Base.setindex!(G::GreenFunction{T,N,A,U}, v, I::Vararg{T1,N1}) where {T,N,A,U,T1,N1}
  ts = last2(I)
  jj = front2(I)

  if ==(ts...)
    G.data[ntuple(i -> Colon(), N-N1)..., jj..., ts...] = v
  else
    G.data[ntuple(i -> Colon(), N-N1)..., jj..., ts...] = v
    G.data[ntuple(i -> Colon(), N-N1)..., jj..., reverse(ts)...] = symmetry(G)(v)
  end
end

for g in (:GreenFunction,)
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

  @eval Base.:+(G::$g{T,N,A,U}, Gs::$g{T,N,A,U}...) where {T,N,A,U} = Base .+ (G.data, getfield.(Gs, :data)...)
end

function Base.show(io::IO, x::GreenFunction)
  #if get(io, :compact, false) || get(io, :typeinfo, nothing) == GreenFunction
  #  return Base.show_default(IOContext(io, :limit => true), x)
  #else
  #  # dump(IOContext(io, :limit => true), p, maxdepth=1)
  #  for field in fieldnames(typeof(x))
  #    if field === :data
  #      print(io, "\ndata: ")
  #      Base.show(io, MIME"text/plain"(), x.data)
  #    else
  #      Base.show(io, getfield(x, field))
  #    end
  #  end
  #end
  return show(io, x.data)
end

Base.resize!(A::GreenFunction, t::Int) = (A.data = _resize!(A.data, t); A)

function _resize!(a::Array{T,N}, t::Int) where {T,N}
  a′ = Array{T,N}(undef, front2(size(a))..., t, t)

  k = min(last(size(a)), t)
  for t in 1:k, t′ in 1:k
    a′[ntuple(i -> Colon(), N-2)..., t′, t] = a[ntuple(i -> Colon(), N-2)..., t′, t]
  end

  return a′
end

"""
    SkewHermitianArray

Provides a different (but not necessarily more efficient) data storage for 
elastic skew-Hermitian data
"""
struct SkewHermitianArray{T,N} <: AbstractGreenFunction{T,N}
  data::Vector{Vector{T}}

  function SkewHermitianArray(a::T) where {T<:Union{Number,AbstractArray}}
    return new{T,2 + ndims(a)}([[a]])
  end
end

@inline Base.size(a::SkewHermitianArray{<:Number}) = (length(a.data), length(a.data))
@inline Base.size(a::SkewHermitianArray{<:AbstractArray}) = (size(a.data[1][1])..., length(a.data), length(a.data))

Base.getindex(::SkewHermitianArray, I) = error("Single indexing not allowed")
@inline function Base.getindex(a::SkewHermitianArray{T,N}, i::Int, j::Int) where {T,N}
  return (i >= j) ? a.data[i - j + 1][j] : -adjoint(a.data[j - i + 1][i])
end

@inline function Base.setindex!(a::SkewHermitianArray, v, i::Int, j::Int)
  if i >= j
    a.data[i - j + 1][j] = v
  else
    a.data[j - i + 1][i] = -adjoint(v)
  end
end

Base.resize!(A::SkewHermitianArray, t::Int) = (_resize!(A, t); A)

function _resize!(a::SkewHermitianArray{T}, t::Int) where {T}
  l = length(a)

  resize!(a.data, t)

  for k in 1:min(l, t)
    resize!(a.data[k], t - k + 1)
  end
  for k in (min(l, t) + 1):t
    a.data[k] = Array{T}(undef, t - k + 1)
  end
end
