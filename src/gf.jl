abstract type AbstractSymmetry end
abstract type AbstractGreenFunction{T,N} <: AbstractArray{T,N} end

"""
    Symmetrical

Exchange the time arguments and transpose the stored value:

``G(t,t') = G(t',t)^\\top``

For scalars this only exchanges the times; for matrices it also exchanges the
internal indices, ``G_{ab}(t,t') = G_{ba}(t',t)``.
"""
struct Symmetrical <: AbstractSymmetry end

"""
    SkewHermitian

Exchange the time arguments and take the negative adjoint of the stored value:

``G(t,t') = -G(t',t)^\\dagger``

For scalars the adjoint is complex conjugation; for matrices,
``G_{ab}(t,t') = -\\overline{G_{ba}(t',t)}``.
"""
struct SkewHermitian <: AbstractSymmetry end

"""
    OnePoint

A single-time field, with time on the last axis and no time-exchange symmetry.
"""
struct OnePoint <: AbstractSymmetry end

@inline symmetry(::Type{<:AbstractSymmetry}) = error("Not defined")
@inline symmetry(::Type{Symmetrical}) = transpose
@inline symmetry(::Type{SkewHermitian}) = (-) ∘ adjoint

"""
    GreenFunction(g::AbstractArray, s::Type{<:AbstractSymmetry})

Wrap `g` with a value symmetry under exchange of its last two time indices.
`Symmetrical` and `SkewHermitian` apply transpose and negative adjoint to the
entire value block, respectively; their usual values are scalars or square
matrices. Other value shapes need an explicit shape-preserving involution.
`OnePoint` instead has only one time axis, the last axis. Dense `Array` storage
can be resized with `resize!`.

The array `g` is not restricted to being contiguous. For example, `g` can have
`Matrix{T}`, `Array{T,4}`, `Matrix{SparseMatrixCSC{T}}`, etc as its type.

# Notes
The full supplied array is stored. Two-time storage includes both time triangles;
construction does not validate or project its values. For two-time fields,
callers must supply compatible time axes and the declared symmetry, including
diagonal values satisfying `v == symmetry(s)(v)`. Diagonal assignments are not
projected. Off-diagonal assignment mirrors the value after conversion to the
storage type. Mutating `.data` or a returned mutable block bypasses this mirroring.

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

`end` follows the underlying array axes. For matrix-valued storage, access the
last time block with `n = size(gf, ndims(gf)); gf[n, n]`.

Whole-container `transpose` and `adjoint`, where defined by Base for vectors
and matrices, return ordinary array views. Use `transpose(gf[i,j])` or
`adjoint(gf[i,j])` to transform a physical value block.

In order to ensure a correct behaviour of the symmetries, `setindex!` is only
defined at the level of time coordinates
```julia-repl
julia> gf[i, j] = a # is equivalent to gf[..., i, j] = a[...]
```

The imaginary part of a `SkewHermitian` GreenFunction is `Symmetrical`.
Multiplication or division of a `SkewHermitian` GreenFunction by a non-real
scalar returns an ordinary array, since the result generally has neither
of these symmetries. Real scalars preserve its symmetry; complex scalars
with zero imaginary part also return an ordinary array.

Custom symmetries can be implemented via a shape-preserving involution (applying
it twice recovers the value). Symmetry-preserving arithmetic overloads are
provided for the built-in symmetries; other operations follow Base array behavior.
```julia-repl
julia> struct MySymmetry <: KadanoffBaym.AbstractSymmetry end
julia> @inline KadanoffBaym.symmetry(::Type{MySymmetry}) = conj
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
julia> @show KadanoffBaym.symmetry(Symmetrical)(gf[2,1])
```
"""
mutable struct GreenFunction{T,N,A,U<:AbstractSymmetry} <: AbstractGreenFunction{T,N}
  data::A
end

function GreenFunction(G::AbstractArray, U::Type{<:AbstractSymmetry})
  if U <: Union{Symmetrical,SkewHermitian}
    @assert ==(last2(size(G)...)...) "Time dimension ($(last2(size(G)...))) must be a square"
  end
  return GreenFunction{eltype(G),ndims(G),typeof(G),U}(G)
end

@inline Base.size(G::GreenFunction, I...) = size(G.data, I...)
@inline Base.axes(G::GreenFunction, I...) = axes(G.data, I...)

Base.copy(G::GreenFunction{T,N,A,U}) where {T,N,A,U} = GreenFunction(copy(G.data), U)

@inline Base.getindex(::GreenFunction{T,N,A,U}, I) where {T,N,A,U} = error("Single indexing not allowed")
Base.@propagate_inbounds Base.getindex(G::GreenFunction{T,N}, I::CartesianIndex{N}) where {T,N} = G.data[I]
Base.@propagate_inbounds Base.getindex(G::GreenFunction{T,N,A,U}, I::Vararg{Any,N1}) where {T,N,A,U,N1} = G.data[ntuple(i -> Colon(), N-N1)..., I...]

Base.@propagate_inbounds function Base.setindex!(G::GreenFunction{T,N,A,U}, v, i1::Int, i2::Int) where {T,N,A,U}
  v = Base.unalias(G.data, v)
  G.data[ntuple(i -> Colon(), Val(N - 2))..., i1, i2] = v

  if i1 != i2
    @views G.data[ntuple(i -> Colon(), Val(N - 2))..., i2, i1] = symmetry(U)(G.data[ntuple(i -> Colon(), Val(N - 2))..., i1, i2])
  end
end

Base.@propagate_inbounds function Base.setindex!(G::GreenFunction{T,4,A,U}, v::AbstractMatrix, i1::Int, i2::Int) where {T<:Number,A,U<:Union{Symmetrical,SkewHermitian}}
  v = Base.unalias(G.data, v)
  Base.setindex_shape_check(v, size(G.data, 1), size(G.data, 2))
  stored = @view G.data[:, :, i1, i2]
  copyto!(stored, v)
  if i1 != i2
    mirror = @view G.data[:, :, i2, i1]
    axes(mirror) == reverse(axes(stored)) || throw(DimensionMismatch("time-exchanged blocks must have transposed axes"))
    @inbounds for a in axes(stored, 1), b in axes(stored, 2)
      mirror[b, a] = symmetry(U)(stored[a, b])
    end
  end
end

Base.@propagate_inbounds Base.getindex(G::GreenFunction{T,N,A,OnePoint}, I::Vararg{Any,N1}) where {T,N,A,N1} = G.data[ntuple(i -> Colon(), N-N1)..., I...]
Base.@propagate_inbounds Base.getindex(G::GreenFunction{T,N,A,OnePoint}, I::CartesianIndex{N}) where {T,N,A} = G.data[I]

Base.@propagate_inbounds function Base.setindex!(G::GreenFunction{T,N,A,OnePoint}, v, I::Vararg{Any,N1}) where {T,N,A,N1}
  G.data[ntuple(i -> Colon(), N-N1)..., I...] = v
end

for f in (:-, :conj, :real, :imag, :zero)
  @eval Base.$f(G::GreenFunction{T,N,A,U}) where {T,N,A,U<:Union{Symmetrical,SkewHermitian,OnePoint}} = GreenFunction(Base.$f(G.data), U)
end

for f in (:/, :\, :*)
  if f != :/
    @eval Base.$f(a::Number, G::GreenFunction{T,N,A,U}) where {T,N,A,U<:Union{Symmetrical,SkewHermitian,OnePoint}} = GreenFunction(Base.$f(a, G.data), U)
  end
  if f != :\
    @eval Base.$f(G::GreenFunction{T,N,A,U}, b::Number) where {T,N,A,U<:Union{Symmetrical,SkewHermitian,OnePoint}} = GreenFunction(Base.$f(G.data, b), U)
  end
end

Base.imag(G::GreenFunction{T,N,A,SkewHermitian}) where {T,N,A} = GreenFunction(imag(G.data), Symmetrical)

for f in (:*, :\)
  @eval Base.$f(a::Number, G::GreenFunction{T,N,A,SkewHermitian}) where {T,N,A} = Base.$f(a, G.data)
  @eval Base.$f(a::Real, G::GreenFunction{T,N,A,SkewHermitian}) where {T,N,A} = GreenFunction(Base.$f(a, G.data), SkewHermitian)
end
for f in (:*, :/)
  @eval Base.$f(G::GreenFunction{T,N,A,SkewHermitian}, a::Number) where {T,N,A} = Base.$f(G.data, a)
  @eval Base.$f(G::GreenFunction{T,N,A,SkewHermitian}, a::Real) where {T,N,A} = GreenFunction(Base.$f(G.data, a), SkewHermitian)
end

Base.show(io::IO, x::GreenFunction) = show(io, x.data)

function Base.resize!(G::GreenFunction, t::Int)
  last2(size(G)...) == (t, t) && return G
  G.data = _resize!(G.data, t)
  return G
end

function _resize!(a::Array{T,N}, t::Int) where {T,N}
  a′ = Array{T,N}(undef, front2(size(a)...)..., t, t)

  k = min(last(size(a)), t)
  @views a′[ntuple(i -> Colon(), N-2)..., 1:k, 1:k] .= a[ntuple(i -> Colon(), N-2)..., 1:k, 1:k]

  return a′
end

function Base.resize!(g::GreenFunction{T,N,A,OnePoint}, t::Int) where {T,N,A}
  last(size(g)) == t && return g
  a′ = Array{T,N}(undef, Base.front(size(g))..., t)

  k = min(last(size(g)), t)
  @views a′[ntuple(i -> Colon(), N-1)..., 1:k] .= g.data[ntuple(i -> Colon(), N-1)..., 1:k]

  g.data = a′
  return g
end

"""
    SkewHermitianArray

Store one triangle of a resizable skew-Hermitian time matrix. Its elements are
scalar or matrix values: `size(G) == (nt, nt)`, while `size(G[i,j])` describes
the internal value shape. Diagonal values must be skew-Hermitian.
"""
struct SkewHermitianArray{T} <: AbstractGreenFunction{T,2}
  data::Vector{Vector{T}}

  function SkewHermitianArray(a::T) where {T<:Union{Number,AbstractArray}}
    return new{T}([[a]])
  end
end

@inline Base.size(a::SkewHermitianArray) = (length(a.data), length(a.data))

@inline function Base.getindex(a::SkewHermitianArray{T}, i::Int, j::Int) where {T}
  return (i >= j) ? a.data[i - j + 1][j] : convert(T, .-adjoint(a.data[j - i + 1][i]))
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
  l = length(a.data)
  l == t && return

  resize!(a.data, t)

  for k in 1:min(l, t)
    resize!(a.data[k], t - k + 1)
  end
  for k in (min(l, t) + 1):t
    a.data[k] = Array{T}(undef, t - k + 1)
  end
end
