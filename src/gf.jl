abstract type GreenFunction end

# """
# Defined as
#     G^<(t,t') = -i < a^{\dagger}(t) a(t') > for t > t'
# """
struct LesserGF{T,S<:AbstractArray{<:T}} <: GreenFunction
    data::S

    function LesserGF{T,S}(data) where {T, S<:AbstractArray{<:T}}
        # Base.require_one_based_indexing(data)
        new{T,S}(copy(data))
    end
end

function LesserGF(A::AbstractArray)
    LinearAlgebra.checksquare(A)
    return lesserGF_type(typeof(A))(A)
end

function lesserGF_type(::Type{T}) where {S<:Number, T<:AbstractArray{S}}
    return LesserGF{S, T}
end
function lesserGF_type(::Type{T}) where {S<:AbstractArray, T<:AbstractArray{S}}
    return LesserGF{AbstractArray, T}
end

# """
# Defined as
#     G^>(t,t') = +i < a^{\dagger}(t') a(t) > for t <= t'
# """
struct GreaterGF{T,S<:AbstractArray{<:T}} <: GreenFunction
    data::S

    function GreaterGF{T,S}(data) where {T, S<:AbstractArray{<:T}}
        # Base.require_one_based_indexing(data)
        new{T,S}(copy(data))
    end
end

function GreaterGF(A::AbstractArray)
    LinearAlgebra.checksquare(A)
    return greaterGF_type(typeof(A))(A)
end

function greaterGF_type(::Type{T}) where {S<:Number, T<:AbstractArray{S}}
    return GreaterGF{S, T}
end
function greaterGF_type(::Type{T}) where {S<:AbstractArray, T<:AbstractArray{S}}
    return GreaterGF{AbstractArray, T}
end

const LesserOrGreater{T,S} = Union{LesserGF{T,S}, GreaterGF{T,S}}

Base.size(G::LesserOrGreater) = size(G.data)
Base.convert(T::Type{<:GreenFunction}, m::AbstractArray) = T(m)

@inline function Base.getindex(G::LesserOrGreater, i::Integer, j::Integer) where {N}
    # @boundscheck Base.checkbounds(G.data, i, j)
    @inbounds if i == j
        return getindex(G.data, i, j)
    elseif (G isa LesserGF) == (i > j)
        return getindex(G.data, i, j)
    else
        return -adjoint(getindex(G.data, j, i))
    end
end

@inline function Base.getindex(G::LesserOrGreater, i::Integer, j::Integer, I...)
    # @boundscheck Base.checkbounds(G.data, i, j)
    @inbounds if i == j
        return getindex(G.data, i, j, I...)
    elseif (G isa LesserGF) == (i > j)
        return getindex(G.data, i, j, I...)
    else
        return -adjoint(getindex(G.data, j, i, I...))
    end
end

function Base.setindex!(G::LesserOrGreater, v, i::Integer, j::Integer)
    @inbounds if i == j
        setindex!(G.data, v, i, j)
    elseif (G isa LesserGF) == (i > j)
        setindex!(G.data, v, i, j)
    else
        setindex!(G.data, -adjoint(v), j, i)
    # else 
    #     setindex!(G.data, v, i, j)
    #     setindex!(G.data, -adjoint(v), j, i)
    end
end

function Base.setindex!(G::LesserOrGreater, v, i::Integer, j::Integer, I...)
    @inbounds if i == j
        setindex!(G.data, v, i, j, I...)
    elseif (G isa LesserGF) == (i > j)
        setindex!(G.data, v, i, j, I...)
    else
        setindex!(G.data, -adjoint(v), j, i, I...)
    # else 
    #     setindex!(G.data, v, i, j, I...)
    #     setindex!(G.data, -adjoint(v), j, i, I...)
    end
end

# struct RetardedGF <: GreenFunction end
# struct AdvancedGF <: GreenFunction end
# struct TimeOrderedGF <: GreenFunction end
# struct AntiTimeOrderedGF <: GreenFunction end

