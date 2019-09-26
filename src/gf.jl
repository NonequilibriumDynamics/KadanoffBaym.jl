using LinearAlgebra

abstract type GreenFunction end

# """
# Defined as
#     G^<(t,t') = -i < a^{\dagger}(t) a(t') > for t > t'
# """
struct LesserGF{T,S<:AbstractMatrix{<:T}} <: GreenFunction
    data::S

    function LesserGF{T,S}(data) where {T, S<:AbstractMatrix{<:T}}
        new{T,S}(copy(data))
    end
end

function LesserGF(A::AbstractMatrix)
    LinearAlgebra.checksquare(A)
    return lesserGF_type(typeof(A))(A)
end

function lesserGF_type(::Type{T}) where {S<:Number, T<:AbstractMatrix{S}}
    return LesserGF{S, T}
end
function lesserGF_type(::Type{T}) where {S<:AbstractMatrix, T<:AbstractMatrix{S}}
    return LesserGF{AbstractMatrix, T}
end

# """
# Defined as
#     G^>(t,t') = +i < a^{\dagger}(t') a(t) > for t <= t'
# """
struct GreaterGF{T,S<:AbstractMatrix{<:T}} <: GreenFunction
    data::S

    function GreaterGF{T,S}(data) where {T, S<:AbstractMatrix{<:T}}
        new{T,S}(copy(data))
    end
end

function GreaterGF(A::AbstractMatrix)
    LinearAlgebra.checksquare(A)
    return greaterGF_type(typeof(A))(A)
end

function greaterGF_type(::Type{T}) where {S<:Number, T<:AbstractMatrix{S}}
    return GreaterGF{S, T}
end
function greaterGF_type(::Type{T}) where {S<:AbstractMatrix, T<:AbstractMatrix{S}}
    return GreaterGF{AbstractMatrix, T}
end

const LesserOrGreater{T,S} = Union{LesserGF{T,S}, GreaterGF{T,S}}

size(G::LesserOrGreater) = size(G.data)

@inline function Base.getindex(G::LesserOrGreater, i::Integer, j::Integer)
    # @boundscheck Base.checkbounds(G.data, i, j)
    @inbounds if i == j
        return G.data[i, j]
    elseif (G isa LesserGF) == (i > j)
        return G.data[i, j]
    else
        return -adjoint(G.data[j, i])
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

# struct RetardedGF <: GreenFunction end
# struct AdvancedGF <: GreenFunction end
# struct TimeOrderedGF <: GreenFunction end
# struct AntiTimeOrderedGF <: GreenFunction end

