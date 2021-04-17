# Return a `Tuple` consisting of all but the last 2 components of `t`.
@inline front2(t::Tuple) = _front2(t...)
_front2() = throw(ArgumentError("Cannot call front2 on an empty tuple."))
_front2(v) = throw(ArgumentError("Cannot call front2 on 1-element tuple."))
@inline _front2(v1,v2) = ()
@inline _front2(v, t...) = (v, _front2(t...)...)

# Return a `Tuple` consisting of the last 2 components of `t`.
@inline last2(t::Tuple) = _last2(t...)
_last2() = throw(ArgumentError("Cannot call last2 on an empty tuple."))
_last2(v) = throw(ArgumentError("Cannot call last2 on 1-element tuple."))
@inline _last2(v1,v2) = (v1, v2)
@inline _last2(v, t...) = _last2(t...)

# From https://github.com/JuliaLang/julia/pull/33515
function unzip(itrs)
    n = Base.haslength(itrs) ? length(itrs) : nothing
    outer = iterate(itrs)
    outer === nothing && return ()
    vals, state = outer
    vecs = ntuple(length(vals)) do i
        x = vals[i]
        v = Vector{typeof(x)}(undef, something(n, 1))
        @inbounds v[1] = x
        return v
    end
    unzip_rest(vecs, typeof(vals), n isa Int ? 1 : nothing, itrs, state)
end

function unzip_rest(vecs, eltypes, i, itrs, state)
    while true
        i isa Int && (i += 1)
        outer = iterate(itrs, state)
        outer === nothing && return vecs
        itr, state = outer
        vals = Tuple(itr)
        if vals isa eltypes
            for (v, x) in zip(vecs, vals)
                if i isa Int
                    @inbounds v[i] = x
                else
                    push!(v, x)
                end
            end
        else
            vecs′ = map(vecs, vals) do v, x
                T = Base.promote_typejoin(eltype(v), typeof(x))
                v′ = Vector{T}(undef, length(v) + !(i isa Int))
                copyto!(v′, v)
                @inbounds v′[something(i, end)] = x
                return v′
            end
            eltypes′ = Tuple{map(eltype, vecs′)...}
            return unzip_rest(Tuple(vecs′), eltypes′, i, itrs, state)
        end
    end
end