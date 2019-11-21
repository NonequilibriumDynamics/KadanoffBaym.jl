# Return a `Tuple` consisting of all but the last 2 components of `t`.
@inline front2(t::Tuple) = _front2(t...)
_front2() = throw(ArgumentError("Cannot call front2 on an empty tuple."))
_front2(v) = throw(ArgumentError("Cannot call front2 on 1-element tuple."))
_front2(v1,v2) = ()
@inline _front2(v, t...) = (v, _front2(t...)...)

# Return a `Tuple` consisting of all but the first 2 components of `t`.
@inline tail2(t::Tuple) = _tail2(t...)
_tail2() = throw(ArgumentError("Cannot call tail2 on an empty tuple."))
_tail2(v) = throw(ArgumentError("Cannot call tail2 on 1-element tuple."))
_tail2(v1,v2) = ()
_tail2(v1, v2, rest...) = rest

# Return a `Tuple` consisting of the first 2 components of `t`.
@inline first2(t::Tuple) = _first2(t...)
_first2() = throw(ArgumentError("Cannot call first2 on an empty tuple."))
_first2(v) = throw(ArgumentError("Cannot call first2 on 1-element tuple."))
@inline _first2(v1, v2, rest...) = (v1, v2)

# Return a `Tuple` consisting of the last 2 components of `t`.
@inline last2(t::Tuple) = _last2(t...)
_last2() = throw(ArgumentError("Cannot call last2 on an empty tuple."))
_last2(v) = throw(ArgumentError("Cannot call last2 on 1-element tuple."))
_last2(v1,v2) = (v1, v2)
@inline _last2(v, t...) = _last2(t...)

# Return a `Tuple` consisting of a tuple of the first 2 and and a tuple of the last components of `t`.
@inline front2_last(t::Tuple) = _front2_last(t...)
_front2_last() = throw(ArgumentError("Cannot call front2_last on an empty tuple."))
_front2_last(v) = throw(ArgumentError("Cannot call front2_last on 1-element tuple."))
_front2_last(v1,v2, rest...) = ((v1,v2), rest)

# Return a `Tuple` consisting of a tuple of the first and a tuple of the last 2 components of `t`.
@inline front_last2(t::Tuple) = _front_last2(t...)
_front_last2() = throw(ArgumentError("Cannot call front_last2 on an empty tuple."))
_front_last2(v) = throw(ArgumentError("Cannot call front_last2 on 1-element tuple."))
_front_last2(v1,v2) = ((), (v1, v2))
_front_last2(v, t...) = tuplejoin((v,), _front_last2(t...)...)

@inline tuplejoin(x::Tuple) = x
@inline tuplejoin(x::Tuple, y::Tuple) = (x..., y...)
@inline tuplejoin(x::Tuple, y::Tuple, z::Tuple) = (tuplejoin(x,y), z)