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
