# Return a `Tuple` consisting of all but the 2 last components of `t`.
@inline front2(t::Tuple) = _front2(t...)
_front2() = throw(ArgumentError("Cannot call front2 on an empty tuple."))
_front2(v) = throw(ArgumentError("Cannot call front2 on 1-element tuple."))
_front2(v1,v2) = ()
@inline _front2(v, t...) = (v, _front2(t...)...)

# Return a `Tuple` consisting of all but the 2 first components of `t`.
@inline tail2(t::Tuple) = _tail2(t...)
_tail2() = throw(ArgumentError("Cannot call tail2 on an empty tuple."))
_tail2(v) = throw(ArgumentError("Cannot call tail2 on 1-element tuple."))
_tail2(v1,v2) = ()
_tail2(v1, v2, rest...) = rest

# Return a `Tuple` consisting of the last 2 components of `t`.
@inline last2(t::Tuple) = _last2(t...)
_last2() = throw(ArgumentError("Cannot call last2 on an empty tuple."))
_last2(v) = throw(ArgumentError("Cannot call last2 on 1-element tuple."))
_last2(v1,v2) = (v1, v2)
@inline _last2(v, t...) = _last2(t...)
