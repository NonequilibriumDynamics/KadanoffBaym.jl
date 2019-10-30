# Return a `Tuple` consisting of all but the 2 last components of `t`.
function front2(t::Tuple)
  Base.@_inline_meta
  _front2(t...)
end
_front2() = throw(ArgumentError("Cannot call front2 on an empty tuple."))
_front2(v) = throw(ArgumentError("Cannot call front2 on 1-element tuple."))
_front2(v1,v2) = ()
function _front2(v, t...)
  @Base._inline_meta
  (v, _front2(t...)...)
end

# Return a `Tuple` consisting of all but the 2 first components of `t`.
function tail2(t::Tuple)
  Base.@_inline_meta
  _tail2(t...)
end
_tail2() = throw(ArgumentError("Cannot call tail2 on an empty tuple."))
_tail2(v) = throw(ArgumentError("Cannot call tail2 on 1-element tuple."))
_tail2(v1,v2) = ()
_tail2(v1, v2, rest...) = rest

# Return a `Tuple` consisting of the last 2 components of `t`.
function last2(t::Tuple)
  Base.@_inline_meta
  _last2(t...)
end
_last2() = throw(ArgumentError("Cannot call last2 on an empty tuple."))
_last2(v) = throw(ArgumentError("Cannot call last2 on 1-element tuple."))
_last2(v1,v2) = (v1, v2)
function _last2(v, t...)
  @Base._inline_meta
  _last2(t...)
end
