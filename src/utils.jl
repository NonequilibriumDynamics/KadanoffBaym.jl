# Starting Step Size: Section II.4
function initial_step(f0, u0, atol, rtol, norm=ODE_DEFAULT_NORM)
  sc = atol + rtol * norm(u0, nothing)
  d0 = norm(u0 ./ sc, nothing)
  d1 = norm(f0 ./ sc, nothing)

  return dt0 = min(d0, d1) < 1e-5 ? 1e-6 : 1e-2 * d0 / d1

  # f1 = f(muladd(dt0, f0, u0), t0+dt0)
  # d2 = norm((f1 - f0) ./ sc) / dt0

  # dt1 = max(d1,d2) <= 1e-15 ? max(1e-6, 1e-3 * dt0) : (1e-2 / max(d1,d2))^(1/(k+1))
  # return min(dt1, 1e2 * dt0)
end

# Returns a `Tuple` consisting of all but the last 2 components of `t`.
@inline front2(t::Tuple) = _front2(t...)
_front2() = throw(ArgumentError("Cannot call front2 on an empty tuple."))
_front2(v) = throw(ArgumentError("Cannot call front2 on 1-element tuple."))
@inline _front2(v1, v2) = ()
@inline _front2(v, t...) = (v, _front2(t...)...)

# Returns a `Tuple` consisting of the last 2 components of `t`.
@inline last2(t::Tuple) = _last2(t...)
_last2() = throw(ArgumentError("Cannot call last2 on an empty tuple."))
_last2(v) = throw(ArgumentError("Cannot call last2 on 1-element tuple."))
@inline _last2(v1, v2) = (v1, v2)
@inline _last2(v, t...) = _last2(t...)
