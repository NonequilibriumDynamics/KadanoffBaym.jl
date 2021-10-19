"""
    wigner_transform(x::AbstractMatrix; ts=1:size(x,1), fourier=true)

Wigner-Ville transformation

``x_W(ω, T) = i ∫dt x(T + t/2, T - t/2) e^{+i ω t}``

or

``x_W(τ, T) = x(T + t/2, T - t/2)``

of a 2-point function `x`. Returns a tuple of `x_W` and the corresponding
axes (`ω, T`) or (`τ`, `T`), depending on the `fourier` keyword.

The motivation for the Wigner transformation is that, given an autocorrelation
function `x`, it reduces to the spectral density function at all times `T` for 
stationary processes, yet it is fully equivalent to the non-stationary 
autocorrelation function. Therefore, the Wigner (distribution) function tells 
us, roughly, how the spectral density changes in time.

# Optional keyword parameters
  - `ts::AbstractVector`: Time grid for `x`. Defaults to a `UnitRange`.
  - `fourier::Bool`: Whether to Fourier transform. Defaults to `true`.

# Notes
The algorithm only works when `ts` – and consequently `x` – is equidistant.

# References
<https://en.wikipedia.org/wiki/Wigner_distribution_function>

<http://tftb.nongnu.org>
"""
function wigner_transform(x::AbstractMatrix; ts=1:size(x, 1), fourier=true)
  LinearAlgebra.checksquare(x)

  Nt = size(x, 1)
  @assert length(ts) == Nt

  # NOTE: code for non-equidistant time-grids
  # The resulting transformation will be in an equidistant grid
  # ts_equidistant = range(ts[1], stop=ts[end], length=length(ts))
  # tᵢ = searchsortedlast.(Ref(ts_equidistant), ts[T]) # map to an equidistant index

  # Change of basis x(t1, t2) → x_W(t1 - t2, (t1 + t2)/2)
  x_W = zero(x)

  for T in 1:Nt
    # For a certain T ≡ (t1 + t2)/2, τ ≡ (t1 - t2) can be at most τ_max
    τ_max = minimum([T - 1, Nt - T, Nt ÷ 2 - 1])

    τs = (-τ_max):τ_max
    is = 1 .+ rem.(Nt .+ τs, Nt)

    for (i, τᵢ) in zip(is, τs)
      x_W[i, T] = x[T + τᵢ, T - τᵢ]
    end

    τ = Nt ÷ 2
    if T <= Nt - τ && T >= τ + 1
      x_W[τ + 1, T] = 0.5 * (x[T + τ, T - τ] + x[T - τ, T + τ])
    end
  end

  x_W = circshift(x_W, (Nt ÷ 2, 0))
  τs = ts - reverse(ts)

  if !fourier
    return x_W, (τs, ts)
  else
    ωs = ft(τs, τs)[1]
    x_W̃ = mapslices(x -> ft(τs, x; mode=+1)[2], x_W; dims=1)
    
    return x_W̃, (ωs, ts)
  end
end

"""Interpolates `x` on an equidistant mesh with the same boundaries and length as `ts` and calls [`wigner_transform`](@ref)"""
function wigner_transform_itp(x::AbstractMatrix, ts::Vector; fourier=true)
  ts_lin = range(first(ts), last(ts); length=length(ts))
  itp = interpolate((ts, ts), x, Gridded(Linear()))
  return wigner_transform([itp(t1, t2) for t1 in ts_lin, t2 in ts_lin]; ts=ts_lin, fourier=fourier)
end

""" Fourier transform """
function ft(xs, ys; mode = +1)
  @assert issorted(xs)

  L = length(xs)
  dx = xs[2] - xs[1]
  
  # FFT
  ŷs = (mode == -1 ? fft(ys) : ifft(ys))
  
  # Because the FFT calculates the transform as
  #   ỹ_k = \sum_j e^{±2pi i j k/n} y_j, from j=0 to j=n-1,
  # we need to transform this into the time and frequency units, 
  # which ends up scaling the frequencies `x̂s` by (2pi / dx).
  x̂s = fftfreq(L, 2π / dx)

  # The resulting Fourier transform also picks up a phase
  ℯⁱᵠ = (mode == -1 ? 1 : L) * dx * exp.(mode * 1.0im * xs[1] .* x̂s) 

  return circshift(x̂s, L ÷ 2), circshift(ℯⁱᵠ .* ŷs, L ÷ 2)
end
