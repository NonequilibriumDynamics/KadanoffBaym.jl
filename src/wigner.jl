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
  # LinearAlgebra.checksquare(x)

  Nt = size(x, 1)
  @assert length(ts) == Nt

  @assert let x = diff(ts); all(z -> z ≈ x[1], x) end "`ts` is not equidistant"

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

  x_W = fftshift(x_W, 1)
  τs = ts - reverse(ts)
  τs = τs .- (isodd(Nt) ? 0.0 : 0.5(τs[2] - τs[1]))

  if !fourier
    return x_W, (τs, ts)
  else
    ωs = ft(τs, τs)[1]
    x_Ŵ = mapslices(x -> ft(τs, x; inverse = false)[2], x_W; dims=1)

    return x_Ŵ, (ωs, ts)
  end
end

""" 
    ft(xs, ys; inverse = false)

Fourier transform of the points `(xs, ys)`:

if `inverse`

``ŷ(t) = ∫dω/(2π) y(ω) e^{- i ω t}``

else

``ŷ(ω) = ∫dt y(t) e^{+ i ω t}``

Returns a tuple of the Fourier transformed points `(x̂s, ŷs)`
"""
function ft(xs, ys; inverse::Bool = false)
  @assert issorted(xs)

  L = length(xs)
  dx = xs[2] - xs[1]

  # Because the FFT calculates the transform as
  #   ỹ_k = \sum_j e^{±2pi i j k/n} y_j, from j=0 to j=n-1,
  # we need to transform this into the time and frequency units, 
  # which ends up scaling the frequencies `x̂s` by (2pi / dx).
  x̂s = fftfreq(L, 2π / dx)

  # The resulting Fourier transform also picks up a phase
  ℯⁱᵠ = (inverse ? 1 / (2π) : 1.0) * dx * exp.((inverse ? -1.0 : 1.0) * im * xs[1] .* x̂s) 

  # FFT
  ŷs = ℯⁱᵠ .* (inverse ? fft : bfft)(ys)

  return fftshift(x̂s), fftshift(ŷs)
end
