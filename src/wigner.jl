"""
    wigner_transform(x::AbstractMatrix; ts=1:size(x,1), fourier=true)

Wigner-Ville transformation

``x_W(ω, T) = ∫dτ x(T + τ/2, T - τ/2) e^{+i ω τ}``

or

``x_W(τ, T) = x(T + τ/2, T - τ/2)``

of a 2-point function `x`. Returns a tuple of `x_W` and the corresponding
axes (`ω, T`) or (`τ`, `T`), depending on the `fourier` keyword.

The Wigner distribution describes how the spectral content of an autocorrelation
function changes with its center time `T`.

# Optional keyword parameters
  - `ts::AbstractVector`: Time grid for `x`. Defaults to a `UnitRange`.
  - `fourier::Bool`: Whether to Fourier transform. Defaults to `true`.

# Notes
The algorithm only works when `ts` – and consequently `x` – is equidistant.
Fourier transformation requires at least two distinct samples and positive finite
time spacing. Ranges use their declared step rather than differences of rounded
sample values.

The returned array has the same shape as `x`, with center times `T = ts` and
lag spacing twice the time-grid spacing. Thus only input pairs with even `i+j`
contribute; the other pairs have center times between grid points and are omitted.
This discrete representation is not invertible for an arbitrary input matrix.

# References
<https://en.wikipedia.org/wiki/Wigner_distribution_function>

<http://tftb.nongnu.org>
"""
function wigner_transform(x::AbstractMatrix; ts=1:size(x, 1), fourier=true)
  LinearAlgebra.checksquare(x)

  Nt = size(x, 1)
  @assert length(ts) == Nt
  @assert Nt >= (fourier ? 2 : 1) "Insufficient time samples"
  Δt = isone(Nt) ? zero(eltype(ts)) : uniform_spacing(ts)

  # Change of basis x(t1, t2) → x_W(t1 - t2, (t1 + t2)/2)
  x_W = zeros(eltype(x), Nt, Nt)

  for T in 1:Nt
    # For a certain T ≡ (t1 + t2)/2, τ ≡ (t1 - t2) can be at most τ_max
    τ_max = min(T - 1, Nt - T)

    for τᵢ in (-τ_max):τ_max
      i = 1 + mod(τᵢ, Nt)
      x_W[i, T] = x[T + τᵢ, T - τᵢ]
    end
  end

  τs = 2Δt .* collect((-Nt ÷ 2):((Nt - 1) ÷ 2))

  if !fourier
    return fftshift(x_W, 1), (τs, ts)
  else
    dτ = 2Δt
    ωs = fftshift(fftfreq(Nt, 2oftype(float(dτ), π) / dτ))
    x_Ŵ = dτ .* fftshift(bfft(x_W, 1), 1)

    return x_Ŵ, (ωs, ts)
  end
end

function uniform_spacing(xs)
  h = xs isa AbstractRange ? step(xs) : xs[2] - xs[1]
  @assert isfinite(h) && h > 0 && all(isfinite, xs) "Time samples must have positive finite spacing"
  @assert all(i -> xs[i] > xs[i-1] && (xs isa AbstractRange || xs[i] - xs[i-1] ≈ h),
              2:length(xs)) "Time samples must be distinct and uniformly spaced"
  return h
end

""" 
    ft(xs, ys; inverse = false)

Fourier transform of the points `(xs, ys)`:

if `inverse`

``ŷ(t) = ∫dω/(2π) y(ω) e^{- i ω t}``

else

``ŷ(ω) = ∫dt y(t) e^{+ i ω t}``

Returns a tuple of the Fourier transformed points `(x̂s, ŷs)`
and requires at least two distinct paired samples with positive finite uniform
spacing. Ranges use their declared step.
"""
function ft(xs, ys; inverse::Bool = false)
  L = length(xs)
  @assert L == length(ys) && L >= 2 "Fourier transformation requires at least two paired samples"
  dx = uniform_spacing(xs)

  # Because the FFT calculates the transform as
  #   ỹ_k = \sum_j e^{±2pi i j k/n} y_j, from j=0 to j=n-1,
  # we need to transform this into the time and frequency units, 
  # which ends up scaling the frequencies `x̂s` by (2pi / dx).
  π2 = 2oftype(float(dx), π)
  x̂s = fftfreq(L, π2 / dx)

  # The resulting Fourier transform also picks up a phase
  ℯⁱᵠ = (inverse ? dx / π2 : dx) .* exp.((inverse ? -im : im) * xs[1] .* x̂s)

  # FFT
  ŷs = ℯⁱᵠ .* (inverse ? fft : bfft)(ys)

  return fftshift(x̂s), fftshift(ŷs)
end
