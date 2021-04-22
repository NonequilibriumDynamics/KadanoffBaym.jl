"""
    wigner_transform(x::AbstractMatrix; ts=1:size(x,1), fourier=true)

Wigner-Ville transformation

    W_x(ω, T) = i ∫dt x(T + t/2, T - t/2) e^{+i ω t}

or

    W_x(τ, T) = x(T + t/2, T - t/2)

of a 2-point function `x`. Returns a tuple of `W_x` and the corresponding
axes (`ω, T`) or (`τ`, `T`), depending on the `fourier` keyword.

The motivation for the Wigner transformation is that, given an autocorrelation
function `x`, it reduces to the spectral density function at all times `T` for 
stationary processes, yet it is fully equivalent to the non-stationary 
autocorrelation function. Therefore, the Wigner (distribution) function tells 
us, roughly, how the spectral density changes in time.

### Keyword Arguments:
  - `ts::AbstractVector`: Time grid for `x`. Defaults to a `UnitRange`
  - `fourier::Bool`: Whether to Fourier transform. Defaults to `true`

## References:
https://en.wikipedia.org/wiki/Wigner_distribution_function

http://tftb.nongnu.org
"""
function wigner_transform(x::AbstractMatrix; ts=1:size(x,1), fourier=true)
  LinearAlgebra.checksquare(x)

  Nt = size(x, 1)
  @assert length(ts) == Nt

  # NOTE: code for non-equidistant time-grids
  # The resulting transformation will be in an equidistant grid
  # ts_equidistant = range(ts[1], stop=ts[end], length=length(ts))
  # tᵢ = searchsortedlast.(Ref(ts_equidistant), ts[T]) # map to an equidistant index

  # Change of basis x(t1, t2) → x′(t1 - t2, (t1 + t2)/2)
  x′ = zero(x)

  for T ∈ 1:Nt
    # For a certain T ≡ (t1 + t2)/2, τ ≡ (t1 - t2) can be at most τ_max
    τ_max = minimum([T-1, Nt-T, Nt÷2-1])

    τs = -τ_max:τ_max
    is = 1 .+ rem.(Nt.+τs, Nt)

    for (i, τᵢ) ∈ zip(is, τs)
      x′[i,T] = x[T+τᵢ,T-τᵢ]
    end
    
    τ = Nt ÷ 2
    if T <= Nt-τ && T >= τ+1
      x′[τ+1,T] = 0.5 * (x[T+τ,T-τ] + x[T-τ,T+τ])
    end
  end

  x′ = circshift(x′, (Nt ÷ 2, 0))
  τs = ts - reverse(ts)

  if !fourier 
    return x′, (τs, ts)
  else
    # Because the FFT calculates the transform as y_k = \sum_j e^{-2pi i j k/n}
    # from j=0 to j=n-1, we need to transform this into our time and frequency
    # units, which ends up scaling the frequencies `ωs` by (-2pi / dτ).
    dτ = τs[2] - τs[1]
    ωs = fftfreq(Nt, -2pi / dτ)
    is = sortperm(ωs)

    x′ = mapslices(x -> im * dτ * fft(x) .* exp.(1.0im * τs[1] .* ωs), x′, dims=1)
    
    return x′[is,:], (ωs[is], ts)
  end
end

function wigner_transform_itp(x::AbstractMatrix, ts::Vector; fourier=true)
  ts_lin = range(first(ts), last(ts), length=length(ts))
  itp = interpolate((ts, ts), x, Gridded(Linear()))
  wigner_transform([itp(t1,t2) for t1 in ts_lin, t2 in ts_lin]; ts=ts_lin, fourier=fourier)
end
