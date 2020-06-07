"""
    Wigner-Ville transformation

  W_x(ω, T) = i ∫dt x(T + t/2, T - t/2) e^{+i ω t}

The motivation for the Wigner function is that it reduces to the spectral 
density function at all times `T` for stationary processes, yet it is fully 
equivalent to the non-stationary autocorrelation function. Therefore, the 
Wigner function tells us (roughly) how the spectral density changes in time.

## Parameters
  - 2-point correlator (`x`)
  - Time-grid (`ts`): Defaults to a `UnitRange` time-grid
  - Number of ω points (`Nf`): Number of frequencies. Defaults to `size(x,1)`.
  - Fourier opt (`fourier`): Whether to Fourier transform. Defaults to `true`

## Returns
  - The transform (`x`) with the corresponding axes (`ω, t`) or (`t`, `t`)

## References:
https://en.wikipedia.org/wiki/Wigner_distribution_function
http://tftb.nongnu.org
"""
function wigner_transform(x; ts=1:size(x,1), Nf=size(x,1), fourier=true)
  LinearAlgebra.checksquare(x)

  Nt = size(x, 1)
  @assert length(ts) == Nt

  # NOTE: code for non-equidistant time-grids
  # The resulting transformation will be in an equidistant grid
  # ts_equidistant = range(ts[1], stop=ts[end], length=length(ts))
  # tᵢ = searchsortedlast.(Ref(ts_equidistant), ts[icol]) # map to an equidistant index

  # Change of basis x(t1, t2) → x′(t1 - t2, (t1 + t2)/2)
  x′ = zero(x)

  for icol=1:Nt    
    tᵢ = icol

    # For a certain tᵢ ≡ (t1 + t2)/2, τ ≡ (t1 - t2) can be at most τ_max
    τ_max = minimum([tᵢ-1, Nt-tᵢ, Nt÷2-1])

    τs = -τ_max:τ_max
    is = 1 .+ rem.(Nt.+τs, Nt)

    for (i, τᵢ) in zip(is, τs)
      x′[i,icol] = x[tᵢ+τᵢ,tᵢ-τᵢ]
    end
    
    τ = Nt ÷ 2
    if tᵢ <= Nt-τ && tᵢ >= τ+1
      x′[τ+1,icol] = 0.5 * (x[tᵢ+τ,tᵢ-τ] + x[tᵢ-τ,tᵢ+τ])
    end
  end

  if !fourier 
    return x′, (ts, ts) #(ts_equidistant, ts_equidistant)
  else
    ωs = fftfreq(Nf, 1.0)
    x′ = circshift(x′, -(isodd(Nt) ? Nt÷2 + 1 : Nt÷2))
    x′ = nfft(NFFTPlan(ωs, 1, size(x′)), x′)
    # x′ = mapslices(x -> nufft2(x, ωs, 1e-8), x′, dims=1)
    # x′ = fft(x′, [1])
    
    # Because the FFT calculates the transform as y_k = \sum_j e^{-2pi i j k/n}
    # from j=0 to j=n-1, we need to transform this into our time and frequency
    # units, which ends up scaling the frequencies `ωs` by (-2pi / dt). Note that
    # dt = 2x the dt of `ts` because τ runs from -t to t.
    ωs *= -pi / (ts[2] - ts[1])

    is = sortperm(ωs)
    return 1.0im * x′[is,:], (ωs[is], ts)
  end
end
