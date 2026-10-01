# CLAUDE.md

## Project Overview

KadanoffBaym.jl is a Julia package for solving Kadanoff-Baym equations — two-time nonlinear Volterra integro-differential equations arising in non-equilibrium quantum field theory and classical stochastic processes. It uses an adaptive variable-order Adams-Bashforth-Moulton (VCABM) method with automatic step size and order control.

**Paper**: [SciPost Phys. Core 5(2):30 (2022)](https://doi.org/10.21468/SciPostPhysCore.5.2.030)
**Repository**: https://github.com/NonequilibriumDynamics/KadanoffBaym.jl

## Source Structure

```
src/
  KadanoffBaym.jl   Main module, exports
  gf.jl             GreenFunction type with symmetry-aware indexing (Symmetrical, SkewHermitian, OnePoint)
  kb.jl             kbsolve! — main solver, fan-like 2-time stepping scheme (Section 3.1 of paper)
  vcabm.jl          VCABMCache, predict!/correct!/adjust! — Adams method (Section 3.2.1)
  quadrature.jl     Lagrange weights, incremental history rules, and fixed-panel integration
  wigner.jl         Wigner-Ville transformation and Fourier transform helpers
  langreth.jl       Langreth rules for time-ordered convolutions
  utils.jl          Norm, error estimation, helper functions
```

## Key Types and Functions

- `GreenFunction{T,N,A,U}` wraps the supplied array, including both time triangles. Off-diagonal `setindex!` applies the value-block transpose or negative adjoint when exchanging the two time indices; construction and diagonal assignment do not project values onto the symmetry. `OnePoint` has one time axis and no exchange rule.
- `kbsolve!(fv!, fd!, u0, (t0, tmax); ...)` is the main entry point. It takes in-place RHS functions and mutates `u0`. Returns `(t=times, w=weights)`.
- One `VCABMCache` holds the coupled predictor-corrector state: two-time fields have growing history rows, and one-time fields have singleton rows. All fields share Adams coefficients, order, accepted-step count, and error control.
- Adams state uses ordinary arrays indexed as `u[field][time]`; each value can be a scalar or a numerical array. The callback-only `FieldSlice` in `kb.jl` exposes the fields at one time coordinate while preserving each field's storage type.
- `TimeOrderedGreenFunction{T}` wraps lesser/greater components for Langreth rule convolutions via `conv`. Keldysh components extracted with `greater`, `lesser`, `retarded`, `advanced`.

## Dependencies

- SpecialMatrices (Vandermonde) — for integration weight computation
- AbstractFFTs — for Wigner transform (FFTW is a test-only dependency)

## Testing

```sh
julia -e 'using Pkg; Pkg.activate("."); Pkg.test()'
```

Tests cover GreenFunction storage, Adams polynomial identities and error estimates, Volterra quadrature, analytical and quantum evolution, Langreth convolutions, and Wigner transforms. Run with bounds checking when changing numerical storage or indexing.

## CI

- GitHub Actions: Julia 1.12 + latest, Ubuntu/macOS/Windows, x64/x86
- Documentation: Documenter.jl 1.x, deployed to GitHub Pages
- Weekly scheduled runs (Monday 2 AM UTC)
- CI workflow was auto-disabled due to inactivity; re-enabled March 2026

## Numerical state

Keep the field-major layout: it retains each field's value type and avoids allocating a separate field vector for every history time. The Adams algebra acts directly on these arrays. `FieldSlice` exposes the fields at one time coordinate; assignment copies values into owned storage with fixed axes, and broadcasting supports ordinary Base array operations.

The Adams core requests derivatives through `f!(out)`. RHS callbacks write directly to this owned output: assigning an array captures its numerical values, while `.=` writes into its buffer. No separate RHS workspace or post-callback copy is needed. Prediction and correction each publish the complete coupled state before invoking the auxiliary callback. Only two-time fields undergo triangular history completion and row extension.

`ϕ_n` stores the accepted Newton differences, including the accepted RHS as its first entry. Prediction rescales these into `ϕstar_n` and reconstructs the unsupported young columns. Correction forms `ϕ_np1`; acceptance swaps the two difference tables, while rejection preserves `ϕ_n`. The saved predictor RHS `f_pred` is separate from these tables. Automatic startup uses the same coupled field norm, adding the vertical derivative at the diagonal to the already present diagonal derivative.

`state_norm` takes the maximum over fields and time coordinates, with component RMS inside each value. Preserve this distinction when constructing test states. Accepted derivative histories and integral accumulators must own their numerical array leaves; shallow copies retain callback workspaces, while `deepcopy` preserves repeated internal aliases.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full release history.
