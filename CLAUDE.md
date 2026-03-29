# CLAUDE.md

## Project Overview

KadanoffBaym.jl is a Julia package for solving Kadanoff-Baym equations — two-time nonlinear Volterra integro-differential equations arising in non-equilibrium quantum field theory and classical stochastic processes. It uses an adaptive variable-order Adams-Bashforth-Moulton (VCABM) method with automatic step size and order control.

**Paper**: [SciPost Phys. Core 5(2):30 (2022)](https://doi.org/10.21468/SciPostPhysCore.5.2.030)
**Repository**: https://github.com/NonequilibriumDynamics/KadanoffBaym.jl

## Source Structure (~850 lines)

```
src/
  KadanoffBaym.jl   Main module, exports
  gf.jl             GreenFunction type with symmetry-aware indexing (Symmetrical, SkewHermitian, OnePoint)
  kb.jl             kbsolve! — main solver, fan-like 2-time stepping scheme (Section 3.1 of paper)
  vcabm.jl          VCABMCache, predict!/correct!/adjust! — Adams method (Section 3.2.1)
  vie.jl            Volterra integral weights via Vandermonde matrices
  wigner.jl         Wigner-Ville transformation and Fourier transform helpers
  langreth.jl       Langreth rules for time-ordered convolutions
  utils.jl          Norm, error estimation, helper functions
```

## Key Types and Functions

- `GreenFunction{T,N,A,U}` wraps an array with symmetry `U` (Symmetrical, SkewHermitian, OnePoint). Only stores the lower triangle; `setindex!` automatically enforces the symmetry relation on the upper triangle.
- `kbsolve!(fv!, fd!, u0, (t0, tmax); ...)` is the main entry point. It takes in-place RHS functions and mutates `u0`. Returns `(t=times, w=weights)`.
- `VCABMCache` holds the predictor-corrector state (Adams coefficients, phi-star arrays, error estimates).
- `VectorOfArray` from RecursiveArrayTools is used internally to reshape the 2-time problem into a univariate ODE (Section 3.1, Eq. 25-26 of the paper).
- `TimeOrderedGreenFunction{T}` wraps lesser/greater components for Langreth rule convolutions via `conv`. Keldysh components extracted with `greater`, `lesser`, `retarded`, `advanced`.

## Dependencies

- RecursiveArrayTools (VectorOfArray) — v2.38+ and v3+ supported
- SpecialMatrices (Vandermonde) — for integration weight computation
- AbstractFFTs — for Wigner transform (FFTW is a test-only dependency)

## Testing

```sh
julia -e 'using Pkg; Pkg.activate("."); Pkg.test()'
```

27 tests across 4 test sets: GreenFunction, Solver (4 benchmarks with analytical solutions), Langreth, Wigner. Tests run in ~13s on Apple Silicon.

## CI

- GitHub Actions: Julia 1.10 (LTS) + latest, Ubuntu/macOS/Windows, x64/x86
- Documentation: Documenter.jl 1.x, deployed to GitHub Pages
- Weekly scheduled runs (Monday 2 AM UTC)
- CI workflow was auto-disabled due to inactivity; re-enabled March 2026

## RecursiveArrayTools v3 Compatibility Notes

RAT v3 introduced three breaking changes that required fixes:
1. `VectorOfArray` is no longer `<: AbstractArray` — `calculate_residuals!` dispatch was widened from `AbstractArray` to untyped
2. `view(va, :)` is broken for non-scalar element types — changed to `view(va, 1, :)` in `kb.jl`
3. Broadcasting `@.` on `VectorOfArray` with scalar elements now broadcasts down to scalars — added `calculate_residuals!` method for `Number`

Do not try to replace VectorOfArray with a custom type — the broadcast and iteration semantics are subtle and deeply coupled to the solver.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full release history.
