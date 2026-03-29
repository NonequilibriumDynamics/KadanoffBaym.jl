# Changelog

## v1.4.0 (2026-03-29)

### Infrastructure
- Updated GitHub Actions to v4 (`checkout`, `cache`, `codecov`, `setup-julia`)
- Julia test matrix now covers 1.10 (LTS) and latest stable across Ubuntu, macOS, Windows
- CI schedule changed from daily to weekly
- Minimum Julia raised to 1.10
- Upgraded Documenter.jl from 0.27 to 1.x
- Fixed documentation workflow (was pinned to Julia 1.7)

### Compatibility
- **RecursiveArrayTools v3 support**: widened compat to `"2.38, 3"` with targeted fixes for API changes
- Widened AbstractFFTs compat to `"1"`, SpecialMatrices to `"3"`

### Code quality
- Replaced floating-point `gamma_star` constants (indices 10-16) with exact rationals
- Added type parameter to `TimeOrderedGreenFunction{T}` for type stability
- Fixed type-generic `skew_hermitify!` (`0.0` -> `zero(eltype(x))`)
- Removed commented-out code, fixed typos

## v1.3.2 (2024-03-17)

- Fixed typos in documentation

## v1.3.1 (2024-03-10)

- Fixed symmetry enforcement: `gf[:,:,t1,t2] = a` now correctly applies the SkewHermitian adjoint operation on the full matrix, not element-wise

## v1.3.0 (2023-12-13)

- Integral-free calculation of the Volterra weights via Vandermonde matrices
- New BEC example for one-point functions
- Performance regression fix from v1.2.x
- Removed dependency on `OrdinaryDiffEq.jl`

## v1.2.1 (2023-10-22)

- Bug fixes and stability improvements

## v1.2.0 (2023-10-15)

- Added support for stepping 1-time functions alongside 2-time functions
- Added Langreth's rules (`TimeOrderedGreenFunction`, `conv`, `greater`, `lesser`, `retarded`, `advanced`)
- Removed some internal dependencies

## v1.0.1 (2022-06-08)

- Improved documentation
- Removed dependency on `EllipsisNotation.jl`
- Test compliance fixes

## v1.0.0 (2022-06-02)

- Initial release accompanying the [paper](https://doi.org/10.21468/SciPostPhysCore.5.2.030)
- Adaptive variable-order Adams-Bashforth-Moulton solver for Kadanoff-Baym equations
- `GreenFunction` type with symmetry-aware indexing (`Symmetrical`, `SkewHermitian`)
- `kbsolve!` integrator
- `wigner_transform` for time-frequency analysis
- Examples: tight-binding, Fermi-Hubbard (2nd Born, T-matrix), bosonic dimer, BEC, stochastic processes
