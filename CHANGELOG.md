# Changelog

## v1.6.0 (unreleased)

### Numerical correctness
- Correct Adams startup, predictor/corrector pairing, and embedded error estimates. Control all coupled one-time and two-time fields together; preserve accepted history across rejected steps and restarts, respect step limits, and handle mixed time precision.
- Reconstruct new-column Adams history from the physical lower triangle, eliminating above-diagonal RHS evaluations. Stabilize Volterra weights on uneven grids and rebuild weights for supplied histories.
- Add `quadrature=:piecewise` for continuous memory kernels with a derivative cusp at the second time argument. It uses causal panel stencils and caps Adams order at three. Tolerances control the Adams estimate; memory quadrature still requires a convergence check.
- Correct GreenFunction time-exchange symmetry, copying, indexing, and packed-storage growth. Preserve full Langreth convolution components, including noncommuting matrix blocks, without imposing an invalid skew-Hermitian projection.
- Correct odd-grid Wigner lag sampling, Float32 output, and Fourier spacing. Document the unchanged Fourier convention and the information omitted by the discrete Wigner representation.
- Correct Hubbard Hartree signs and time-local interaction vertices in the examples. Solve T-matrix endpoint equations directly with an LU factorization.

### Implementation and validation
- Remove RecursiveArrayTools. Use ordinary field-major arrays, accepted Newton differences, and reusable numerical buffers for the coupled Adams state.
- Add independent trapezoidal and joint Lobatto reference solvers, polynomial identities, quantum convergence tests, and allocation/runtime benchmarks. Reference solvers remain experimental test and benchmark tools.

### Compatibility notes
- RHS assignment `out[a] = value` now copies into field-owned storage and preserves its axes and element type. Use `out[a] .= value` to write directly; reusable workspaces must be local to each thread. Corrected error control can change adaptive grids at the same tolerances.
- Packed matrix-valued GreenFunctions now expose the two time axes as their array dimensions. Whole-container transpose/adjoint use Base views; arithmetic returns ordinary arrays when the original symmetry is not preserved. In particular, `imag` of a skew-Hermitian GreenFunction is symmetric, and non-real scalar scaling returns an ordinary array. Code relying on exact result types or `.data` after these operations may need adjustment.

## v1.5.0 (2026-06-18)

> Released as a minor version (not a patch) because the minimum Julia was raised
> from 1.10 to 1.12, which narrows the supported Julia range. Julia's AutoMerge
> forbids narrowing the supported Julia versions in a patch release, so the
> never-registered `v1.4.1` tag is superseded by this `v1.5.0` release.

### Bug fixes
- Fixed crash with matrix-valued GreenFunctions (e.g. `GreenFunction(zeros(ComplexF64, 2, 2, 1, 1), SkewHermitian)`) on RecursiveArrayTools v3: `VectorOfArray[i, :]` now returns collapsed arrays, so `extend!` accesses `.u` directly to preserve element types

### Tests
- Added fermionic dimer test (2-site tight-binding with exact matrix exponential solution)
- Added Ornstein-Uhlenbeck (Brownian motion) test with closed-form analytical solution
- Added geometric Brownian motion test with coupled nonlinear dynamics
- Added Bose-Einstein condensate test with mixed OnePoint/SkewHermitian evolution
- Test count increased from 27 to 32

### Other
- Minimum Julia raised to 1.12
- Added list of 21 papers citing the SciPost publication to README and docs
- Added stable docs, version, and license badges to README

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
