
# KadanoffBaym.jl

[![CI](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/workflows/CI/badge.svg)](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/actions?query=workflow%3ACI)

## Overview

This software provides an [_adaptive_ time-stepping algorithm for the solution of Kadanoff-Baym equations](https://scipost.org/SciPostPhysCore.5.2.030). The code is written in [Julia](https://julialang.org).


## Installation

To install, use Julia's built-in package manager

```julia
julia> ] add KadanoffBaym
```

The most recent version of `KadanoffBaym.jl` requires Julia v1.7 or later.


## Documentation

`KadanoffBaym.jl` was designed to be lean and simple and hence only exports a handful of functions, namely `GreenFunction` (together with two possible time symmetries, `Symmetrical` and `SkewHermitian`) and the integrator `kbsolve!`. The documentation for these can be accessed through Julia's built-in documenter

```julia
julia> ? kbsolve!
```

Importing the external `FFTW` and `Interpolations` packages will also export `wigner_transform` and `wigner_transform_itp` for Wigner transformations.


## Examples

Various examples of the algorithm in action are found in the [examples](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/tree/master/examples) folder, including the T-matrix approximation for the Fermi-Hubbard model.

`KadanoffBaym.jl` is very easy to use. For example, we can solve the tight-binding model in a few lines:


```julia
using LinearAlgebra, KadanoffBaym

# quantum numbers
dim = 10

# Allocate the initial lesser and greater Green functions (time arguments at the end)
GL = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)
GG = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)

# initial conditions (only first site occupied)
GL[1, 1] = +im * diagm([isone(i) ? 1.0 : 0.0 for i in 1:dim])
GG[1, 1] = -im * I(dim) + GL[1, 1];

# spacing of energy levels
ε = 5e-2

# Hamiltonian with on-site energies and nearest-neighbour hopping
H = SymTridiagonal([ε * (i-1) for i in 1:dim], -ones(dim))

# now specify the right-hand sides of the equations of motion
# for the "vertical" evolution
function fv!(out, times, h1, h2, t1, t2)
    out[1] = -im * H * GL[t1, t2]
    out[2] = -im * H * GG[t1, t2]
end

# for the "diagonal" evolution
function fd!(out, times, h1, h2, t1, t2)
  fv!(out, times, h1, h2, t1, t2)
  out[1] -= adjoint(out[1])
  out[2] -= adjoint(out[2])
end

# call the solver
sol = kbsolve!(fv!, fd!, [GL, GG], (0.0, 100.0); atol=1e-6, rtol=1e-3)
```


## Scalability

For now, `KadanoffBaym.jl` is restricted to run on a single machine, for which the maximum number of threads available will be used. You can set this number by running Julia with the `thread` [flag](https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading)
```
julia -t auto
```


## Contributing

This is meant to be a community project and all contributions, via [issues](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/issues), [PRs](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/pulls) and [discussions](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/discussions) are encouraged and greatly appreciated.


## Citing

If you use `KadanoffBaym.jl` in your research, please cite our work:
```
@Article{10.21468/SciPostPhysCore.5.2.030,
	title={{Adaptive Numerical Solution of Kadanoff-Baym Equations}},
	author={Francisco Meirinhos and Michael Kajan and Johann Kroha and Tim Bode},
	journal={SciPost Phys. Core},
	volume={5},
	issue={2},
	pages={30},
	year={2022},
	publisher={SciPost},
	doi={10.21468/SciPostPhysCore.5.2.030},
	url={https://scipost.org/10.21468/SciPostPhysCore.5.2.030},
}
```
