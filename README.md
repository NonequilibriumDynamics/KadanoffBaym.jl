
# KadanoffBaym.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nonequilibriumdynamics.github.io/KadanoffBaym.jl/dev/)
[![CI](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/workflows/CI/badge.svg)](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/NonequilibriumDynamics/KadanoffBaym.jl/branch/master/graph/badge.svg?token=AAZVQLIKN2)](https://codecov.io/gh/NonequilibriumDynamics/KadanoffBaym.jl)

## Overview

`KadanoffBaym.jl` is the first fully *adaptive* solver for Kadanoff-Baym equations written in Julia. To learn more about the solver and Kadanoff-Baym equations, have a look into our [accompanying paper](https://doi.org/10.21468/SciPostPhysCore.5.2.030).


## Installation

To install, simply use Julia's built-in package manager

```julia
julia> ] add KadanoffBaym
```

The most recent version of `KadanoffBaym.jl` requires Julia `v1.7` or later.


## Documentation

Our documentation can be found [here](https://nonequilibriumdynamics.github.io/KadanoffBaym.jl).

`KadanoffBaym.jl` was designed to be lean and simple, and hence only exports a handful of functions, namely the data structure [`GreenFunction`](https://nonequilibriumdynamics.github.io/KadanoffBaym.jl/dev/#KadanoffBaym.GreenFunction) and the integrator [`kbsolve!`](https://nonequilibriumdynamics.github.io/KadanoffBaym.jl/dev/#KadanoffBaym.kbsolve!-Tuple{Any,%20Any,%20Vector{%3C:GreenFunction},%20Any}).


## Examples

To learn how to work with `KadanoffBaym.jl`, there are two options:

- The [examples folder](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/tree/master/examples) of our repository, which contains notebooks for all of the systems studied in [our paper](https://doi.org/10.21468/SciPostPhysCore.5.2.030).

- The examples section of our [documentation](https://nonequilibriumdynamics.github.io/KadanoffBaym.jl). If you are interested in _quantum_ dynamics, we recommend you start with the [tight-binding model](@ref TightBinding). More advanced users can jump directly to [Fermi-Hubbard model part I](@ref FHM_I) about the _second Born approximation_. [Part II](@ref FHM_II) shows how to solve the more involved ``T``-matrix approximation. `KadanoffBaym.jl` is versatile and can also be used to simulate _stochastic processes_. An introduction to this topic is given [here](StochasticProcesses).

`KadanoffBaym.jl` is very easy to use. For example, we can solve the tight-binding model in a few lines:


```julia
using KadanoffBaym, LinearAlgebra

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
sol = kbsolve!(fv!, fd!, [GL, GG], (0.0, 20.0); atol=1e-8, rtol=1e-6)
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
