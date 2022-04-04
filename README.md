# KadanoffBaym.jl

## Overview

This software provides an _adaptive_ time-stepping algorithm for the solution of Kadanoff-Baym equations, two-time Volterra integro-differential equations. The code is 
written in [Julia][].


## Installation

To install, use Julia's built-in package manager

```julia
julia> ] add KadanoffBaym
```

The most recent version of KadanoffBaym.jl requires Julia v1.7 or later.


## Documentation

KadanoffBaym.jl exports a very little amount of functions, namely `kbsolve!`, `GreenFunction` and their possible time-symmetries, `Symmetrical` and `SkewHermitian`. Their documention can be accessed through Julia's built-in documenter

```julia
julia> ? kbsolve!
```

Importing the external `FFTW` and `Interpolations` packages will also export `wigner_transform` and `wigner_transform_itp` for Wigner transformations.


## Examples

Various examples of the algorithm in action are found in the [examples](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/tree/master/examples) folder.


## Scalability

For now, KadanoffBaym.jl is restricted to run on a single machine, for which the maximum number of threads available will be used. You can set this number by running Julia with the `thread` [flag](https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading)
```
julia -t auto
```


## Contributing

This is meant to be a community project and all contributions, via [issues](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/issues), [PRs](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/pulls) and [discussions](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/discussions) are encouraged and greatly appreciated.


## Citing

If you use KadanoffBaym.jl in your research, please cite our work
```
@misc{2110.04793,
    title={Adaptive Numerical Solution of Kadanoff-Baym Equations}, 
    author={Francisco Meirinhos and Michael Kajan and Johann Kroha and Tim Bode},
    year={2021},
    eprint={2110.04793},
}
```