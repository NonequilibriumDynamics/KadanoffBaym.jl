# Documentation

KadanoffBaym.jl is the first fully *adaptive* solver for Kadanoff-Baym equations written in Julia. To learn more about the solver and Kadanoff-Baym equations, have a look into our [accompanying paper](https://scipost.org/SciPostPhysCore.5.2.030).

## Installation

To install, use Julia's built-in package manager

```julia
julia> ] add KadanoffBaym
```

The most recent version of `KadanoffBaym.jl` requires Julia v1.7 or later.

## Examples

For now, please see the [examples folder](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/tree/master/examples).

## Library

`KadanoffBaym.jl` was designed to be lean and simple and hence only exports a handful of functions, namely [`GreenFunction`](@ref) (together with two possible time symmetries, `Symmetrical` and `SkewHermitian`) and the integrator [`kbsolve!`](@ref).

When you import the external packages `FFTW` and `Interpolations`, `KadanoffBaym.jl` will also export the functions `wigner_transform` and `wigner_transform_itp` to perform Wigner transformations.


### Index

```@index
```

### Solver

```@docs
kbsolve!(fv!, fd!, u0::Vector{<:GreenFunction}, (t0, tmax))
```

### Green Functions

```@docs
GreenFunction{T,N,A,U<:AbstractSymmetry}
```

#### Wigner Transformation

```@docs
wigner_transform(x::AbstractMatrix)
```

```@docs
wigner_transform_itp(x::AbstractMatrix, ts::Vector)
```

## Citation

If you use `KadanoffBaym.jl` in your research, please cite our [paper](https://scipost.org/SciPostPhysCore.5.2.030):
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