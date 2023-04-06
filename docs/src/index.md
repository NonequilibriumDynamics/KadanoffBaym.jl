# Welcome!

`KadanoffBaym.jl` is the first fully *adaptive* solver for Kadanoff-Baym equations written in Julia.

!!! tip
	To learn more about the solver and Kadanoff-Baym equations, have a look into our [accompanying paper](https://doi.org/10.21468/SciPostPhysCore.5.2.030).

## Installation

To install, use Julia's built-in package manager

```julia
julia> ] add KadanoffBaym
```

## Scalability

For now, `KadanoffBaym.jl` is restricted to run on a single machine, for which the maximum number of threads available will be used. You can set this number by running Julia with the `thread` [flag](https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading)

```
julia -t auto
```

## Examples

To learn how to work with `KadanoffBaym.jl`, there are two options:

- The [examples folder](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/tree/master/examples) of our repository, which contains notebooks for all of the systems studied in [our paper](https://doi.org/10.21468/SciPostPhysCore.5.2.030).

- The examples section of this documentation. If you are interested in _quantum_ dynamics, we recommend you start with the [tight-binding model](@ref TightBinding). More advanced users can jump directly to [Fermi-Hubbard model part I](@ref FHM_I) about the _second Born approximation_. [Part II](@ref FHM_II) shows how to solve the more involved ``T``-matrix approximation. 

!!! note
	`KadanoffBaym.jl` can also be used to simulate _stochastic processes_. An introduction to this topic is given [here](@ref StochasticProcesses).

## Library

`KadanoffBaym.jl` was designed to be lean and simple and hence only exports a handful of functions, namely [`GreenFunction`](@ref) (together with two possible time symmetries, `Symmetrical` and `SkewHermitian`) and the integrator [`kbsolve!`](@ref). Besides these, [`wigner_transform`](@ref) can be used to analyze data in a Wigner(-Ville) transformed basis and [`TimeOrderedGreenFunction`](@ref) to quickly compute the Keldysh components (`greater`, `lesser`, `retarded` and `advanced`) of time [`conv`](@ref)olutions via the Langreth rules.

!!! note
	You need to import an FFT library -- e.g., `FFTW` -- to use `wigner_transform`.


### Index

```@index
```

### Solver

```@docs
kbsolve!
```

### Green Functions

```@docs
GreenFunction
```

### Wigner Transformation

```@docs
wigner_transform
```

### Langreth's rules

```@docs
TimeOrderedGreenFunction
```

```@docs
conv
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