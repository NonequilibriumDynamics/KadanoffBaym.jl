
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


## Documentation

The documentation can be found [here](https://nonequilibriumdynamics.github.io/KadanoffBaym.jl).



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
