
# KadanoffBaym.jl

## Overview

This software provides an _adaptive_ time-stepping algorithm for the solution of Kadanoff-Baym equations, two-time Volterra integro-differential equations. The code is written in [Julia](https://github.com/JuliaLang/julia).


## Installation

To install, use Julia's built-in package manager

```julia
julia> ] add KadanoffBaym
```

The most recent version of `KadanoffBaym.jl` requires Julia v1.7 or later.


## Documentation

KadanoffBaym.jl exports a very little amount of functions, namely `kbsolve!`, `GreenFunction` and their possible time-symmetries, `Symmetrical` and `SkewHermitian`. Their documention can be accessed through Julia's built-in documenter

```julia
julia> ? kbsolve!
```

Importing the external `FFTW` and `Interpolations` packages will also export `wigner_transform` and `wigner_transform_itp` for Wigner transformations.


## Examples


Various examples of the algorithm in action are found in the [examples](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/tree/master/examples) folder, including the T-matrix approximation for the Fermi-Hubbard model.

`KadanoffBaym.jl` is very easy use. For example, we can solve the tight-binding model in a few lines:


```julia
# quantum numbers
dim = 10

# Allocate the initial lesser and greater Green functions (time arguments at the end)
GL = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)
GG = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)

# initial conditions (only first site occupied)
GL[1, 1] = +im * diagm(vcat([1.0], [0.0 for _ in 1:dim-1]))
GG[1, 1] = -im * I(dim) + GL[1, 1];

# spacing of energy levels
ε = 5e-2

# Hamiltonian with on-site energies
H  = Diagonal([ε * k for k in 0:dim-1]) 

# add a nearest-neighbour hopping J=-1
H -= (SymTridiagonal(ones(dim, dim)) - 1.0I(dim)) |> Matrix

# now specify the right-hand sides of the equations of motion
# for the "vertical" evolution
function fv!(out, times, h1, h2, t1, t2)
    out[1] = -1.0im * H * GL[t1, t2]
    out[2] = -1.0im * H * GG[t1, t2]
end

# for the "diagonal" evolution
function fd!(out, times, h1, h2, t1, t2)
  fv!(out, times, h1, h2, t1, t2)
  out[1] .-= adjoint(out[1])
  out[2] .-= adjoint(out[2])
end

# call the solver
sol = kbsolve!(fv!, fd!, [GL, GG], (0.0, 100.0); atol=1e-7, rtol=1e-5)
```


## Scalability

For now, KadanoffBaym.jl is restricted to run on a single machine, for which the maximum number of threads available will be used. You can set this number by running Julia with the `thread` [flag](https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading)
```
julia -t auto
```


## Contributing

This is meant to be a community project and all contributions, via [issues](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/issues), [PRs](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/pulls) and [discussions](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/discussions) are encouraged and greatly appreciated.


## Citing

If you use KadanoffBaym.jl in your research, please cite our work:
```
@misc{2110.04793,
    title={Adaptive Numerical Solution of Kadanoff-Baym Equations}, 
    author={Francisco Meirinhos and Michael Kajan and Johann Kroha and Tim Bode},
    year={2021},
    eprint={2110.04793},
}
```
