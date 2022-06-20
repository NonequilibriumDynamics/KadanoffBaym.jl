# [Fermi-Hubbard Model I] (@id FHM_I)

!!! note
    A [Jupyter notebook](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/blob/master/examples/fermi-hubbard.ipynb) related to this example is available in our [examples folder](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/tree/master/examples).


To see `KadanoffBaym.jl` in full strength, we need to consider an interacting system such as the _Fermi-Hubbard_ model
```math
\begin{align*}
	\hat{H} &= - J \sum_{\langle{i,\,j}\rangle}\sum_\sigma \hat{c}^{\dagger}_{i,\sigma} \hat{c}^{\phantom{\dagger}}_{i+1,\sigma} + U\sum_{i=1}^L  \hat{c}^{\dagger}_{i,\uparrow} \hat{c}^{\phantom{\dagger}}_{i,\uparrow}   \hat{c}^{\dagger}_{i,\downarrow} \hat{c}^{\phantom{\dagger}}_{i,\downarrow}, 
\end{align*}
```
where the ``\hat{c}_{i,\sigma}^{\dagger},\, \hat{c}_{i,\sigma}^{\phantom{\dagger}}``, ``\sigma=\uparrow, \downarrow`` are now spin-dependent fermionic creation and annihilation operators. The model describes electrons on a lattice that can hop to neighbouring sites via the coupling ``J`` while also feeling an on-site interaction ``U``. The *spin-diagonal* Green functions are defined by
```math
\begin{align*}
	\left[\boldsymbol{G}_\sigma^<(t, t')\right]_{ij} &= G^<_{ij, \sigma}(t, t') = \phantom{-} \mathrm{i}\left\langle{\hat{c}_{j, \sigma}^{{\dagger}}(t')\hat{c}_{i, \sigma}^{\phantom{\dagger}}(t)}\right\rangle, \\
	\left[\boldsymbol{G}_\sigma^>(t, t')\right]_{ij} &= G^>_{ij, \sigma}(t, t') = -\mathrm{i}\left\langle{\hat{c}_{i, \sigma}^{\phantom{\dagger}}(t)\hat{c}_{j, \sigma}^{{\dagger}}(t')}\right\rangle.
\end{align*}
```
The equations of motion for these Green functions in "vertical" time ``t`` can be written compactly as
```math
\begin{align*}
        (\mathrm{i}\partial_t - \boldsymbol{h}) \boldsymbol{G}_\sigma^\lessgtr(t, t') &= \int_{t_0}^{t}\mathrm{d}{s}  \left[\boldsymbol{\Sigma}_\sigma^>(t, s) - \boldsymbol{\Sigma}_\sigma^<(t, s) \right] \boldsymbol{G}_\sigma^\lessgtr(s, t') + \int_{t_0}^{t'}\mathrm{d}{s}  \boldsymbol{\Sigma}_\sigma^\lessgtr(t, s) \left[\boldsymbol{G}_\sigma^<(s, t') - \boldsymbol{G}_\sigma^>(s, t') \right],
\end{align*}
```
where ``\boldsymbol{h}`` describes the single-particle contributions (i.e. the hopping), and the matrices ``\boldsymbol{G}^\lessgtr`` are assumed to be block-diagonal in the spin degree-of-freedom. The *Hartree-Fock* part of the self-energy now is
```math
\begin{align*}
    \Sigma^{\mathrm{HF}}_{\uparrow,\,ij}(t, t') = {\mathrm{i}}\delta_{ij}\delta(t - t') G^<_{\downarrow,ii}(t, t),\\
    \Sigma^{\mathrm{HF}}_{\downarrow,\,ij}(t, t') = {\mathrm{i}}\delta_{ij}\delta(t - t') G^<_{\uparrow,ii}(t, t).
\end{align*}    
```

In the so-called _second Born approximation_, the contribution to next order in ``U`` is also taken into account:
```math
\begin{align*}
    \Sigma^\lessgtr_{ij, \uparrow}  (t, t') = U^2 G^\lessgtr_{ij, \uparrow}(t, t') G^\lessgtr_{ij, \downarrow}(t, t') G^\gtrless_{ji, \downarrow}(t', t),\\
    \Sigma^\lessgtr_{ij, \downarrow}(t, t') = U^2 G^\lessgtr_{ij, \downarrow}(t, t') G^\lessgtr_{ij, \uparrow}(t, t') G^\gtrless_{ji, \uparrow}(t', t).
\end{align*}
```


Now that we have the equations set up, we import `KadanoffBaym.jl` alongside some auxiliary packages:
```julia
using KadanoffBaym, LinearAlgebra, BlockArrays
```
Then, we use `KadanoffBaym`'s built-in data structure [`GreenFunction`](@ref) to define our lesser and greater Green functions
```julia
# Lattice size
L = 8

# Allocate the initial Green functions (time arguments at the end)
GL_u = GreenFunction(zeros(ComplexF64, L, L, 1, 1), SkewHermitian)
GG_u = GreenFunction(zeros(ComplexF64, L, L, 1, 1), SkewHermitian)
GL_d = GreenFunction(zeros(ComplexF64, L, L, 1, 1), SkewHermitian)
GG_d = GreenFunction(zeros(ComplexF64, L, L, 1, 1), SkewHermitian)
```
Observe that we denote ``\boldsymbol{G}_\uparrow^<`` by `GL_u` and ``\boldsymbol{G}_\downarrow^<`` by `GL_d`, for instance. As a lattice structure, we choose the 8-site _3D qubic lattice_ shown in Fig. 8 of [our paper](https://doi.org/10.21468/SciPostPhysCore.5.2.030).

As an (arbitrary) Gaussian initial condition, we take a non-equilibrium distribution of the charge over the cube (all electrons at the bottom of the cube):
```julia
# Initial conditions
N_u = zeros(L)
N_d = zeros(L)

N_u[1:4] = 0.1 .* [1, 1, 1, 1]
N_d[1:4] = 0.1 .* [1, 1, 1, 1]

N_u[5:8] = 0.0 .* [1, 1, 1, 1]
N_d[5:8] = 0.0 .* [1, 1, 1, 1]

GL_u[1, 1] = 1.0im * diagm(N_u)
GG_u[1, 1] = -1.0im * (I - diagm(N_u))
GL_d[1, 1] = 1.0im * diagm(N_d)
GG_d[1, 1] = -1.0im * (I - diagm(N_d))
```

!!! note
    Accessing [`GreenFunction`](@ref) with only *two* arguments gives the whole matrix at a given time, i.e. `GL_u[1, 1]` is equivalent to `GL_u[:, :, 1, 1]`.

To keep our data ordered, we define an auxiliary `struct` to hold them:
```julia
Base.@kwdef struct FermiHubbardData2B{T}
    GL_u::T
    GG_u::T
    GL_d::T
    GG_d::T

    ΣL_u::T = zero(GL_u)
    ΣG_u::T = zero(GG_u)
    ΣL_d::T = zero(GL_d)
    ΣG_d::T = zero(GG_d)
end

data = FermiHubbardData2B(GL_u=GL_u, GG_u=GG_u, GL_d=GL_d, GG_d=GG_d)
```
Furthermore, we also defined an auxiliary `struct` specifying the parameters of the model:
```julia
Base.@kwdef struct FermiHubbardModel{T}
    # interaction strength
    U::T

    # 8-site 3D cubic lattice
    h = begin
        h = BlockArray{ComplexF64}(undef_blocks, [4, 4], [4, 4])
        diag_block = [0 -1 0 -1; -1 0 -1 0; 0 -1 0 -1; -1 0 -1 0]
        setblock!(h, diag_block, 1, 1)
        setblock!(h, diag_block, 2, 2)
        setblock!(h, Diagonal(-1 .* ones(4)), 1, 2)
        setblock!(h, Diagonal(-1 .* ones(4)), 2, 1)

        h |> Array
    end

    H_u = h
    H_d = h
end

# Relatively small interaction parameter
const U₀ = 0.25
model = FermiHubbardModel(U = t -> U₀)
```
Note how we have defined the interaction parameter ``U`` as a (constant) Julia `Function` - this enables us to study quenches with time-dependent interaction (s. also [our paper](https://doi.org/10.21468/SciPostPhysCore.5.2.030)).

The final step before setting up the actual equations is to define a callback function for the self-energies in *second Born approximation*:
```julia
# Callback function for the self-energies
function second_Born!(model, data, times, _, _, t, t′)
    # Unpack data and model
    (; GL_u, GG_u, GL_d, GG_d, ΣL_u, ΣG_u, ΣL_d, ΣG_d) = data
    (; U) = model
        
    # Resize self-energies when Green functions are resized    
    if (n = size(GL_u, 3)) > size(ΣL_u, 3)
        resize!(ΣL_u, n)
        resize!(ΣG_u, n)
        resize!(ΣL_d, n)
        resize!(ΣG_d, n)        
    end
    
    # The interaction varies as a function of the forward time (t+t')/2
    U_t = U((times[t] + times[t′])/2)
    
    # Define the self-energies
    ΣL_u[t, t′] = U_t^2 .* GL_u[t, t′] .* GL_d[t, t′] .* transpose(GG_d[t′, t])
    ΣL_d[t, t′] = U_t^2 .* GL_u[t, t′] .* GL_d[t, t′] .* transpose(GG_u[t′, t])
    
    ΣG_u[t, t′] = U_t^2 .* GG_u[t, t′] .* GG_d[t, t′] .* transpose(GL_d[t′, t])
    ΣG_d[t, t′] = U_t^2 .* GG_u[t, t′] .* GG_d[t, t′] .* transpose(GL_u[t′, t])
end
```
!!! note
    The omitted arguments `_` in the function definition refer to the _adaptive_ integration weights of `KadanoffBaym`. They are only needed when the self-energy contains actual time integrals (as in the ``T``-matrix approximation).

Unlike for the non-interacting [tight-binding model](@ref TightBinding), the equations of motion above contain _integrals_ over time. This makes them so-called _Volterra integro-differential equations_ (VIDEs). Now the integrals are always either of the _first_ form
``\int_{t_0}^{t}\mathrm{d}{s}  \left[\boldsymbol{\Sigma}_\sigma^>(t, s) - \boldsymbol{\Sigma}_\sigma^<(t, s) \right] \boldsymbol{G}_\sigma^\lessgtr(s, t')``
or of the _second_ form
``\int_{t_0}^{t'}\mathrm{d}{s}  \boldsymbol{\Sigma}_\sigma^\lessgtr(t, s) \left[\boldsymbol{G}_\sigma^<(s, t') - \boldsymbol{G}_\sigma^>(s, t') \right].`` The discretization of these time convolutions results in a sum of matrix-products over all time indices. To be as efficient as possible, this suggests to introduce two auxiliary functions that handle these integrations by avoiding unnecessary allocations:
```julia
# Auxiliary integrator for the first type of integral
function integrate1(hs::Vector, t1, t2, A::GreenFunction, B::GreenFunction, C::GreenFunction; tmax=t1)
    retval = zero(A[t1, t1])

    @inbounds for k in 1:tmax
        @views LinearAlgebra.mul!(retval, A[t1, k] - B[t1, k], C[k, t2], hs[k], 1.0)
    end
    return retval
end

# Auxiliary integrator for the second type of integral
function integrate2(hs::Vector, t1, t2, A::GreenFunction, B::GreenFunction, C::GreenFunction; tmax=t2)
    retval = zero(A[t1, t1])

    @inbounds for k in 1:tmax
        @views LinearAlgebra.mul!(retval, A[t1, k], B[k, t2] - C[k, t2], hs[k], 1.0)
    end
    return retval
end
```
!!! note
    The first argument `hs::Vector` denotes the _adaptive_ integration weights provided by `KadanoffBaym`. Since these depend on the boundary points of the integration, there will usually be different weight vectors for the two integrals.

We are finally ready to define the equations of motion! In "vertical" time, we have
```julia
# Right-hand side for the "vertical" evolution
function fv!(model, data, out, times, h1, h2, t, t′)
    # Unpack data and model
    (; GL_u, GG_u, GL_u, GG_d, ΣL_u, ΣG_u, ΣL_d, ΣG_d) = data
    (; H_u, H_d, U) = model

    # Real-time collision integrals
    ∫dt1(A, B, C) = integrate1(h1, t, t′, A, B, C)
    ∫dt2(A, B, C) = integrate2(h2, t, t′, A, B, C)
    
    # The interaction varies as a function of the forward time (t+t')/2
    U_t = U((times[t] + times[t′])/2)
    
    # Hartree-Fock self-energies
    ΣHF_u(t, t′) = im * U_t * Diagonal(GL_d[t, t])
    ΣHF_d(t, t′) = im * U_t * Diagonal(GL_u[t, t])
    
    # Equations of motion
    out[1] = -1.0im * ((H_u + ΣHF_u(t, t′)) * GL_u[t, t′] + 
            ∫dt1(ΣG_u, ΣL_u, GL_u) + ∫dt2(ΣL_u, GL_u, GG_u)
        )

    out[2] = -1.0im * ((H_u + ΣHF_u(t, t′)) * GG_u[t, t′] + 
            ∫dt1(ΣG_u, ΣL_u, GG_u) + ∫dt2(ΣG_u, GL_u, GG_u)
        )

    out[3] = -1.0im * ((H_d + ΣHF_d(t, t′)) * GL_d[t, t′] + 
            ∫dt1(ΣG_d, ΣL_d, GL_d) + ∫dt2(ΣL_d, GL_d, GG_d)
        )

    out[4] = -1.0im * ((H_d + ΣHF_d(t, t′)) * GG_d[t, t′] +
            ∫dt1(ΣG_d, ΣL_d, GG_d) + ∫dt2(ΣG_d, GL_d, GG_d)
        )  
    
    return out
end
```
As for the [tight-binding model](@ref TightBinding), the equation of motion in "diagonal" time ``T`` follows by subtracting its own adjoint from the vertical equation:
```julia
# Right-hand side for the "diagonal" evolution
function fd!(model, data, out, times, h1, h2, t, t′)
    fv!(model, data, out, times, h1, h2, t, t)
    out .-= adjoint.(out)
end
```
After defining a final time and some tolerances, we give everything to [`kbsolve!`](@ref):
```julia
# final time
tmax = 4

# tolerances
atol = 1e-8
rtol = 1e-6

# Call the solver
sol = kbsolve!(
    (x...) -> fv!(model, data, x...),
    (x...) -> fd!(model, data, x...),
    [data.GL_u, data.GG_u, data.GL_d, data.GG_d],
    (0.0, tmax);
    callback = (x...) -> second_Born!(model, data, x...),
    atol = atol,
    rtol = rtol,
    stop = x -> (println("t: $(x[end])"); flush(stdout); false)
)
```
That's it! Results for `tmax=32` are shown in [our paper](https://doi.org/10.21468/SciPostPhysCore.5.2.030). Note that this implementation via `KadanoffBaym` is about as compact as possible.



