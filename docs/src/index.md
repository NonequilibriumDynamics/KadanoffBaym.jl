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

#### Continuous problem and time integration

Write ``D(t)=G(t,t)`` for the diagonal and ``F_v[G,v](t,s)`` for the
vertical right-hand side, including its memory integrals. The continuous problem
on the lower triangle is

```math
\begin{aligned}
D'(t) &= F_d[G,v](t), \\
v'(t) &= F_1[G,v](t), \\
G(t,s) &= D(s) + \int_s^t F_v[G,v](r,s)\,dr, \qquad t\ge s.
\end{aligned}
```

The symmetry determines the upper triangle. On a step starting at ``t_n``, the
variables to advance are

```math
Y_n(t)=\bigl(G(t,t_0),\ldots,G(t,t_n),D(t),v(t)\bigr).
```

At ``t=t_n``, the newest fixed column and moving diagonal have the same value,
but their derivatives are ``F_v[G,v](t_n,t_n)`` and ``F_d[G,v](t_n)``.
This is why both entries are present in the time integrator's state.

For a differentiable skew-Hermitian extension, diagonal compatibility requires
``F_d(t)=F_v(t,t)-F_v(t,t)^\dagger``. This guarantees only first-derivative
matching, not the higher regularity needed by memory quadrature. Also,
``\partial_1G(t,s)=-\partial_2G(s,t)^\dagger``: reflecting a value changes the
derivative direction, so it cannot supply above-diagonal vertical RHS samples.

Closing equations for these nodal variables requires a further numerical
approximation. A general memory kernel also depends on ``G(t,s)`` inside the new
triangle ``t_n<s<t\le t_{n+1}``, which the row and diagonal values alone do not
determine. The supplied memory weights approximate integrands from samples at
the stored times. This interpolation is part of the discrete method.

For a method with internal stages, the reconstruction of the two-time history,
stage equations, and memory quadrature must be specified together. One possible
construction uses a common trial field on the new strip and its triangular
corner. Existing columns start from their accepted values at ``t_n``; new
columns start from ``G(s,s)=D(s)``. Stage evaluations then use that field in the
memory integrands, without accepting it as history. Temporal and memory
quadratures can have different nodes and weights, but their combined accuracy
and stability need analysis. Nonlinear kernels may require more accurate
quadrature than the polynomial degree of the trial field suggests.

The current time integrator uses a PECE Adams method: predict, evaluate the
right-hand side, apply one Adams–Moulton correction, and evaluate again. It does
not solve the implicit Adams–Moulton equation to convergence. Its divided
differences, reconstruction for newly added columns, and local error estimator
belong to that particular method. Other time integrators would need their own
stage and error-control implementation; `kbsolve!` does not yet provide a backend
interface.

For a local defect ``e_{a,j,\alpha}`` in field ``a``, time column ``j``, and
internal component ``\alpha``, the acceptance norm is

```math
\max_{a,j}\left(\frac{1}{m_a}\sum_{\alpha=1}^{m_a}
\left|\frac{e_{a,j,\alpha}}
{\mathrm{atol}+\mathrm{rtol}\max(|u^{\mathrm{prev}}_{a,j,\alpha}|,
                               |u^{\mathrm{next}}_{a,j,\alpha}|)}\right|^2\right)^{1/2}.
```

Here ``m_a`` is the number of internal components of one field value. Thus
matrix components use RMS scaling, but adding time columns or inactive fields
does not dilute a localized defect. The same norm is used for order selection
and automatic initial-step selection. It still measures an Adams local defect;
it does not estimate accumulated solution error or old memory-quadrature bias.

#### A common-quadrature reference

The fixed-grid method in `test/reference.jl` applies composite trapezoids to both
the evolution and memory integrals. Its formulas and panel selection are independent
of production Adams; it shares Green-function storage and the polynomial-weight
primitive, which is checked separately against exact polynomial moments.
For every existing column, its implicit equation is

```math
G_{n+1,j}=G_{n,j}+\frac{h_n}{2}
  \left(F^Q_{v,n,j}+F^Q_{v,n+1,j}\right),\qquad j\le n,
```

where ``F^Q`` uses trapezoidal memory weights on its own endpoint grid. The
diagonal and one-time variables obey the same formula with their respective
right-hand sides. All new values are solved together. In particular, the newest
column uses ``F_{v,n,n}``, while the moving diagonal uses ``F_{d,n}``.

For memory variables of the form ``z(t)=\int_{t_0}^t b(s,y(s))\,ds``, this is
algebraically equivalent to applying the trapezoidal method to the augmented ODE
``z'=b(t,y)``. This equivalence does not extend to arbitrary transformations of a
memory equation. The reference interpolates sampled kernel values; integrating
a product of reconstructed Green functions exactly would give a different rule.

With bounded third derivatives along the evolution direction, bounded second
derivatives of each smooth kernel branch, and a discrete right-hand side uniformly
Lipschitz over the grids in the maximum norm over the complete history, both
quadrature errors accumulate as ``O(H^2)``, where ``H=\max_n h_n`` includes the
supplied history. The diagonal cusps lie on panel boundaries. For sufficiently
small steps, discrete Grönwall then bounds the solution error by
``C_T(E_0+H^2+\sum_n\varepsilon_n)``, with ``E_0`` the error in the supplied
history and ``\varepsilon_n`` the residual of each solved step. This assumes that
callbacks evaluate the specified discrete right-hand side; errors from any
approximate auxiliary solves contribute additional defects. A fixed nonlinear
residual tolerance must therefore be tightened when testing convergence on
increasingly fine grids.

The reference uses Picard iteration and reports failure if it does not converge;
it is not a stiff solver. It provides an independent check of the adaptive method,
without replacing its higher-order formulas or introducing another public API.
The weighted-callback interface assumes continuous integrands at panel boundaries;
the kernel interface below also permits distinct memory endpoint traces.
Coefficient quenches still need separate endpoint RHS traces. Callbacks may evaluate algebraic auxiliary
quantities or controlled auxiliary solves; arbitrary coupled integral equations
are not solved merely by repeating a mutation callback.

Both references also accept `equations = (; fv!, fd!, f1!, callback)` in place of
the two RHS arguments, with `f1! = nothing` when unused. Their signatures are
`fv!(out, ts, integral, i, j)`, likewise `fd!`, `f1!(out, ts, integral, i)`, and
`callback(ts, integral, i, j)`. The model now states its integrands and limits:

```julia
memory = integral(k -> K(i, k, j), j, i; init=zero(G[i,j]))
```

Here `k`, `i`, and `j` are **node indices**. The solver supplies the panel rule;
the model supplies `K`. `init` is an additive initial value with a compatible
accumulation type and is never mutated. Equal bounds do not evaluate the kernel.
General intervals are assembled from local panel integrals, avoiding cancellation
when a tiny interval follows a much longer history. Existing weighted callbacks
remain supported, and `benchmarks/quantum.jl` supplies both interfaces from the
same dimer and second-Born equations. This interface is experimental and unexported.
The weighted adapters preserve existing behavior; opaque weighted callbacks do
not acquire kernel sampling-domain checks automatically.

With the shared equations and matrix views, five alternating warmed measurements
at 16 stored intervals reduced second-Born trapezoidal runtime from 19.7 to 15.8 ms
and Lobatto runtime from 25.6 to 19.3 ms, using one Julia and BLAS thread.
Allocations fell from 90.2 to 57.3 MB and from 115.6 to 73.8 MB, respectively.
The free dimer's runtime was unchanged within measurement scatter. On 8, 16, and
32 intervals, both quantum models agreed with their preceding implementations
to roundoff over the complete stored field.

The default `sample_domain=min(i,j):max(i,j)` permits samples only between the
integration limits. A stencil requiring any other node is rejected before sampling.
Widening `sample_domain` declares a smooth continuation of **this kernel**, not
additional history values: only the current row or jointly solved block is available.
For example, split a cusp into its two smooth branches, with `s=ts[j]`:

```julia
memory = integral(k -> s-ts[k], 1, j; sample_domain=eachindex(ts), init=0.0)
memory = integral(k -> ts[k]-s, j, i; sample_domain=eachindex(ts), init=memory)
```

Each branch is linear on its full sampling domain, so either reference integrates
this cusp exactly. The closures may also supply different traces at a shared
endpoint. For a general kernel, including products of history values, the model
must actually provide the declared smooth continuation; splitting bounds alone
does not provide it. There is no off-grid reconstruction or automatic regularity test.

#### Experimental smooth block reference

`test/lobatto_reference.jl` implements a fixed-block alternative that solves
startup, time evolution, and memory integration together. It is a numerical
experiment, not a replacement for the adaptive `kbsolve!` interface. Run
`julia --project --threads=1 benchmarks/quantum.jl 128 3` to compare it with
Adams and the independent trapezoidal reference on the quantum examples.

For a block ``[T,T+h]``, let ``c=(0,1/2,1)``. Integrating the three Lagrange
basis polynomials gives the two nonzero rows

```math
A=\begin{pmatrix}5/24&1/3&-1/24\\1/6&2/3&1/6\end{pmatrix}.
```

These rows advance existing columns, the diagonal, and one-time variables.
The same rows integrate each memory panel: complete blocks use Simpson weights;
a midpoint limit uses the first row, including the endpoint of its own block.
Midpoints remain in accepted history because subsequent memory quadrature uses them.
All unknown stage values and their callback-managed self-energies are updated
before evaluating any stage derivative. This is the integrated-polynomial
construction underlying [collocation methods](https://beckassets.blob.core.windows.net/product/readingsample/425979/9780521806152_excerpt_001.pdf).

The remaining new corner has a direct triangular construction. Write
``F_{ij}=f_v(T+c_i h,T+c_j h)`` and ``D_1=G(T+h/2,T+h/2)`` with indices
``i,j\in\{0,1,2\}``. Integrating the quadratic, total-degree-two interpolant of
``F`` on the lower triangle gives

```math
G_{21}=D_1+\frac h4(F_{11}+F_{21})
             -\frac h{24}(F_{20}-2F_{10}+F_{00}).
```

The last term is the curvature contribution from the block's first column.
It requires no above-diagonal RHS evaluation. Substituting the newly computed
``D_1`` into this equation makes the block iteration depend on known boundary
values plus ``h`` times derivatives.

More generally, if ``P_2(t,s)`` interpolates the six lower-triangle vertical RHS
samples and ``d_2(s)`` interpolates the diagonal RHS, the corresponding cubic patch is

```math
D_h(s)=D(T)+\int_T^s d_2(u)\,du,\qquad
U_h(t,s)=D_h(s)+\int_s^t P_2(u,s)\,du.
```

The corner formula is the value of this patch at ``(T+h,T+h/2)``. Compatible
diagonal stage derivatives make its reflected first derivatives match along
the diagonal; they do not impose matching of higher derivatives.

For smooth compatible equations, complete-block evolution has local defect
``O(h^5)``. Midpoint stages and the new corner have defect ``O(h^4)``, but each
column receives its birth defect only once. Memory quadrature has error
``O(H^4)``, where ``H`` is the largest block width, also on nonuniform block grids.
The full-field argument assumes bounded fourth evolution derivatives, bounded
mixed derivatives through degree three on the new triangle, and bounded third
and fourth integration-coordinate derivatives of the exact memory integrands
on their actual sampling support. The discrete RHS, including eliminated
auxiliary quantities, must be Lipschitz in the maximum history norm with a
constant independent of the number of nodes.

Under these assumptions, follow each column from its diagonal birth: at most
two short pieces contribute ``O(H^4)`` defects, while complete-block defects sum
to ``O(\sum_n h_n^5)=O(H^4)``. Complete memory blocks have positive weights of
total mass ``h_n``; a midpoint panel has absolute weight sum ``7h_n/12``.
Consequently history perturbations are uniformly bounded, and enter evolution
multiplied by ``h_n``. Absorbing the current-block term for sufficiently small
blocks and applying discrete Grönwall gives the conditional, finite-time nodal
estimate ``C_T(H^4+\sum_n\varepsilon_n)``. Here ``\varepsilon_n`` bounds each
block's equation residual in solution-increment units. Choosing
``\varepsilon_n=O(h_nH^4)`` keeps nonlinear solve error at fourth order.
The constant can depend on physical frequencies and ``T``; this is not a stiff
or long-time stability result. Tests cover the full stored field, coupled
one-time variables, and nonuniform blocks.

A future history-restart extension must also account for the quadrature error
of that history, not just its nodal error. Continuing on an unchanged block grid
needs the accepted nodes, block partition, and consistent auxiliary values.
Continuous reconstruction is additionally needed for off-grid sampling or
regridding; generic quadratic interpolation of ``G`` is only third-order accurate.

In the two-site second-Born example, 16, 32, and 64 stored time intervals gave
full-field errors ``7.17\times10^{-6}``, ``4.57\times10^{-7}``, and
``2.87\times10^{-8}`` against the independent refined trapezoidal reference.
The free dimer also showed fourth-order convergence. On the tested dyadic meshes,
the least costly second-Born runs meeting error ``10^{-5}`` took approximately
25.4 ms with this reference and 174 ms with Adams (three warmed runs, one Julia
and BLAS thread). These are sampled fixed-grid comparisons: finer mesh searches
can change the ratio, and the block solve costs more on the same grid.

For weighted callbacks, sum over `eachindex(w1)` and `eachindex(w2)`,
including samples beyond the integration limit. Callbacks that truncate at
`t1` or `t2`, or solve auxiliary equations sequentially, need reformulation.
Only initial data are accepted; adaptive control and stored-history restarts are
not implemented. Each sampled kernel must be smooth across its interpolation block.
An unsplit diagonal cusp violates this assumption; separate branch kernels can
use smooth continuations as above, while `quadrature=:piecewise` uses one-sided
stencils without requiring them. Picard convergence also requires
small enough steps, so this reference is not a stiff solver or a method that
removes the cost of fast oscillations.

The scalar oscillator illustrates this distinction. With ``z=i\omega h``, the
endpoint factor is ``(1+z/2+z^2/12)/(1-z/2+z^2/12)``, of modulus one. The
midpoint and new-corner factor is ``(1-z^2/24)/(1-z/2+z^2/12)``, whose modulus
is smaller than one for nonzero real ``\omega h``. Endpoint amplitude preservation
does not extend to every stored pair. Likewise, the manufactured cusp problem
``F_v=i[\int_0^t|\tau-s|\,d\tau-(s^2+(t-s)^2)/2]`` has exact solution
``G=i``, but this block rule produces ``D_h(1)-i=-ih^2/18`` on a uniform macro
grid even when enforcing ``F_d=F_v-F_v^\dagger``. Regression tests retain both
identities as limits of the unsplit formulation; the explicitly split cusp test
recovers the constant solution to roundoff with both references.

Adaptive block control remains a separate problem. Step doubling on unchanged
history cannot reveal shared old quadrature bias. Nor does a universal Richardson
factor of 15 apply when a coarse midpoint becomes a refined macro endpoint.
Independent memory indicators need additional accurate samples or controlled
history reconstruction. Correcting inaccurate past solution values requires
replaying their dependent evolution or solving a global correction equation.

#### Memory quadrature

The default `quadrature=:smooth` uses variable-order polynomial quadrature and
assumes that the kernel is smooth across each interpolation stencil. Its accuracy
also depends on the conditioning of those stencils; smoothness alone does not
prevent large interpolation weights on a strongly graded grid.

The integration interval and the sample domain are different mathematical objects.
For example, samples at ``0,h,2h`` give the quadratic-exact weights
``(5h/12,2h/3,-h/12)`` for an integral over ``[0,h]``. Using the sample at ``2h``
is causal once the current time has reached it, provided the integrand has a
smooth continuation there. Crossing the time diagonal is therefore not itself
an error: regularity must be checked for the actual integrand, including any
time-ordering factors. See [NESSi, Sections 8.5–8.6](https://arxiv.org/abs/1911.01211).

The current callback interface uses stored prefix weights: `w2` for the second
grid node (`t2 == 2`) always has just the two endpoint samples. Linear exactness forces
the trapezoidal rule, which approximates ``\int_0^h\tau^2\,d\tau`` by ``h^3/2``
instead of ``h^3/3``. Thus increasing the Adams order does not establish uniformly
high-order accuracy for every memory integral. Improving these short prefixes
requires additional samples or derivative information and a consistent startup
procedure; changing later weights alone cannot repair already accepted errors.

For a continuous kernel whose derivative changes at the second time argument,
or a quadrature rule with an explicit bound on weight amplification, use

```julia
sol = kbsolve!(fv!, fd!, [G], (t0, tmax); quadrature=:piecewise)
```

On a panel ``[t_i,t_{i+1}]`` of width ``h``, use the closest past node ``t_\ell``
on the same smooth branch satisfying ``t_i-t_\ell\ge h``. Integrate the quadratic
interpolant through these three nodes. If no such node exists, use a trapezoid.
The stencil is determined by the grid and the branch, independently of the Adams
order. In particular, a tiny preceding interval cannot produce arbitrarily large
quadrature weights when a larger step follows it.

For ``r=h/(t_i-t_\ell)\le1``, the three quadratic weights are

```math
\omega_i=h\left(-\frac{r^2}{6(1+r)},\;
                   \frac12+\frac r6,\;
                   \frac{3+2r}{6(1+r)}\right).
```

Their absolute sum is at most ``7h/6``. For a kernel with three bounded derivatives
on each branch, the accumulated quadrature error is ``O(H^3)``, where ``H`` is
the largest panel width. Trapezoids are needed only while the new panel is wider
than all available history on that branch; those startup panels also contribute
``O(H^3)`` in total. The implementation pairs this quadrature with Adams order at
most three. This is a choice of numerical method, not a restriction on the
continuous problem. Smaller `kmax` and `kmax_vie` remain valid upper bounds.

Accepted panels stay fixed, and reconstructing the same grid gives the same
quadrature. No panel uses samples from later intervals. Splitting at ``t_j``
replaces quadratic panels satisfying ``t_\ell<t_j\le t_i`` with trapezoids;
panels wholly to the left of the split remain unchanged.
With zero-padded panel vectors, the weights used by `fv!` and `callback`
are

```math
w_n^{(j)} = w_n + \sum_{\ell_i<j\le i}(\tau_i-\omega_i).
```

Each panel correction has three entries. The weights for all second-time points
are constructed once per trial grid and reused through prediction and correction.
Rebuilding them on a rejected trial requires no separate accepted quadrature
state. The returned `sol.w` contains the ordinary cumulative weights. Copy the
temporary callback weights when retaining them beyond that evaluation.
A jump in the kernel's value requires separate one-sided endpoint values and is
not covered by this continuous-cusp rule.

`atol` and `rtol` control the Adams local error estimate. They do not certify the
accuracy of a memory integral computed from unresolved samples. Check memory
quadrature convergence by refining `dtmax` and any supplied initial time grid;
smaller future steps cannot recover information missing from past panels.

### Green Functions

```@docs
GreenFunction
Symmetrical
SkewHermitian
OnePoint
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
greater
lesser
retarded
advanced
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

A list of papers citing this work can be found [here](@ref "Papers citing KadanoffBaym.jl").
