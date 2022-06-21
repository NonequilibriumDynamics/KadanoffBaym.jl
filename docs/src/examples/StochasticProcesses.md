# [Stochastic Processes] (@id StochasticProcesses)

`KadanoffBaym.jl` can also be used to simulate _stochastic processes_. Below, we will give the simplest example to illustrate how to do this. In cases where [other methods](https://diffeq.sciml.ai/stable/tutorials/sde_example/) are too expensive or inapplicable, this approach can be an economic and insightful alternative. 

!!! tip
    More background on the connection between `KadanoffBaym.jl` and stochastic processes can be found in section 4.2 of [our paper](https://doi.org/10.21468/SciPostPhysCore.5.2.030) and also [here](https://doi.org/10.1088/1751-8121/ac73c6).


## [Ornstein-Uhlenbeck Process] (@id OUProcess)

The Ornstein-Uhlenbeck (OU) process\cite{VanKampen2007, gardiner1985handbook} is defined by the stochastic differential equation (SDE)
```math
\begin{align*}
    \mathrm{d} x(t) = -\theta x(t)\mathrm{d} t + \sqrt{D}\mathrm{d} W(t)
\end{align*}
```
where ``W(t)``, ``t >0`` is a one-dimensional Brownian motion and ``\theta > 0``. The Onsager-Machlup path integral
```math
\begin{align*}
    \int\mathcal{D}x\exp{\left\{-\frac{1}{2D}\int\mathrm{d}t\, \left(\partial_t{x}(t) +\theta x(t)\right)^2\right\}}
\end{align*}
```
is a possible starting point to derive the corresponding MSR action via a Hubbard-Stratonovich transformation. The classical MSR action
```math
\begin{align*}
    S[x, \hat{x}] =   \int\mathrm{d}t\,\left[  \mathrm{i}\hat{x}(t)(\partial_t{x}(t) + \theta x(t)) + D \hat{x}^2(t)/2 \right]
\end{align*}
```
is then equivalent to it. Note also that we are employing It\^o regularisation for simplicity. It is also common to define a purely imaginary response field ``\tilde{\hat{x}} = \mathrm{i}\hat{x}``, which is then integrated along the imaginary axis. The retarded Green function ``G^R`` and the statistical propagator ``F`` are then commonly defined as
```math
\begin{align*}
    G^R(t, t') &= \langle{x(t)\hat{x}(t')}\rangle, \\
	F(t, t') &= \langle{x(t)x(t')}\rangle - \langle{x(t)}\rangle \langle{x(t')}\rangle.
\end{align*}
```
The equations of motion of the response Green functions are
```math
\begin{align*}
	\delta(t - t') &= -\mathrm{i}\partial_t G^A(t, t') + \mathrm{i}\theta G^A(t, t'), \\    
	\delta(t - t') &= \phantom{-}\mathrm{i}\partial_t G^R(t, t') + \mathrm{i}\theta G^R(t, t'),
\end{align*}
```
admitting the solutions
```math
\begin{align*}
	G^A(t, t') = G^A(t - t') = -\mathrm{i}\Theta(t' - t)\mathrm{e}^{-\theta{(t' - t)}}, \\
	G^R(t, t') = G^R(t - t') = -\mathrm{i}\Theta(t - t')\mathrm{e}^{-\theta{(t - t')}}.
\end{align*}
```
The equations of motion of the statistical propagator in "vertical" time ``t`` and "horizontal" time ``t'`` read
```math
\begin{align*}
    \partial_t F(t, t')    & =  -\theta F(t, t') + \mathrm{i} DG^A(t, t'),  \\
    \partial_{t'} F(t, t') & =  -\theta F(t, t') + \mathrm{i} DG^R(t, t'),   
\end{align*}
```
respectively, while in Wigner coordinates ``T = (t+t')/2``, ``\tau = t - t'``, we find
```math
\begin{align*}
    \partial_{T} F(T, \tau)_W    & =  -2\theta F(T, \tau)_W + \mathrm{i} D\left( G^A(T, \tau)_W + G^R(T, \tau)_W \right),  \\
    \partial_{\tau} F(T, \tau)_W & =  \frac{\mathrm{i} D}{2} \left( G^A(T, \tau)_W - G^R(T, \tau)_W \right). 
\end{align*}
```
To cover the two-time mesh completely, one could in principle use any two of the four equations for ``F``. Our convention is to pick the equation in "vertical" time ``t`` with ``t>t'`` and the equation in "diagonal" time ``T`` with ``\tau=0``, such that together with the symmetries relations of the classical Green functiosn, the problem is fully determined by the initial conditions and these two equations:
```math
\begin{align*}
    \partial_t F(t, t')    & =  -\theta F(t, t') + \mathrm{i} DG^A(t, t'), \\
    \partial_{T} F(T, 0)_W & =  -2\theta F(T, 0)_W + D,    
\end{align*}
```
where we have used the response identity ``G^A(T, 0)_W + G^R(T, 0)_W = -\mathrm{i}``, and ``G^A(t, t') = 0`` when ``t > t'``. For comparison, the analytical solution for the variance or statistical propagator reads
```math
\begin{align*}
    \mathcal{F}(t, t') &= \mathcal{F}(0, 0)\mathrm{e}^{-\theta(t + t')} - \frac{D}{2\theta} \left( \mathrm{e}^{-\theta(t + t')} - \mathrm{e}^{-\theta |t - t'|} \right)\\
	&= \left( \mathcal{F}(0, 0) - \frac{D}{2\theta} \right)\mathrm{e}^{-\theta(t + t')} + \frac{\mathrm{i} D}{2\theta}\left( G^A(t, t') + G^R(t, t') \right).
\end{align*}
```

To apply `KadanoffBaym.jl`, we begin by defining parameters and initial conditions:
```julia
# Final time
tmax = 4.0

# Drift
θ = 1.

# Diffusion strength (in units of θ)
D = 8.0

# Initial condition
N₀ = 1.0
F = GreenFunction(N₀ * ones(1, 1), Symmetrical)
```
!!! note
    Observe how we have used the symmetry `Symmetrical` to define the _classical_ type of [`GreenFunction`](@ref). 

Now the equation in "vertical" time ``t`` is simply    
```julia
# Right-hand side for the "vertical" evolution
function fv!(out, _, _, _, t1, t2)
    out[1] = -θ * F[t1, t2]
end
```

!!! warning
    In this numerical version of the analytical equation above, we have made explicit use of the fact that ``G^A(t, t') = 0`` when ``t > t'``. However, since `KadanoffBaym.jl` uses a so-called _multi-step predictor-corrector method_ to solve equations (i.e. a method using not a single point but _multiple_ points from the past to predict the next point), it can happen that points with ``t' > t`` are actually accessed, in which case the ``G^A(t, t')`` term in the above equation _does_ contribute. One way to prevent this would be to restrict the solver to using a one-step method early on, i.e. until enough points with ``t > t'`` are known. 

In "diagonal" time ``T`` we have
```julia
# Right-hand side for the "diagonal" evolution
function fd!(out, _, _, _, t1, t2)
    out[1] = -θ * 2F[t1, t2] + D
end
```
To learn about the signatures `(out, _, _, _, t1, t2)` of `fv!` and `fd!`, consult the documentation of [`kbsolve!`](@ref). All we need to do now is
```julia
# Call the solver
sol = kbsolve!(fv!, fd!, [F], (0.0, tmax); atol=1e-8, rtol=1e-6)
```
The numerical results obtained with `KadanoffBaym.jl` are shown in [our paper](https://doi.org/10.21468/SciPostPhysCore.5.2.030) and are in agreement with the analytical solutions.

!!! hint
    A [Jupyter notebook](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/blob/master/examples/brownian-motion.ipynb) for this example is available in our [examples folder](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/tree/master/examples).