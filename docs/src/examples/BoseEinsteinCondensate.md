# [Bose-Einstein Condensate] (@id BEC)

In this example we will use `KadanoffBaym.jl` to study _dephasing_ in Bose-Einstein condensates (see Chp. 3 [here](https://bonndoc.ulb.uni-bonn.de/xmlui/handle/20.500.11811/8961)). To do this, we will need to solve _one-time_ differential equations for the condensate amplitude ``\varphi(t)`` and the so-called equal-time _Keldysh Green_ function ``G^K(t, t)``.

!!! hint
    A [Jupyter notebook](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/blob/master/examples/bose-einstein-condensate.ipynb) for this example is available in our [examples folder](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/tree/master/examples).

The Lindblad master equation describing this systems reads
```math
\begin{align}
    \begin{split}
    	\dot{\hat{\rho}} &= -i\omega_0[a^\dagger a, \hat{\rho}] +\frac{\lambda}{2}\left\{ 2a\hat{\rho} a^{\dagger} - \left( a^{\dagger}a\hat{\rho} + \hat{\rho} a^{\dagger}a \right)\right\} + \frac{\gamma}{2}\left\{ 2a^{\dagger}\hat{\rho} a - \left( aa^{\dagger}\hat{\rho} + \hat{\rho} aa^{\dagger} \right)\right\} \\
        &+ D\left\{ 2a^{\dagger}a\hat{\rho} a^{\dagger}a - \left( a^{\dagger}aa^{\dagger}a\hat{\rho} + \hat{\rho} a^{\dagger}aa^{\dagger}a \right)\right\},    
    \end{split}
\end{align}
```

where $\lambda > 0$ is the loss parameter, $\gamma > 0$ represents the corresponding gain, and $D > 0$ is the constant that introduces dephasing. The derivation for the equations of motion for $\varphi(t)$ and $G^K(t, t)$ is again given [here](https://bonndoc.ulb.uni-bonn.de/xmlui/handle/20.500.11811/8961) and leads to
```math
\begin{align}
    \begin{split}
    \dot{\varphi}(t) &=  -i\omega_0\varphi(t) -\frac{1}{2}{(\lambda - \gamma + {2} D)}\varphi(t), \\
    \dot{G}^K(t, t) &= -{(\lambda - \gamma)}G^K(t, t) - i{\left(\lambda + \gamma + {2} D |\varphi(t)|^2\right)}.
    \end{split}
\end{align}
```
To make these expressions more transparent, we set $\varphi(t) = \sqrt{2N(t)}\mathrm{e}^{i \theta(t)}$ and $G^K(t, t) = -i{(2\delta N(t) + 1)}$, where $N$ and $\delta N$ are the condensate and non-condensate occupation, respectively. For these quantities, we obtain
```math
\begin{align}
    \begin{split}
    \dot{N} &=  {(\gamma - \lambda -{2} D)}N, \\
    \delta \dot{N} &=  \gamma{(\delta N + 1)} - \lambda\delta N + {2}DN.
    \end{split}
\end{align}
```


Translating all of this into code is now straightforward! We start by defining the condensate and the _Keldysh_ Green function along with their initial conditions:
```julia
using KadanoffBaym, LinearAlgebra

# parameters
ω₀ = 1.0
λ = 0.0
γ = 0.0
D = 1.0 

# initial occupations
N = 1.0
δN = 0.0

# One-time function for the condensate
φ = GreenFunction(zeros(ComplexF64, 1, 1), OnePoint)

# Allocate the initial Green functions (time arguments at the end)
GK = GreenFunction(zeros(ComplexF64, 1, 1, 1, 1), SkewHermitian)

# Initial conditions
GK[1, 1, 1, 1] = -im * (2δN + 1)
φ[1, 1] = sqrt(2N)
```

In the next step, we write down the equations of motion:
```julia
# we leave the vertical equation empty since we can solve for GK in equal-time only
function fv!(out, ts, h1, h2, t, t′)
    out[1] = zero(out[1])
end

# diagonal equation for GK
function fd!(out, ts, h1, h2, t, _)
    # cast condensate to the right type
    φ_mat = reduce(hcat, abs.(φ[t]).^2)
    out[1] = -(λ - γ) * GK[t, t] .- im * (λ + γ + 2D .* φ_mat)
end

# one-time equation for condensate amplitude
function f1!(out, ts, h1, t)
    out[1] = -im * ω₀ * φ[t] - (1/2) * (λ - γ + 2D) * φ[t]
end
```

Calling the solver is again a one-liner:
```julia
sol = kbsolve!(fv!, fd!, [GK,], (0.0, 1.0); atol=1e-6, rtol=1e-4, v0 = [φ,], f1! =f1!)
```

If you want a plot of the results, you can find it in our corresponding [Jupyter notebook](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl/blob/master/examples/bose-einstein-condensate.ipynb).