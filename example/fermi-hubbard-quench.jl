### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ c925a3be-ab56-11ec-1b59-9d92ed462b9b
begin
	using Pkg; Pkg.activate()
	using KadanoffBaym
	using LinearAlgebra, BlockArrays
	using JLD

	using PyPlot
	PyPlot.plt.style.use("./paper.mplstyle")
	using LaTeXStrings
end

# ╔═╡ 12610b5c-5508-4dcb-a4ff-252eb654f1a2
using FFTW, Interpolations

# ╔═╡ 188dbc64-b2f4-4486-a849-9dcf1cd5cb4a
md"""
### Hamiltonian

```math
\begin{align}\begin{split}
    \hat{H} &= - J \sum_{\langle{i,\,j}\rangle}\sum_\sigma \hat{c}^{\dagger}_{i,\sigma} \hat{c}^{\phantom{\dagger}}_{i+1,\sigma} + U\sum_{i=1}^L  \hat{c}^{\dagger}_{i,\uparrow} \hat{c}^{\phantom{\dagger}}_{i,\uparrow}   \hat{c}^{\dagger}_{i,\downarrow} \hat{c}^{\phantom{\dagger}}_{i,\downarrow}, 
\end{split}\end{align}
```

### Green functions

```math
    G^>_{ij}(t, t') = -i \left\langle \hat{c}^{\phantom{\dagger}}_{i,\uparrow}(t) \hat{c}^{{\dagger}}_{i,\uparrow}(t') \right\rangle\\
    F^>_{ij}(t, t') = -i \left\langle \hat{c}^{\phantom{\dagger}}_{i,\downarrow}(t) \hat{c}^{{\dagger}}_{i,\downarrow}(t') \right\rangle\\
```

### Self-energies

```math
    \Sigma^{\mathrm{HF}}_{\uparrow,\,ij}(t, t') = {\mathrm{i}}\delta_{ij}\delta(t - t') F^<_{ii}(t, t)\\
    \Sigma^{\mathrm{HF}}_{\downarrow,\,ij}(t, t') = {\mathrm{i}}\delta_{ij}\delta(t - t') G^<_{ii}(t, t)
```

```math
    \Sigma^{\mathrm{NCA}}_{\uparrow,\,ij}(t, t') = U^2 G_{ij}(t, t') F_{ij}(t, t') F_{ji}(t', t)\\
    \Sigma^{\mathrm{NCA}}_{\downarrow,\,ij}(t, t') = U^2 F_{ij}(t, t') G_{ij}(t, t') G_{ji}(t', t)
```
"""

# ╔═╡ 582a1fa3-bb83-4751-8feb-3c1ad577aed6
Base.@kwdef struct FermiHubbardModel
    U
    
    # 8-site 3D cubic lattice
    H = begin
        h = BlockArray{ComplexF64}(undef_blocks, [4, 4], [4, 4])
        diag_block = [0 -1 0 -1; -1 0 -1 0; 0 -1 0 -1; -1 0 -1 0]
        setblock!(h, diag_block, 1, 1)
        setblock!(h, diag_block, 2, 2)
        setblock!(h, Diagonal(-1 .* ones(4)), 1, 2)
        setblock!(h, Diagonal(-1 .* ones(4)), 2, 1)

        full_h = BlockArray{ComplexF64}(undef_blocks, [8, 8], [8, 8])
        setblock!(full_h, h |> Array, 1, 1)
        setblock!(full_h, h |> Array, 2, 2)
        setblock!(full_h, zeros(ComplexF64, 8, 8), 1, 2)
        setblock!(full_h, zeros(ComplexF64, 8, 8), 2, 1)
        
        full_h |> Array
    end
    
    H1 = H[1:8, 1:8]
    H2 = H[1 + 8:2 * 8, 1 + 8:2 * 8]
end

# ╔═╡ bdf0d70e-7c3b-423c-aee7-e222164cdee9
struct FermiHubbardData{T}
    GL::T
    GG::T
    FL::T
    FG::T

    ΣNCA_up_L::T
    ΣNCA_up_G::T
    ΣNCA_down_L::T
    ΣNCA_down_G::T

    # Initialize problem
    function FermiHubbardData(GL::T, GG::T, FL::T, FG::T) where {T}
        new{T}(GL, GG, FL, FG, zero(GL), zero(GG), zero(FL), zero(FG))
    end
end

# ╔═╡ fc979ae4-1421-47a2-b8f4-ca69cbca2a3d
function integrate(times::Vector, t1, t2, i, j, A::GreenFunction, B::GreenFunction)
    retval = zero(A[i,j])

    rold = zero(retval)
    rnew = zero(retval)

    LinearAlgebra.mul!(rnew, view(A.data, :, :, t1, min(i,j)), view(B.data, :, :, min(i, j), t2))

    @inbounds @fastmath @simd for k = min(i, j):(max(i, j)-1)
        rold, rnew = rnew, rold # no data transfer
        LinearAlgebra.mul!(rnew, view(A.data, :, :, t1, k+1), view(B.data, :, :, k+1, t2))

        @. retval += (times[k+1] - times[k]) * (rnew + rold)
    end

    return 1 // 2 * sign(j - i) * retval
end;

# ╔═╡ fce4c81d-3091-41cf-8005-c54bff63c825
begin
	function fv!(model, data, out, times, t, t′)
	    (; GL, GG, FL, FG, ΣNCA_up_L, ΣNCA_up_G, ΣNCA_down_L, ΣNCA_down_G) = data
	    (; H1, H2, U) = model
	
	    # real-time collision integral
	    ∫dt(x...) = integrate(times, t, t′, x...)

		U_ = U((times[t] + times[t′])/2)
		
	    ΣHF_up(t, t′) = im * U_ * Diagonal(FL[t, t])
	    ΣHF_down(t, t′) = im * U_ * Diagonal(GL[t, t])
	    
	    out[1] = -1.0im * ((H1 + ΣHF_up(t, t′)) * GL[t, t′] + 
	            ∫dt(1, t, ΣNCA_up_G, GL) + ∫dt(t, t′, ΣNCA_up_L, GL) - ∫dt(1, t′, ΣNCA_up_L, GG)
	        )
	
	    out[2] = -1.0im * ((H1 + ΣHF_up(t, t′)) * GG[t, t′] + 
	            ∫dt(t′, t, ΣNCA_up_G, GG) - ∫dt(1, t, ΣNCA_up_L, GG) + ∫dt(1, t′, ΣNCA_up_G, GL)
	        )
	
	    out[3] = -1.0im * ((H2 + ΣHF_down(t, t′)) * FL[t, t′] + 
	            ∫dt(1, t, ΣNCA_down_G, FL) + ∫dt(t, t′, ΣNCA_down_L, FL) - ∫dt(1, t′, ΣNCA_down_L, FG)
	        )
	
	    out[4] = -1.0im * ((H2 + ΣHF_down(t, t′)) * FG[t, t′] +
	            ∫dt(t′, t, ΣNCA_down_G, FG) - ∫dt(1, t, ΣNCA_down_L, FG) + ∫dt(1, t′, ΣNCA_down_G, FL)
	        )
	
	    return out
	end
	
	function fd!(model, data, out, times, t, t′)
	    fv!(model, data, out, times, t, t)
	    out .-= adjoint.(out)
	end
	
	function self_energies!(model, data, times, t, t′)
	    (; GL, GG, FL, FG, ΣNCA_up_L, ΣNCA_up_G, ΣNCA_down_L, ΣNCA_down_G) = data
	    (; U) = model
	
	    if (n = size(GL, 3)) > size(ΣNCA_up_L, 3)
	        resize!(ΣNCA_up_L, n)
	        resize!(ΣNCA_up_G, n)
	        resize!(ΣNCA_down_L, n)
	        resize!(ΣNCA_down_G, n)
	    end

		U_ = U((times[t] + times[t′])/2)
		
	    ΣNCA_up_L[t, t′] = U_^2 .* GL[t, t′] .* FL[t, t′] .* transpose(FG[t′, t])
	    ΣNCA_up_G[t, t′] = U_^2 .* GG[t, t′] .* FG[t, t′] .* transpose(FL[t′, t])
	
	    ΣNCA_down_L[t, t′] = U_^2 .* FL[t, t′] .* GL[t, t′] .* transpose(GG[t′, t])
	    ΣNCA_down_G[t, t′] = U_^2 .* FG[t, t′] .* GG[t, t′] .* transpose(GL[t′, t])
	end
end

# ╔═╡ 8d547d1b-b4b0-4467-a3aa-1499d358e5ee
begin
	# quantum numbers
	dim = 8
	
	# Allocate the initial Green functions (time arguments at the end)
	GL = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)
	GG = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)
	FL = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)
	FG = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)
	
	# Initial condition
	N_up = zeros(8)
	N_down = zeros(8)
	N_up[1:4] = [0.7, 0.0, 0.7, 0.0]
	N_down[1:4] = [0.0, 0.25, 0.0, 0.25]
	
	N_up[5:8] = [0.0, 0.4, 0.0, 0.4]
	N_down[5:8] = [0.65, 0.0, 0.65, 0.0]
	
	GL[1, 1] = 1.0im * diagm(N_up)
	GG[1, 1] = -1.0im * (I - diagm(N_up))
	FL[1, 1] = 1.0im * diagm(N_down)
	FG[1, 1] = -1.0im * (I - diagm(N_down))
	
	data = FermiHubbardData(GL, GG, FL, FG)
	U₀ = 10.
	tmax = 16;
	model = FermiHubbardModel(U = t -> U₀ * (1 + exp(-10(t - 1.)))^(-1))
	# model = FermiHubbardModel(U = t -> 0.5U₀ * (1 + sign(t-1)));
	model = FermiHubbardModel(U = t -> -U₀ * [(-1)^k * (1 + exp(-20(t - 2k)))^(-1) for k in 1:tmax-1] |> sum);
	# model = FermiHubbardModel(U = t -> -0.5U₀ * [(-1)^k * (1 + sign(t-k)) for k in 1:tmax-1] |> sum);
	# model = FermiHubbardModel(U = t -> 0.);
end

# ╔═╡ 96571bb6-8413-4e52-9d96-efdabe6ac864
begin
	t_vals = 0:0.01:tmax
	fig = figure(figsize=(3, 2))
	plot(t_vals, model.U.(t_vals))
	xlabel("\$t\$")
	ylabel("\$U(t)\$")
	tight_layout()
	fig
end

# ╔═╡ 1fc3b692-5989-4961-89a2-a229682d461f
begin
	atol = 1e-5
	rtol = 1e-3
end

# ╔═╡ b3722fac-563c-4e79-8848-09103bb8c00a
@time sol = kbsolve!(
    (x...) -> fv!(model, data, x...),
    (x...) -> fd!(model, data, x...),
    [data.GL, data.GG, data.FL, data.FG],
    (0.0, tmax);
    callback = (x...) -> self_energies!(model, data, x...),
    atol = atol,
    rtol = rtol,
);

# ╔═╡ cc88f5dd-4999-4db3-b675-e3003c209690
save("quenched_FH_3D_sol_U_"*string(model.U)*"_tmax_"*string(tmax)*"_atol_"*string(atol)*"_rtol_"*string(rtol)*".jld", "solution", sol)

# ╔═╡ f45264e1-0a66-4f18-b19d-844f4f1e1049
loaded_sol = load("quenched_FH_3D_sol_U_"*string(model.U)*"_tmax_"*string(tmax)*"_atol_"*string(atol)*"_rtol_"*string(rtol)*".jld");

# ╔═╡ 72adcafe-cf35-4a99-8078-e354eff4e773
let
	tt = loaded_sol["solution"].t
	fig = figure(figsize=(8, 4))
	ax = subplot(221)
	ax.plot(tt[2:end], map(t -> t[2] - t[1], zip(tt[1:end-1], tt[2:end])), "-s", ms=3, markeredgecolor="#22577c")
	ax.set_xticks([0, 4, 8, 12, 16])
	xlim(0, tmax)
	ylim(0, 0.2)
	ax.set_xticklabels([])
	ylabel("\$J h\$")
	
	ax = subplot(223)
	ax.plot(t_vals, model.U.(t_vals), "-k")
	ax.set_xticks([0, 4, 8, 12, 16])
	xlim(0, tmax)
	xlabel("\$Jt\$")
	ylabel("\$U(t)\$")

	# quantum number to look at
	idx = 1
	
	ρτ, (τs, ts) = wigner_transform_itp((loaded_sol["solution"].u[2].data[idx, idx, :, :] - loaded_sol["solution"].u[1].data[idx, idx, :, :]), 
	    loaded_sol["solution"].t[1:end], fourier=false);
	ρω, (ωs, ts) = wigner_transform_itp((loaded_sol["solution"].u[2].data[idx, idx, :, :] - loaded_sol["solution"].u[1].data[idx, idx, :, :]), 
	    loaded_sol["solution"].t[1:end], fourier=true);

	t_scale = 1
	ω_scale = 1;

	function meshgrid(xin,yin)
	  nx=length(xin)
	  ny=length(yin)
	  xout=zeros(ny,nx)
	  yout=zeros(ny,nx)
	  for jx=1:nx
	      for ix=1:ny
	          xout[ix,jx]=xin[jx]
	          yout[ix,jx]=yin[ix]
	      end
	  end
	  return (x=xout, y=yout)
	end
	
	Y, X = meshgrid(loaded_sol["solution"].t, loaded_sol["solution"].t);

	cmap = "gist_heat";

	# fig = figure(figsize=(7, 3))
	t_scale = 1
	vmin = -1.0
	vmax = 1.0
	
	ax = subplot(122)
	heatmap = ax.pcolormesh(X, Y, imag(loaded_sol["solution"].u[1].data[1, 1, :, :]) .- imag(loaded_sol["solution"].u[2].data[1, 1, :, :]), cmap=cmap, rasterized=true, vmin=vmin, vmax=vmax)
	heatmap.set_edgecolor("face")
	ax.set_aspect("equal")
	cbar = colorbar(mappable=heatmap)
	cbar.formatter.set_powerlimits((0, 0))
	ax.set_xlabel("\$J t\$")
	ax.set_ylabel("\$J t'\$")
	ax.set_xlim(0, t_scale * tmax)
	ax.set_ylim(0, t_scale * tmax)
	ax.set_xlim(0, t_scale * 8)
	ax.set_ylim(0, t_scale * 8)
	# ax.set_xticks(t_scale .* [0, tmax/2, tmax])
	# ax.set_yticks(t_scale .* [0, tmax/2, tmax])
	
	ax.set_xticks(t_scale .* [0, 2, 4, 6, 8])
	ax.set_yticks(t_scale .* [0, 2, 4, 6, 8])
	
	tight_layout(pad=0.75, w_pad=0.5, h_pad=0)
	savefig("quenched_fermi_hubbard_example_two_times.pdf")
	fig
end

# ╔═╡ Cell order:
# ╠═c925a3be-ab56-11ec-1b59-9d92ed462b9b
# ╟─188dbc64-b2f4-4486-a849-9dcf1cd5cb4a
# ╠═582a1fa3-bb83-4751-8feb-3c1ad577aed6
# ╠═bdf0d70e-7c3b-423c-aee7-e222164cdee9
# ╠═fce4c81d-3091-41cf-8005-c54bff63c825
# ╟─fc979ae4-1421-47a2-b8f4-ca69cbca2a3d
# ╠═8d547d1b-b4b0-4467-a3aa-1499d358e5ee
# ╠═96571bb6-8413-4e52-9d96-efdabe6ac864
# ╠═1fc3b692-5989-4961-89a2-a229682d461f
# ╠═b3722fac-563c-4e79-8848-09103bb8c00a
# ╠═cc88f5dd-4999-4db3-b675-e3003c209690
# ╠═f45264e1-0a66-4f18-b19d-844f4f1e1049
# ╠═12610b5c-5508-4dcb-a4ff-252eb654f1a2
# ╠═72adcafe-cf35-4a99-8078-e354eff4e773
