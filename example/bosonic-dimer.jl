### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ 808dc0ac-aabe-11ec-22c0-0fb52d386c36
begin
	using Pkg; Pkg.activate()
	using KadanoffBaym
	using LinearAlgebra
	
	using FFTW, Interpolations
	
	using PyPlot
	PyPlot.plt.style.use("./paper.mplstyle")
	using LaTeXStrings
end

# ╔═╡ fd801ca5-eeb2-4659-88ab-f49779b9f60e
begin
	# quantum numbers
	dim = 2

	# Allocate the initial Green functions (time arguments at the end)
	GL = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)
	GG = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)
	
	# initial condition
	GL[1, 1] = -im * diagm([0.0, 2])
	GG[1, 1] = -im * I(2) + GL[1,1]
end;

# ╔═╡ 9102675f-ccbf-4b35-a125-02a9f0865ecf
begin
	# Non-Hermitian Hamiltonian and jump operator
	ω₁ = 2.5
	ω₂ = 0.0
	J = pi / 4

	γ = 1
	
	N₁ = 1.
	N₂ = 0.1
	
	H = [ω₁ - 0.5im * ((N₁ + 1) + N₁ * γ) J; J ω₂ - 0.5im * ((N₂ + 1) + N₂ * γ)]
	
	function fv!(out, times, t, t′)
	    out[1] = -1.0im * (H * GL[t, t′] + [[1.0im * N₁ * γ, 0] [0, 1.0im * N₂ * γ]] * GL[t, t′])
	    out[2] = -1.0im * (adjoint(H) * GG[t, t′] - 1.0im * [[(N₁ + 1), 0] [0, (N₂ + 1)]] * GG[t, t′])
	end
	
	function fd!(out, times, t, t′)
	    out[1] = (-1.0im * (H * GL[t, t] - GL[t, t] * adjoint(H)
	             + 1.0im * γ * [[N₁ * (GL[1, 1, t, t] + GG[1, 1, t, t]), (N₁ + N₂) * (GL[2, 1, t, t] + GG[2, 1, t, t]) / 2] [(N₁ + N₂) * (GL[1, 2, t, t] + GG[1, 2, t, t]) / 2, N₂ * (GL[2, 2, t, t] + GG[2, 2, t, t])]])
	             )
	    out[2] = (-1.0im * (adjoint(H) * GG[t, t] - GG[t, t] * H
	             - 1.0im * [[(N₁ + 1) * (GL[1, 1, t, t] + GG[1, 1, t, t]), (N₁ + N₂ + 2) * (GG[2, 1, t, t] + GL[2, 1, t, t]) / 2] [(N₁ + N₂ + 2) * (GG[1, 2, t, t] + GL[1, 2, t, t]) / 2, (N₂ + 1) * (GL[2, 2, t, t] + GG[2, 2, t, t])]])
	             )
	end
end

# ╔═╡ 0b7cdcc0-d537-4001-9d39-f3a09e71e565
begin
	# call the solver
	sol = kbsolve!(fv!, fd!, [GL, GG], (0.0, 32.0); atol=1e-6, rtol=1e-4);
end;

# ╔═╡ 226e5c11-a358-4dce-b5e2-54b39cf47878
begin
	ρ_11_wigner, (taus, ts) = wigner_transform_itp((GG.data - GL.data)[1, 1, :, :], sol.t; fourier=false);
	
	ρ_22_wigner, (taus, ts) = wigner_transform_itp((GG.data - GL.data)[2, 2, :, :], sol.t; fourier=false);
	
	ρ_11_FFT, (ωs, ts) = wigner_transform_itp((GG.data - GL.data)[1, 1, :, :], sol.t; fourier=true);
	
	ρ_22_FFT, (ωs, ts) = wigner_transform_itp((GG.data - GL.data)[2, 2, :, :], sol.t; fourier=true);
end;

# ╔═╡ f1c21946-ed89-4bfe-9f3b-e59ac1bc20db
let
	function meshgrid(xin, yin)
	  nx=length(xin)
	  ny=length(yin)
	  xout=zeros(ny, nx)
	  yout=zeros(ny, nx)
	  for jx=1:nx
	      for ix=1:ny
	          xout[ix, jx]=xin[jx]
	          yout[ix, jx]=yin[ix]
	      end
	  end
	  return (x=xout, y=yout)
	end

	steps = 1
	cmap = "gist_heat";
	
	Y, X = meshgrid(sol.t[1:steps:end], sol.t[1:steps:end]);
	
	xpad = 8
	ypad = 5
	
	fig = figure(figsize=(7, 3))
	
	ax = subplot(121)
	
	plot(sol.t, [-imag(GL.data[1, 1, k, k]) for k in 1:length(sol.t)], ls="--", c="C3", label=L"i=1", lw=1.5)
	plot(sol.t, [-imag(GL.data[2, 2, k, k]) for k in 1:length(sol.t)], ls="-", c="C0", label=L"i=2", lw=1.5)

	T = sol.t[end]
	
	ax.set_xlim(0, T/2)
	ax.set_xticks([0, T/4, T/2])
	ax.set_ylim(0, 2.0)
	ax.set_xlabel(L"\lambda T")
	ax.set_ylabel(L"-\mathrm{Im}\; G^<_{ii}(t, t)")
	ax.xaxis.set_tick_params(pad=xpad)
	ax.yaxis.set_tick_params(pad=ypad)
	ax.legend(loc="best", handlelength=1.9, frameon=false, borderpad=0, labelspacing=0.25)
	
	ax = subplot(122)
	X, Y = meshgrid(ts[1:steps:end], taus[1:steps:end]);
	vmin = 1.0
	vmax = -0.5
	
	heatmap = ax.pcolormesh(X, Y, -ρ_11_wigner[1:steps:end, 1:steps:end] |> imag, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=true)
	heatmap.set_edgecolor("face")
	
	ax.set_xlabel(L"\lambda T")
	ax.set_ylabel(L"\lambda \tau")
	ax.set_xlim(0, T)
	ax.set_xticks([0, T/2, T])
	ax.set_ylim(-T/2, T/2)
	ax.set_yticks([-T/2, 0, T/2])
	colorbar(mappable=heatmap)
	ax.set_aspect("equal")
	
	tight_layout()
	# savefig("boson_example_1.pdf")
	fig
end

# ╔═╡ ad178852-d5b6-4c56-bcc3-d9b325366112
let
	xpad = 8
	ypad = 5

	T = sol.t[end]
	
	fig = figure(figsize=(7, 3))
	
	ax = subplot(121)
	plot(taus, -ρ_11_wigner[:, Int(floor(length(taus)/2))] |> imag, ls="--", c="C3", label=L"i=1", lw=1.5) # fixed T
	plot(taus, -ρ_22_wigner[:, Int(floor(length(taus)/2))] |> imag, ls="-", c="C0", label=L"i=2", lw=1.5)
	ax.set_xlabel(L"\lambda \tau")
	ax.set_xlim(-T/2, T/2)
	ax.set_ylim(-0.5, 1.0)
	ax.set_xticks([-T/2, -T/4, 0, T/4, T/2])
	ax.xaxis.set_tick_params(pad=xpad)
	ax.yaxis.set_tick_params(pad=ypad)
	ax.set_ylabel(L"-\textrm{Im}\, A_{ii}(T, \tau)_W")
	ax.legend(loc="best", handlelength=1.4, frameon=false, borderpad=0, labelspacing=0.25)
	
	ax = subplot(122)
	plot(ωs, -ρ_11_FFT[:, Int(floor(length(taus)/2))] |> imag, ls="--", c="C3", lw=1.5)
	plot(ωs, -ρ_22_FFT[:, Int(floor(length(taus)/2))] |> imag, ls="-", c="C0", lw=1.5)
	ax.set_xlabel(L"\omega/\lambda")
	ax.set_xlim(10 .* (-1, 1))
	ax.set_xticks([-10, -5, 0, 5, 10])
	ax.xaxis.set_tick_params(pad=xpad)
	ax.yaxis.set_tick_params(pad=ypad)
	ax.set_ylabel(L"-\textrm{Im}\, A_{ii}(T, \omega)_{\tilde{W}}", labelpad=16)
	ax.yaxis.set_label_position("right")
	ax.yaxis.set_ticks_position("both")
	
	tight_layout(pad=0.1, w_pad=0.5, h_pad=0)
	# savefig("boson_example_2.pdf")
	fig
end

# ╔═╡ Cell order:
# ╠═808dc0ac-aabe-11ec-22c0-0fb52d386c36
# ╠═fd801ca5-eeb2-4659-88ab-f49779b9f60e
# ╠═9102675f-ccbf-4b35-a125-02a9f0865ecf
# ╠═0b7cdcc0-d537-4001-9d39-f3a09e71e565
# ╠═226e5c11-a358-4dce-b5e2-54b39cf47878
# ╠═f1c21946-ed89-4bfe-9f3b-e59ac1bc20db
# ╠═ad178852-d5b6-4c56-bcc3-d9b325366112
