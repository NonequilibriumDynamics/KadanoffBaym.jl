### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ 0315adb0-aac3-11ec-1e7c-c3dd091b47d2
begin
	using Pkg; Pkg.activate()
	using KadanoffBaym
	using LinearAlgebra

	using FFTW, Interpolations

	using PyPlot
	PyPlot.plt.style.use("./paper.mplstyle")
	using LaTeXStrings
end

# ╔═╡ eb35f164-b059-4e71-ae29-a48a8e5e81f3
begin
	# Self-energy
	function self_energies!(data, t1, t2)
	    (; GL, GG, ΣL, ΣG, L, H, J) = data
	
	    # adjust array size
	    if (n = size(GL, 3)) > size(ΣL, 3)
	        resize!(ΣL, n)
	        resize!(ΣG, n)
	    end
	  
	    # index mapping
	    NNs = [1, 0] # nearest neighbours
	    N_mu = mu -> NNs[mu % L + 1] + mu - mu % L
	    idxs = [[(mu + L) % 2L, N_mu(mu), (N_mu(mu) + L) % 2L] .+ 1 for mu in 0:2L-1]
	    idxs = hcat(idxs...)
	    
	    # self-energies using the index mapping
	    ΣL[t1, t2] = -J^2 * diag(GL[t1, t2])[idxs[1, :]] .* diag(GL[t1, t2])[idxs[2, :]] .* diag(GG[t2, t1])[idxs[3, :]] |> diagm
	    ΣG[t1, t2] = -J^2 * diag(GG[t1, t2])[idxs[1, :]] .* diag(GG[t1, t2])[idxs[2, :]] .* diag(GL[t2, t1])[idxs[3, :]] |> diagm
	end
end

# ╔═╡ 4e89378c-804b-4b12-a06e-7afc1b4fd753
begin
	# Container for problem data
	struct ProblemData
	    GL::GreenFunction{ComplexF64, 4, Array{ComplexF64, 4}, SkewHermitian}
	    GG::GreenFunction{ComplexF64, 4, Array{ComplexF64, 4}, SkewHermitian}
	    ΣL::GreenFunction{ComplexF64, 4, Array{ComplexF64, 4}, SkewHermitian}
	    ΣG::GreenFunction{ComplexF64, 4, Array{ComplexF64, 4}, SkewHermitian}
	    L::Int64
	    H::Matrix{ComplexF64}
	    J::Float64
	  
	    # Initialize problem
	    function ProblemData(GL0, L, H, J)
	        @assert H == H' "Non-hermitian Hamiltonian"
	
	        data = new(
	          GreenFunction(reshape(GL0, size(GL0)..., 1, 1), SkewHermitian),
	          GreenFunction(reshape(GL0 - 1.0im * I, size(GL0)..., 1, 1), SkewHermitian),
	          GreenFunction(zeros(ComplexF64, size(GL0)..., 1, 1), SkewHermitian),
	          GreenFunction(zeros(ComplexF64, size(GL0)..., 1, 1), SkewHermitian),
	          L,
	          H,
	          J
	        )
	
	        # Initialize self-energies
	        self_energies!(data, 1, 1)
	
	        return data
	    end
	end
end

# ╔═╡ 7d664f41-5fff-4c14-8cc2-ffd8844166f8
# problem data
data = begin

    # parameters
    L = 2
    h = 5.0
    H_ = ComplexF64[-h 0 0 0; 0 -h 0 0; 0 0 h 0; 0 0 0 h];
    J = 1.

    # initial condition
    delta_n = 0.1
    GL0 = -1.0im .* 2e-1 .* [2 0 0 0; 0 1 0 0; 0 0 2 0; 0 0 0 0.5]

    ProblemData(GL0, L, H_, J)
end;

# ╔═╡ 7d7f9775-926e-4bc3-88f9-80ff755e7606
begin
	atol = 1e-8
	rtol = 1e-6
end;

# ╔═╡ 8bf9b261-98b9-4ece-bf78-49b80cd012f8
begin
	# right-hand side for the "vertical" evolution
	function fv!(out, data, ts, t1, t2)
	  (; GL, GG, ΣL, ΣG, L, H) = data
	  
	  # real-time collision integral
	  ∫dt(i, j, A, B) = sign(j-i) * integrate(ts[min(i, j):max(i, j)], [A[t1, t] * B[t, t2] for t=min(i, j):max(i, j)])
	
	  out[1] = -1.0im * (H * GL[t1, t2] + ∫dt(1, t1, ΣG, GL) - ∫dt(1, t1, ΣL, GL) + ∫dt(1, t2, ΣL, GL) - ∫dt(1, t2, ΣL, GG))
	  out[2] = -1.0im * (H * GG[t1, t2] + ∫dt(1, t1, ΣG, GG) - ∫dt(1, t1, ΣL, GG) + ∫dt(1, t2, ΣG, GL) - ∫dt(1, t2, ΣG, GG))
	end
	
	# right-hand side for the "diagonal" evolution
	function fd!(out, data, ts, t1, t2)
	  fv!(out, data, ts, t1, t2)
	  out .-= adjoint.(out)
	end

	# trapezoid method (for higher-order methods, see below)
	function integrate(x::AbstractVector, y::AbstractVector)
	    if isone(length(x))
	        return zero(first(y))
	    end
	
	    @inbounds retval = (x[2] - x[1]) * (y[1] + y[2])
	    @inbounds @fastmath @simd for i in 2:(length(y) - 1)
	        retval += (x[i+1] - x[i]) * (y[i] + y[i+1])
	    end
	    return 1//2 * retval
	end;

	# call the solver
	sol = kbsolve!(
	  (out, x...) -> fv!(out, data, x...), 
	  (out, x...) -> fd!(out, data, x...), 
	  [data.GL, data.GG], 
	  (0.0, 5.0); 
	  callback = (_, x...) -> self_energies!(data, x...), 
	  atol=atol, 
	  rtol=rtol);
end;

# ╔═╡ 28c479e9-4e61-4f69-aa5e-00cefc9e74e9
begin
	using PyCall
	qt = pyimport("qutip")
	
	# time parameters
	times = range(first(sol.t), stop=last(sol.t), length=2^7+1)
	n = length(times) - 1
	
	n_max = 2; # Fock-space truncation
	
	# initial state
	psi0 = qt.tensor([(sqrt(1 - 1.0im * data.GL[1, 1][k, k]) * qt.basis(n_max + 1, 0) 
	            + sqrt(1.0im * data.GL[1, 1][k, k]) * qt.basis(n_max + 1, 1)).unit() for k in 1:2*L]);
	
	# operators
	ids = [qt.qeye(n_max + 1), qt.qeye(n_max + 1), qt.qeye(n_max + 1), qt.qeye(n_max + 1)]
	ops = [deepcopy(ids) for _ in 1:4]
	
	for (i, op) in enumerate(ops)
	    op[i] = qt.destroy(n_max + 1)
	end
	
	ops = qt.tensor.(ops)
	b1, b2, a1, a2 = ops;
	
	# make diagonal density matrix (i.e. dropping coherences)
	rho0 = psi0 * psi0.dag();
	for j in 1:rho0.shape[2] , i in 1:rho0.shape[1]
	    i != j ? rho0.data[i, j] = 0.0 : continue
	end
	
	# Hamiltonian (hard-coded for two sites)
	H  = -h * b1.dag() * b1
	H += -h * b2.dag() * b2
	H += +h * a1.dag() * a1
	H += +h * a2.dag() * a2
	H += J * (a1.dag() * b1 * b2.dag() * a2 + b1.dag() * a1 * a2.dag() * b2)
	
	# observables
	obs = [b1.dag() * b1, b2.dag() * b2, a1.dag() * a1, a2.dag() * a2];

	opts = qt.Options(atol=atol, rtol=rtol, nsteps=5_000)
	
	# quickly solve once for observables
	me = qt.mesolve(H, rho0, times, [], obs, options=opts)

	# solve for the time-dependent density matrix
	t_sols = qt.mesolve(H, rho0, times, options=opts); # t_sols.states returns density matrices
	
	# compute two-time average -i<X(t2)Y(t1)>
	function two_time_average(X, Y, rho_t1)
	    X_t2_Y_t1 = zeros(ComplexF64, n + 1, n + 1)
	    for k in 1:(n + 1)
	        Y_rho_t1 = qt.mesolve(H, Y * rho_t1.states[k], times, options=opts).states
	        for l in 1:(n + 1)
	            X_t2_Y_t1[k, l] = -1.0im * (X * Y_rho_t1[l]).tr()    
	        end
	    end
	    
	    # transform to regular two-time square used by the KB solver
	    rotated_X_t2_Y_t1 = zeros(ComplexF64, n + 1, 2*(n + 1) - 1)
	    for (k, x) in enumerate([X_t2_Y_t1[k, :] for k in 1:(n + 1)])
	        for (l, y) in enumerate(x)
	            ind = k + l - 1
	            rotated_X_t2_Y_t1[k, ind] = y 
	        end
	    end    
	    
	    return rotated_X_t2_Y_t1[:, 1:n+1]
	end;
	
	create_destroy = [zeros(ComplexF64, n + 1, n + 1) for _ in 1:1]#2L]
	destroy_create = [zeros(ComplexF64, n + 1, n + 1) for _ in 1:1]#2L];
	
	destroyers = [b1, b2, a1, a2]
	for k in 1:1#2L
	    create_destroy[k] = two_time_average(destroyers[k].dag(), destroyers[k], t_sols)
	    destroy_create[k] = two_time_average(destroyers[k], destroyers[k].dag(), t_sols)
	    
	    # flip real part when creation operator is to the right (equivalent to flipping the sign of tau)
	    destroy_create[k] = -conj(destroy_create[k])
	end
	
	b1_dag_b1 = create_destroy[1]
	b1_b1_dag = destroy_create[1];
	
	# compute missing two-time triangle from symmetry
	full_square = A -> (A - adjoint(A) - (A |> diag |> diagm));
end

# ╔═╡ 8c12998a-d3d3-446a-94cd-89ec391fe197
# begin
# 	# Vertical rhs
# 	function fv!(out, data, v1, v2, ts, t1, t2)
# 	  @unpack GL, GG, ΣL, ΣG, L, H = data
	
# 	  out[1] = -1.0im * (H * GL[t1, t2] + v1[1] + v2[1])
# 	  out[2] = -1.0im * (H * GG[t1, t2] + v1[2] + v2[2])
# 	end
	
# 	function kv1!(out, data, ts, t1, t2, τ)
# 	    @unpack GL, GG, ΣL, ΣG, L, H = data
	    
# 	    out[1] = ΣG[t1, τ] .* GL[τ, t2] - ΣL[t1, τ] .* GL[τ, t2]
# 	    out[2] = ΣG[t1, τ] .* GG[τ, t2] - ΣL[t1, τ] .* GG[τ, t2]                      
# 	end
	
# 	function kv2!(out, data, ts, t1, t2, τ)
# 	    @unpack GL, GG, ΣL, ΣG, L, H = data
	    
# 	    out[1] = ΣL[t1, τ] .* GL[τ, t2] - ΣL[t1, τ] .* GG[τ, t2]
# 	    out[2] = ΣG[t1, τ] .* GL[τ, t2] - ΣG[t1, τ] .* GG[τ, t2]   
# 	end
	
	
# 	# Diagonal rhs
# 	function fd!(out, data, v1, v2, ts, t1, t2)
# 	  @unpack GL, GG, ΣL, ΣG, L, H = data
	    
# 	  fv!(out, data, v1, v2, ts, t1, t2)
# 	  out .-= adjoint.(out)
# 	end
	
# 	function kd1!(out, data, ts, t1, t2, τ)
# 	    kv1!(out, data, ts, t1, t2, τ)
# 	end
	
# 	function kd2!(out, data, ts, t1, t2, τ)
# 	    kv2!(out, data, ts, t1, t2, τ)
# 	end
	
# 	# Integration
# 	sol = kbsolve!(
# 	  (out, x...) -> fv!(out, data, x...), 
# 	  (out, x...) -> (println(" t: $(x[3][x[4]])"); fd!(out, data, x...)), 
# 	  [data.GL, data.GG], 
# 	  tspan;
# 	  kv1! = (out, x...) -> kv1!(out, data, x...),
# 	  kv2! = (out, x...) -> kv2!(out, data, x...),
# 	  kd1! = (out, x...) -> kv1!(out, data, x...),
# 	  kd2! = (out, x...) -> kv2!(out, data, x...),
# 	  callback = (_, x...) -> self_energies!(data, x...), 
# 	  atol=1e-10, 
# 	  rtol=1e-8);

# end

# ╔═╡ 016355b7-584b-40cf-a019-2f582b5a27bc
let
	fig = figure(figsize=(6, 2))
	subplot(121)
	imshow(real(full_square(b1_dag_b1)), cmap="plasma")
	
	subplot(122)
	imshow(real(full_square(b1_b1_dag)), cmap="plasma")
	
	tight_layout()
	fig
end

# ╔═╡ 8ec47511-a470-4dae-8dc0-eced566f67d7
let
	# final time
	T = sol.t[end]
	
	xpad = 8
	ypad = 5
	
	fig = figure(figsize=(7, 3))
	
	ax = subplot(121)
	idx = 1
	ax.plot(sol.t, data.GL.data[idx, idx ,:,:] |> ((-) ∘ imag ∘ diag), ls="-", label="\$i="*string((idx - 1) % 2 + 1)*"\$", c="C0")
	ax.plot(times, me.expect[idx], "--", c="C0", lw=3, alpha=0.5)
	idx = 2
	ax.plot(sol.t, data.GL.data[idx, idx ,:,:] |> ((-) ∘ imag ∘ diag), ls=":", label="\$i="*string((idx - 1) % 2 + 1)*"\$", c="C1")
	ax.plot(times, me.expect[idx], "-.", c="C1", lw=3, alpha=0.5)
	ax.set_xlabel("\$Jt\$") 
	ylabel("\$-\\mathrm{Im}\\; \\mathcal{A}^<_{i,\\, i}(t, t)\$", labelpad=10)
	ax.set_xticks([0, 2.5, 5])
	ax.set_xlim(0, sol.t[end]) 
	ax.set_ylim(0, 0.5)
	ax.legend(loc="best", handlelength=1.4, frameon=false, borderpad=0, labelspacing=0.25)
	
	ax = subplot(122)
	idx = 3
	ax.plot(sol.t, data.GL.data[idx, idx ,:,:] |> ((-) ∘ imag ∘ diag), ls="-", label="\$i="*string((idx - 1) % 2 + 1)*"\$", c="C2")
	ax.plot(times, me.expect[idx], "--", c="C2", lw=3, alpha=0.5)
	idx = 4
	ax.plot(sol.t, data.GL.data[idx, idx ,:,:] |> ((-) ∘ imag ∘ diag), ls=":", label="\$i="*string((idx - 1) % 2 + 1)*"\$", c="C3")
	ax.plot(times, me.expect[idx], "-.", c="C3", lw=3, alpha=0.5)
	ax.set_xlabel("\$Jt\$") 
	ylabel("\$-\\mathrm{Im}\\; \\mathcal{B}^<_{i,\\, i}(t, t)\$", labelpad=16)
	ax.yaxis.set_label_position("right")
	ax.set_xticks([0, 2.5, 5])
	ax.set_yticklabels([])
	ax.set_xlim(0, sol.t[end]) 
	ax.set_ylim(0, 0.5)
	ax.legend(loc="best", handlelength=1.4, frameon=false, borderpad=0, labelspacing=0.25)
	
	tight_layout(pad=0.25, w_pad=1, h_pad=0)
	# savefig("interacting_bosons_example_1.pdf")
	fig
end

# ╔═╡ 0463eb3d-0310-4975-99ba-09b4e1d3330d
	let
	# quantum number to look at
	idx = 1;
	T = sol.t[end]
	
	ρ_11_kb = interpolate((sol.t, sol.t), view(data.GG.data .- data.GL.data, idx, idx, :, : ), Gridded(Linear()));
	ρ_11_qt = interpolate((times, times), view(full_square(destroy_create[idx] - create_destroy[idx]), :, : ), Gridded(Linear()));
	
	new_times = range(first(sol.t), stop=last(sol.t), length=2048);
	
	ρ_11_kb_wigner, (taus, ts) = wigner_transform([ρ_11_kb(t1, t2) for t1 in new_times, t2 in new_times]; ts=new_times, fourier=false);
	ρ_11_qt_wigner, (taus_qt, ts_qt) = wigner_transform([ρ_11_qt(t1, t2) for t1 in new_times, t2 in new_times]; ts=new_times, fourier=false);

	xpad = 8
	ypad = 5
	
	center = floor(length(new_times) / 2) |> Int
	
	fig = figure(figsize=(7, 3))
	
	ax = subplot(121)
	plot(taus, ρ_11_kb_wigner[:, center] |> real, ls="-", c="C0", lw=1.5)
	plot(taus_qt, ρ_11_qt_wigner[:, center] |> real, ls="--", c="C0", lw=2.5, alpha=0.5)
	
	ax.xaxis.set_tick_params(pad=xpad)
	ax.yaxis.set_tick_params(pad=ypad)
	ax.set_xlabel(L"J \tau")
	ax.set_ylabel(L"\mathrm{Re}\,A_{\mathcal{A}_{1,\, 1}}(T, \tau)_W")
	ax.set_xlim(-T, T)
	ax.set_ylim(-1.0, 1.0)
	
	ax = subplot(122)
	plot(taus, -ρ_11_kb_wigner[:, center] |> imag, ls="-", c="C0", lw=1.5)
	plot(taus_qt, -ρ_11_qt_wigner[:, center] |> imag, ls="--", c="C0", lw=2.5, alpha=0.5)
	ax.set_xlabel(L"J \tau")
	ax.set_xlim(-T, T)
	ax.set_ylim(-1.0, 1.0)
	ax.set_yticklabels([])
	ax.xaxis.set_tick_params(pad=xpad)
	ax.yaxis.set_tick_params(pad=ypad)
	ax.set_ylabel(L"-\mathrm{Im}\,A_{\mathcal{A}_{1,\, 1}}(T, \tau)_W", labelpad=16)
	ax.yaxis.set_label_position("right")
	
	tight_layout(pad=0.1, w_pad=1, h_pad=0)
	# savefig("interacting_bosons_example_2.pdf")
	fig
end

# ╔═╡ Cell order:
# ╠═0315adb0-aac3-11ec-1e7c-c3dd091b47d2
# ╠═4e89378c-804b-4b12-a06e-7afc1b4fd753
# ╠═eb35f164-b059-4e71-ae29-a48a8e5e81f3
# ╠═7d664f41-5fff-4c14-8cc2-ffd8844166f8
# ╠═7d7f9775-926e-4bc3-88f9-80ff755e7606
# ╠═8bf9b261-98b9-4ece-bf78-49b80cd012f8
# ╠═8c12998a-d3d3-446a-94cd-89ec391fe197
# ╠═28c479e9-4e61-4f69-aa5e-00cefc9e74e9
# ╠═016355b7-584b-40cf-a019-2f582b5a27bc
# ╠═8ec47511-a470-4dae-8dc0-eced566f67d7
# ╠═0463eb3d-0310-4975-99ba-09b4e1d3330d
