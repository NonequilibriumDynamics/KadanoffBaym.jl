var documenterSearchIndex = {"docs":
[{"location":"#Documentation","page":"Documentation","title":"Documentation","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"KadanoffBaym.jl is the first fully adaptive solver for Kadanoff-Baym equations written in Julia. ","category":"page"},{"location":"","page":"Documentation","title":"Documentation","text":"tip: Tip\nTo learn more about the solver and Kadanoff-Baym equations, have a look into our accompanying paper.","category":"page"},{"location":"#Installation","page":"Documentation","title":"Installation","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"To install, use Julia's built-in package manager","category":"page"},{"location":"","page":"Documentation","title":"Documentation","text":"julia> ] add KadanoffBaym","category":"page"},{"location":"","page":"Documentation","title":"Documentation","text":"The most recent version of KadanoffBaym.jl requires Julia v1.7 or later.","category":"page"},{"location":"#Examples","page":"Documentation","title":"Examples","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"For now, please see the examples folder.","category":"page"},{"location":"#Library","page":"Documentation","title":"Library","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"KadanoffBaym.jl was designed to be lean and simple and hence only exports a handful of functions, namely GreenFunction (together with two possible time symmetries, Symmetrical and SkewHermitian) and the integrator kbsolve!.","category":"page"},{"location":"","page":"Documentation","title":"Documentation","text":"note: Note\nWhen you import the external packages FFTW and Interpolations, KadanoffBaym.jl will also export the functions wigner_transform and wigner_transform_itp to perform Wigner transformations.","category":"page"},{"location":"#Index","page":"Documentation","title":"Index","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"","category":"page"},{"location":"#Solver","page":"Documentation","title":"Solver","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"kbsolve!(fv!, fd!, u0::Vector{<:GreenFunction}, (t0, tmax))","category":"page"},{"location":"#KadanoffBaym.kbsolve!-Tuple{Any, Any, Vector{<:GreenFunction}, Any}","page":"Documentation","title":"KadanoffBaym.kbsolve!","text":"kbsolve!(fv!, fd!, u0, (t0, tmax); ...)\n\nSolves the 2-time Voltera integro-differential equation\n\ndu(t_1t_2)  dt_1 = f_v(t_1t_2) = vut_1t_2 + _t0^t1 dτ K_1^vut_1t_2τ + _t0^t2 dτ K_2^vut_1t_2τ\n\ndu(t_1t_2)  dt_2 = f_h(t_1t_2) = hut_1t_2 + _t0^t1 dτ K_1^hut_1t_2τ + _t0^t2 dτ K_2^hut_1t_2τ\n\nfor 2-point functions u0 from t0 to tmax.\n\nParameters\n\nfv!(out, ts, w1, w2, t1, t2): The right-hand side of dudt_1 at indices (t1, t2)  on the time-grid (ts x ts). The weights w1 and w2 can be used to integrate the Volterra kernels K1v and K2v as sum_i w1_i K1v_i and  sum_i w2_i K2v_i, respectively. The output is saved in-place in out, which has the same shape as u0.\nfd!(out, ts, w1, w2, t1, t2): The right-hand side of (dudt_1 + dudt_2)_t_2  t_1\nu0::Vector{<:GreenFunction}: List of 2-point functions to be time-stepped\n(t0, tmax): A tuple with the initial time(s) t0 – can be a vector of  past times – and final time tmax\n\nOptional keyword parameters\n\ncallback(ts, w1, w2, t1, t2): A function that gets called everytime the  2-point function at indices (t1, t2) is updated. Can be used to update functions which are not being integrated, such as self-energies.\nstop(ts): A function that gets called at every time-step that stops the  integration when it evaluates to true\natol::Real: Absolute tolerance (components with magnitude lower than  atol do not guarantee number of local correct digits)\nrtol::Real: Relative tolerance (roughly the local number of correct digits)\ndtini::Real: Initial step-size\ndtmax::Real: Maximal step-size\nqmax::Real: Maximum step-size factor when adjusting the time-step\nqmin::Real: Minimum step-size factor when adjusting the time-step\nγ::Real: Safety factor for the calculated time-step such that it is  accepted with a higher probability\nkmax::Integer: Maximum order of the adaptive Adams method\n\nNotes\n\nDue to high memory and computation costs, kbsolve! mutates the initial condition u0  and only works with in-place rhs functions, unlike standard ODE solvers.\nThe Kadanoff-Baym timestepper is a 2-time generalization of the variable Adams method presented in E. Hairer, S. Norsett and G. Wanner, Solving Ordinary Differential Equations I: Non- stiff Problems, vol. 8, Springer-Verlag Berlin Heidelberg, ISBN 978-3-540-56670-0, doi:10.1007/978-3-540-78862-1 (1993).\n\n\n\n\n\n","category":"method"},{"location":"#Green-Functions","page":"Documentation","title":"Green Functions","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"GreenFunction{T,N,A,U<:AbstractSymmetry}","category":"page"},{"location":"#KadanoffBaym.GreenFunction","page":"Documentation","title":"KadanoffBaym.GreenFunction","text":"GreenFunction(g::AbstractArray, s::AbstractSymmetry)\n\nA container interface for g with array indexing respecting some symmetry rule s. Because of that, g must be square in its last 2 dimensions, which can be resized  with resize!.\n\nThe array g is not restricted to being contiguous. For example, g can have Matrix{T}, Array{T,4}, Matrix{SparseMatrixCSC{T}}, etc as its type.\n\nNotes\n\nThe GreenFunction does not own g. Proper care must be taken when using multiple GreenFunctions since using the same array will result in unexpected behaviour\n\njulia> data = zeros(2,2)\njulia> g1 = GreenFunction(data, Symmetrical)\njulia> g2 = GreenFunction(data, Symmetrical)\njulia> g1[1,1] = 3\njulia> @show g2[1,1]\njulia> g1.data === g2.data # they share the same data\n\nIndexing with less indices than the dimension of g results in a  \"take-all-to-the-left\" indexing\n\njulia> gf[i,j] == gf[:,:,...,:,i,j]\njulia> gf[i,j,k] == gf[:,:,...,:,i,j,k]\n\nCustom symmetries can be implemented via multiple dispatch\n\njulia> struct MySymmetry <: KadanoffBaym.AbstractSymmetry end\njulia> @inline KadanoffBaym.symmetry(::GreenFunction{T,N,A,MySymmetry}) where {T,N,A} = conj\n\nExamples\n\nGreenFunction simply takes some data g and embeds the symmetry s in its indexing\n\njulia> time_dim = 3\njulia> spin_dim = 2\njulia> data = zeros(spin_dim, spin_dim, time_dim, time_dim)\njulia> gf = GreenFunction(data, Symmetrical)\njulia> gf[2,1] = rand(spin_dim, spin_dim)\njulia> @show gf[1,2]\njulia> @show KadanoffBaym.symmetry(gf)(gf[2,1])\n\n\n\n\n\n","category":"type"},{"location":"#Wigner-Transformation","page":"Documentation","title":"Wigner Transformation","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"wigner_transform(x::AbstractMatrix)","category":"page"},{"location":"#KadanoffBaym.wigner_transform-Tuple{AbstractMatrix}","page":"Documentation","title":"KadanoffBaym.wigner_transform","text":"wigner_transform(x::AbstractMatrix; ts=1:size(x,1), fourier=true)\n\nWigner-Ville transformation\n\nx_W(ω T) = i dt x(T + t2 T - t2) e^+i ω t\n\nor\n\nx_W(τ T) = x(T + t2 T - t2)\n\nof a 2-point function x. Returns a tuple of x_W and the corresponding axes (ω, T) or (τ, T), depending on the fourier keyword.\n\nThe motivation for the Wigner transformation is that, given an autocorrelation function x, it reduces to the spectral density function at all times T for  stationary processes, yet it is fully equivalent to the non-stationary  autocorrelation function. Therefore, the Wigner (distribution) function tells  us, roughly, how the spectral density changes in time.\n\nOptional keyword parameters\n\nts::AbstractVector: Time grid for x. Defaults to a UnitRange.\nfourier::Bool: Whether to Fourier transform. Defaults to true.\n\nNotes\n\nThe algorithm only works when ts – and consequently x – is equidistant.\n\nReferences\n\nhttps://en.wikipedia.org/wiki/Wigner_distribution_function\n\nhttp://tftb.nongnu.org\n\n\n\n\n\n","category":"method"},{"location":"","page":"Documentation","title":"Documentation","text":"wigner_transform_itp(x::AbstractMatrix, ts::Vector)","category":"page"},{"location":"#KadanoffBaym.wigner_transform_itp-Tuple{AbstractMatrix, Vector}","page":"Documentation","title":"KadanoffBaym.wigner_transform_itp","text":"wigner_transform_itp(x::AbstractMatrix, ts::Vector; fourier=true)\n\nInterpolates x on an equidistant mesh with the same boundaries and length as ts and calls wigner_transform\n\n\n\n\n\n","category":"method"},{"location":"#Citation","page":"Documentation","title":"Citation","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"If you use KadanoffBaym.jl in your research, please cite our paper:","category":"page"},{"location":"","page":"Documentation","title":"Documentation","text":"@Article{10.21468/SciPostPhysCore.5.2.030,\n\ttitle={{Adaptive Numerical Solution of Kadanoff-Baym Equations}},\n\tauthor={Francisco Meirinhos and Michael Kajan and Johann Kroha and Tim Bode},\n\tjournal={SciPost Phys. Core},\n\tvolume={5},\n\tissue={2},\n\tpages={30},\n\tyear={2022},\n\tpublisher={SciPost},\n\tdoi={10.21468/SciPostPhysCore.5.2.030},\n\turl={https://scipost.org/10.21468/SciPostPhysCore.5.2.030},\n}","category":"page"}]
}
