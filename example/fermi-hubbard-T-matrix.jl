using Pkg; Pkg.activate()

using LinearAlgebra, BlockArrays
using KadanoffBaym

using JLD

using PyPlot
PyPlot.plt.style.use("./paper.mplstyle")


# ==========================================================================================================

function fixed_point(F::Function, x0::AbstractArray; 
        mixing::Float64=0.5, 
        abstol::Float64=1e-12, 
        maxiter::Int=1000, 
        verbose::Bool=true, 
        norm=x -> LinearAlgebra.norm(x, Inf)
    )
    
    x_old = copy(x0)

    step = 0
    while step < maxiter
        x = F(x_old)
        res = norm(x - x_old)
        if verbose
            @info "step: $step // res: $res"
        end
        if res < abstol
            break
        end
        @. x_old = mixing * x + (1.0 - mixing) * x_old
        step += 1
    end

    if step == maxiter
        @warn "No convergence reached."
    end

    return x_old
end

# ==========================================================================================================

Base.@kwdef struct FermiHubbardModel
    U::Float64
    
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

# ==========================================================================================================

struct FermiHubbardData{T}
    GL::T
    GG::T
    FL::T
    FG::T
    
    TL::T
    TG::T    
    
    ΣNCA_c_L::T
    ΣNCA_c_G::T
    ΣNCA_f_L::T
    ΣNCA_f_G::T
    
    ΦL::T
    ΦG::T
    
    # Initialize problem
    function FermiHubbardData(GL::T, GG::T, FL::T, FG::T, TL::T, TG::T) where T
        new{T}(GL, GG, FL, FG, TL, TG, zero(GL), zero(GG), zero(FL), zero(FG), zero(FL), zero(FG))
    end
end

# ==========================================================================================================

function fv!(model, data, out, times, h1, h2, t, t′)
    (; GL, GG, FL, FG, TL, TG, ΣNCA_c_L, ΣNCA_c_G, ΣNCA_f_L, ΣNCA_f_G) = data
    (; H1, H2, U) = model 
    
    # real-time collision integral
    ∫dt1(A,B,C) = sum(h1[s] * ((A[t, s] - B[t, s]) * C[s, t′]) for s in 1:t)
    ∫dt2(A,B,C) = sum(h2[s] * (A[t, s] * (B[s, t′] - C[s, t′])) for s in 1:t′)
    
    ΣHF_c(t, t′) = 1.0im * U * Diagonal(FL[t, t])
    ΣHF_f(t, t′) = 1.0im * U * Diagonal(GL[t, t])
    
    out[1] = -1.0im * ((H1 + ΣHF_c(t, t′)) * GL[t, t′] + 
            ∫dt1(ΣNCA_c_G, ΣNCA_c_L, GL) + ∫dt2(ΣNCA_c_L, GL, GG)
        )

    out[2] = -1.0im * ((H1 + ΣHF_c(t, t′)) * GG[t, t′] + 
            ∫dt1(ΣNCA_c_G, ΣNCA_c_L, GG) + ∫dt2(ΣNCA_c_G, GL, GG)
        )

    out[3] = -1.0im * ((H2 + ΣHF_f(t, t′)) * FL[t, t′] + 
            ∫dt1(ΣNCA_f_G, ΣNCA_f_L, FL) + ∫dt2(ΣNCA_f_L, FL, FG)
        )

    out[4] = -1.0im * ((H2 + ΣHF_f(t, t′)) * FG[t, t′] +
            ∫dt1(ΣNCA_f_G, ΣNCA_f_L, FG) + ∫dt2(ΣNCA_f_G, FL, FG)
        )    
end

function fd!(model, data, out, times, h1, h2, t, t′)
    fv!(model, data, out, times, h1, h2, t, t)
    out[1:4] .-= adjoint.(out[1:4])
end

# ==========================================================================================================

function second_Born!(model, data, times, h1, h2, t, t′)
    (; GL, GG, FL, FG, TL, TG, ΣNCA_c_L, ΣNCA_c_G, ΣNCA_f_L, ΣNCA_f_G) = data
    (; U) = model
        
    if (n = size(GL, 3)) > size(ΣNCA_c_L, 3)
        resize!(ΣNCA_c_L, n)
        resize!(ΣNCA_c_G, n)
        resize!(ΣNCA_f_L, n)
        resize!(ΣNCA_f_G, n)
        
        resize!(TL, n)
        resize!(TG, n)
        resize!(data.ΦL, n)
        resize!(data.ΦG, n)
    end
    
    TL[t, t′] = -1.0im .* GL[t, t′] .* FL[t, t′]
    TG[t, t′] = -1.0im .* GG[t, t′] .* FG[t, t′]        
    
    ΣNCA_c_L[t, t′] = 1.0im .* U^2 .* TL[t, t′] .* transpose(FG[t′, t])
    ΣNCA_f_L[t, t′] = 1.0im .* U^2 .* TL[t, t′] .* transpose(GG[t′, t])
    
    ΣNCA_c_G[t, t′] = 1.0im .* U^2 .* TG[t, t′] .* transpose(FL[t′, t])
    ΣNCA_f_G[t, t′] = 1.0im .* U^2 .* TG[t, t′] .* transpose(GL[t′, t])
end

# ==========================================================================================================

function T_matrix!(model, data, times, h1, h2, t, t′)
    (; GL, GG, FL, FG, TL, TG, ΣNCA_c_L, ΣNCA_c_G, ΣNCA_f_L, ΣNCA_f_G) = data
    (; U) = model
        
    # real-time collision integral
    ∫dt1(A,B,C) = sum(h1[s] * ((A[t, s] - B[t, s]) * C[s, t′]) for s in 1:t)
    ∫dt2(A,B,C) = sum(h2[s] * (A[t, s] * (B[s, t′] - C[s, t′])) for s in 1:t′)
    
    if (n = size(GL, 3)) > size(ΣNCA_c_L, 3)
        resize!(ΣNCA_c_L, n)
        resize!(ΣNCA_c_G, n)
        resize!(ΣNCA_f_L, n)
        resize!(ΣNCA_f_G, n)
        
        resize!(TL, n)
        resize!(TG, n)
        resize!(data.ΦL, n)
        resize!(data.ΦG, n)
    end
    
    # need to set all Φs at the very first t′ since they are _all_ known by then
    if t′ == 1
        for t′ in 1:t
            data.ΦL[t, t′] = -1.0im .* GL[t, t′] .* FL[t, t′]
            data.ΦG[t, t′] = -1.0im .* GG[t, t′] .* FG[t, t′]
        end
    end
    
    # cached version
    TL[t, t′], TG[t, t′] = let
        ∫dt_(x...) = (x[1] < 1 || x[2] < 1) ? zero(x[3][1,1]) : ∫dt(x...)
        
        ΦL_ = data.ΦL[t, t′]
        ΦG_ = data.ΦG[t, t′]
        
        I1 = sum(h1[s] * ((data.ΦG[t, s] - data.ΦL[t, s]) * TL[s, t′]) for s in 1:(t-1); init=zero(ΦL_))
        I2 = sum(h2[s] * (data.ΦL[t, s] * (TL[s, t′] - TG[s, t′])) for s in 1:(t′-1); init=zero(ΦL_))
        I3 = sum(h1[s] * ((data.ΦG[t, s] - data.ΦL[t, s]) * TG[s, t′]) for s in 1:(t-1); init=zero(ΦL_))
        I4 = sum(h2[s] * (data.ΦG[t, s] * (TL[s, t′] - TG[s, t′])) for s in 1:(t′-1); init=zero(ΦL_))
        
        L_ = data.ΦL[t, t′] - U * (I1 + I2)
        G_ = data.ΦG[t, t′] - U * (I3 + I4)
        
        fixed_point([L_, G_]; mixing=0.5, verbose=false) do x
            TL[t, t′], TG[t, t′] = x[1], x[2]

            [
                L_ - U * (h1[t] * (data.ΦG[t,t] - data.ΦL[t,t]) * TL[t,t′] + h2[t′] * data.ΦL[t,t′] * (TL[t′,t′] - TG[t′,t′])),
                G_ - U * (h1[t] * (data.ΦG[t,t] - data.ΦL[t,t]) * TG[t,t′] + h2[t′] * data.ΦG[t,t′] * (TL[t′,t′] - TG[t′,t′]))
            ]
        end
    end
    
    ΣNCA_c_L[t, t′] = 1.0im .* U^2 .* TL[t, t′] .* transpose(FG[t′, t])
    ΣNCA_f_L[t, t′] = 1.0im .* U^2 .* TL[t, t′] .* transpose(GG[t′, t])
    
    ΣNCA_c_G[t, t′] = 1.0im .* U^2 .* TG[t, t′] .* transpose(FL[t′, t])
    ΣNCA_f_G[t, t′] = 1.0im .* U^2 .* TG[t, t′] .* transpose(GL[t′, t])
end

# ==========================================================================================================

# quantum numbers
dim = 8

# Define your Green functions at (t0, t0) – time arguments at the end
GL = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)
GG = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)
FL = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)
FG = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)

TL = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)
TG = GreenFunction(zeros(ComplexF64, dim, dim, 1, 1), SkewHermitian)

# Initial condition
N_c = zeros(dim)
N_f = zeros(dim)

N_c[1:4] = 0.1 .* [1, 1, 1, 1]
N_f[1:4] = 0.1 .* [1, 1, 1, 1]

N_c[5:8] = 0.0 .* [1, 1, 1, 1]
N_f[5:8] = 0.0 .* [1, 1, 1, 1]

GL[1, 1] = 1.0im * diagm(N_c)
GG[1, 1] = -1.0im * (I - diagm(N_c))
FL[1, 1] = 1.0im * diagm(N_f)
FG[1, 1] = -1.0im * (I - diagm(N_f))

TL[1, 1] = -1.0im .* GL[1, 1] .* FL[1, 1]
TG[1, 1] = -1.0im .* GG[1, 1] .* FG[1, 1]

data = FermiHubbardData(GL, GG, FL, FG, TL, TG)
data.ΦL[1, 1] = data.TL[1, 1]
data.ΦG[1, 1] = data.TG[1, 1]

data_2B = deepcopy(data)

model = FermiHubbardModel(U = 2.)
tmax = 2.0;

# ==========================================================================================================

atol = 1e-10
rtol = 1e-8;

atol = 1e-10
rtol = 1e-6;

# atol = 1e-8
# rtol = 1e-6;

atol = 1e-6
rtol = 1e-4;

# ==========================================================================================================

@time sol = kbsolve!(
    (x...) -> fv!(model, data, x...),
    (x...) -> fd!(model, data, x...),
    [data.GL, data.GG, data.FL, data.FG],
    (0.0, tmax);
    callback = (x...) -> T_matrix!(model, data, x...),
    atol = atol,
    rtol = rtol,
    dtini=1e-10,
#     qmax=4,
#     γ=19/20,
    stop = x -> (println("t: $(x[end])"); flush(stdout); false)
);

save("FH_3D_T_matrix_sol_U_$(model.U)_tmax_$(tmax)_atol_$(atol)_rtol_$(rtol).jld", "solution", sol)

# ==========================================================================================================

@time sol_2B = kbsolve!(
    (x...) -> fv!(model, data_2B, x...),
    (x...) -> fd!(model, data_2B, x...),
    [data_2B.GL, data_2B.GG, data_2B.FL, data_2B.FG],
    (0.0, tmax);
    callback = (x...) -> second_Born!(model, data_2B, x...),
    atol = atol,
    rtol = rtol,
    dtini=1e-10,
#     γ=19/20,
    stop = x -> (println("t: $(x[end])"); flush(stdout); false)
);

save("FH_3D_second_Born_sol_U_$(model.U)_tmax_$(tmax)_atol_$(atol)_rtol_$(rtol).jld", "solution", sol)

# ==========================================================================================================