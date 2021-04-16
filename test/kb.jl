@testset "1-time benchmark" begin
    λ = 0.2
    atol=1e-8
    rtol=1e-5

    function fv(u, ts, t, t′)
        return [1im * u[2][t,t′], 1im * u[1][t,t′] - λ * u[2][t,t′]]
    end

    function fd(u, ts, t)
        return [0im * u[2][t,t], 0im * u[1][t,t]]
    end

    G = GreenFunction(zeros(ComplexF64,1,1), Greater)
    L = GreenFunction(1im * ones(ComplexF64,1,1), Lesser)

    kb = kbsolve(fv, fd, [G, L], (0.0, 100.0); atol=atol, rtol=rtol);

    function sol1(t, g0, l0)
        s = sqrt(Complex(λ^2 - 4))
        return exp(-λ * t / 2) * (l0 * cosh(0.5 * t * s) + (2im * g0 + l0 * λ) * sinh(0.5 * t * s) / s)
    end

    function sol2(t, g0, l0)
        s = sqrt(Complex(λ^2 - 4))
        return exp(-λ * t / 2) * (g0 * cosh(0.5 * t * s) + (2im * l0 - g0 * λ) * sinh(0.5 * t * s) / s)
    end

    @testset begin
        @test G[:,1] ≈ [sol1(t1, L[1,1], G[1,1]) for t1 in kb.t] atol=1e1atol rtol=1e1rtol
        @test L[:,1] ≈ [sol2(t1, L[1,1], G[1,1]) for t1 in kb.t] atol=1e1atol rtol=1e1rtol
    end
end

@testset "2-time benchmark" begin
    λ = 0.2
    atol=1e-8
    rtol=1e-5

    function fv(u, ts, t, t′)
        return [-1im * λ * cos(λ * (ts[t] - ts[t′])) * u[1][t,t′]]
    end

    function fd(u, ts, t)
        return [0im * u[1][t,t]]
    end

    L = GreenFunction(-1im * ones(ComplexF64,1,1), Lesser)

    kb = kbsolve(fv, fd, [L, ], (0.0, 200.0); atol=atol, rtol=rtol);

    function sol(t, t′)
        return -1.0im * exp(-1.0im * sin(λ * (t - t′)))
    end

    @test L.data ≈ [sol(t1, t2) for t1 in kb.t, t2 in kb.t] atol=1e1atol rtol=1e1rtol
end
