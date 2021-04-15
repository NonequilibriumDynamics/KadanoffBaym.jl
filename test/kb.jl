@testset "1-time benchmark" begin
    λ = 0.2

    function fv(u, ts, t, t′)
        return [1im * u[2][t,t′], 1im * u[1][t,t′] - λ * u[2][t,t′]]
    end

    function fd(u, ts, t)
        return [0im * u[2][t,t], 0im * u[1][t,t]]
    end

    G = GreenFunction(zeros(ComplexF64,1,1,1,1), Greater)
    L = GreenFunction(1im * ones(ComplexF64,1,1,1,1), Lesser)

    u0 = [G, L]

    sol = kbsolve(fv, fd, u0, (0.0, 1.0); atol=1e-6, rtol=1e-4);

    function sol1(t, g0, l0)
        s = sqrt(Complex(λ^2 - 4))
        return exp(-λ * t / 2) * (l0 * cosh(0.5 * t * s) + (2im * g0 + l0 * λ) * sinh(0.5 * t * s) / s)
    end

    function sol2(t, g0, l0)
        s = sqrt(Complex(λ^2 - 4))
        return exp(-λ * t / 2) * (g0 * cosh(0.5 * t * s) + (2im * l0 - g0 * λ) * sinh(0.5 * t * s) / s)
    end

    @testset begin
        for (i, t) in enumerate(sol.t)
            @test G[i,1][1,1] ≈ sol1(t, L[1,1,1,1], G[1,1,1,1]) atol=1e-6
            @test L[i,1][1,1] ≈ sol2(t, L[1,1,1,1], G[1,1,1,1]) atol=1e-6
        end
    end
end

@testset "2-time benchmark" begin
    λ = 0.2

    function fv(u, ts, t, t′)
        return [-1im * cos(λ * ts[t]) * u[1][t,t′]]
    end

    function fd(u, ts, t)
        return [0im * u[1][t,t]]
    end

    L = GreenFunction(-1im * ones(ComplexF64,1,1), Lesser)

    sol = kbsolve(fv, fd, [L, ], (0.0, 2.0); atol=1e-9, rtol=1e-6);

    function sol1(t, l0)
        return -1.0im * exp(-1.0im * 1/λ * sin(λ * t))
    end

    @testset begin
        for (i, tᵢ) in enumerate(sol.t)
            for (j, tⱼ) in enumerate(sol.t)
                @test L[i,j] ≈ sol1(tᵢ-tⱼ, L[1,1]) atol=1e-6 rtol=1e-3
            end
        end
    end
end
