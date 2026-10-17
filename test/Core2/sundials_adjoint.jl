using SciMLSensitivity, Sundials, OrdinaryDiffEq, ForwardDiff, Zygote, Test
using SciMLSensitivity: alg_autodiff

function lotka!(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
    return nothing
end

u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 10.0)
prob = ODEProblem(lotka!, u0, tspan, p)
ts = collect(0.0:0.5:10.0)

# G(u0, p) = sum over save times of sum(u(t))
function G(θ)
    _prob = remake(prob, u0 = θ[1:2], p = θ[3:end])
    _sol = solve(_prob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = ts)
    return sum(sum(u) for u in _sol.u)
end
refgrad = ForwardDiff.gradient(G, [u0; p])
dgdu(out, u, p, t, i) = (out .= 1.0)

@testset "adjoint_sensitivities interface" begin
    sol = solve(
        prob, CVODE_BDF(), abstol = 1.0e-10, reltol = 1.0e-10, saveat = ts
    )
    @testset "vjp = $(nameof(typeof(vjp)))" for vjp in (
            ReverseDiffVJP(), EnzymeVJP(), false,
        )
        du0, dp = adjoint_sensitivities(
            sol, CVODE_BDF(); t = ts, dgdu_discrete = dgdu,
            sensealg = SundialsAdjoint(autojacvec = vjp),
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test du0 ≈ refgrad[1:2] rtol = 1.0e-5
        @test vec(dp) ≈ refgrad[3:end] rtol = 1.0e-5
    end

    @testset "default autojacvec" begin
        du0, dp = adjoint_sensitivities(
            sol, CVODE_BDF(); t = ts, dgdu_discrete = dgdu,
            sensealg = SundialsAdjoint(),
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test du0 ≈ refgrad[1:2] rtol = 1.0e-5
        @test vec(dp) ≈ refgrad[3:end] rtol = 1.0e-5
    end

    @testset "GMRES linear solver" begin
        du0, dp = adjoint_sensitivities(
            sol, CVODE_BDF(linear_solver = :GMRES); t = ts, dgdu_discrete = dgdu,
            sensealg = SundialsAdjoint(),
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test du0 ≈ refgrad[1:2] rtol = 1.0e-5
        @test vec(dp) ≈ refgrad[3:end] rtol = 1.0e-5
    end

    @testset "CVODE_Adams and :polynomial interpolation" begin
        sol_adams = solve(
            prob, CVODE_Adams(), abstol = 1.0e-10, reltol = 1.0e-10, saveat = ts
        )
        du0, dp = adjoint_sensitivities(
            sol_adams, CVODE_Adams(); t = ts, dgdu_discrete = dgdu,
            sensealg = SundialsAdjoint(interp = :polynomial),
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test du0 ≈ refgrad[1:2] rtol = 1.0e-5
        @test vec(dp) ≈ refgrad[3:end] rtol = 1.0e-5
    end

    @testset "subset of times, endpoint only" begin
        du0, dp = adjoint_sensitivities(
            sol, CVODE_BDF(); t = [tspan[2]], dgdu_discrete = dgdu,
            sensealg = SundialsAdjoint(),
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        function Gend(θ)
            _prob = remake(prob, u0 = θ[1:2], p = θ[3:end])
            _sol = solve(
                _prob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12,
                save_everystep = false
            )
            return sum(_sol.u[end])
        end
        refend = ForwardDiff.gradient(Gend, [u0; p])
        @test du0 ≈ refend[1:2] rtol = 1.0e-5
        @test vec(dp) ≈ refend[3:end] rtol = 1.0e-5
    end
end

@testset "solver objects survive GC during integration" begin
    # The FixedPoint nonlinear solver handle used for `method = :Functional`
    # must stay rooted for the whole CVodeF/CVodeB integration; GC pressure in
    # the RHS segfaulted when it was collectable (its finalizer frees the
    # SUNDIALS object). GC in every RHS call makes that deterministic.
    function lotka_gc!(du, u, p, t)
        GC.gc(false)
        lotka!(du, u, p, t)
        return nothing
    end
    prob_gc = ODEProblem(lotka_gc!, u0, tspan, p)
    sol_gc = solve(prob_gc, CVODE_Adams(), abstol = 1.0e-8, reltol = 1.0e-8, saveat = ts)
    du0, dp = adjoint_sensitivities(
        sol_gc, CVODE_Adams(); t = ts, dgdu_discrete = dgdu,
        sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP()),
        abstol = 1.0e-8, reltol = 1.0e-8
    )
    @test du0 ≈ refgrad[1:2] rtol = 1.0e-4
    @test vec(dp) ≈ refgrad[3:end] rtol = 1.0e-4
end

@testset "continuous cost functionals" begin
    sol = solve(prob, CVODE_BDF(), abstol = 1.0e-10, reltol = 1.0e-10)

    # G = ∫ (p1 * (u1 + u2)) dt via an augmented quadrature state
    function lotka_quad!(du, u, p, t)
        lotka!(du, u, p, t)
        du[3] = p[1] * (u[1] + u[2])
        return nothing
    end
    function Gcont(θ)
        _prob = ODEProblem(lotka_quad!, [θ[1], θ[2], 0.0], tspan, θ[3:end])
        _sol = solve(
            _prob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12,
            save_everystep = false
        )
        return _sol.u[end][3]
    end
    refcont = ForwardDiff.gradient(Gcont, [u0; p])

    dgdu_cont(out, u, p, t) = (out .= p[1])
    dgdp_cont(out, u, p, t) = (out .= 0.0; out[1] = u[1] + u[2])
    g_cont(u, p, t) = p[1] * (u[1] + u[2])

    @testset "dgdu_continuous + dgdp_continuous" begin
        du0, dp = adjoint_sensitivities(
            sol, CVODE_BDF(); dgdu_continuous = dgdu_cont,
            dgdp_continuous = dgdp_cont,
            sensealg = SundialsAdjoint(),
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test du0 ≈ refcont[1:2] rtol = 1.0e-5
        @test vec(dp) ≈ refcont[3:end] rtol = 1.0e-5
    end

    @testset "scalar g" begin
        du0, dp = adjoint_sensitivities(
            sol, CVODE_BDF(); g = g_cont,
            sensealg = SundialsAdjoint(),
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test du0 ≈ refcont[1:2] rtol = 1.0e-5
        @test vec(dp) ≈ refcont[3:end] rtol = 1.0e-5
    end

    @testset "mixed discrete + continuous" begin
        sol_saved = solve(
            prob, CVODE_BDF(), abstol = 1.0e-10, reltol = 1.0e-10, saveat = ts
        )
        du0, dp = adjoint_sensitivities(
            sol_saved, CVODE_BDF(); t = ts, dgdu_discrete = dgdu,
            dgdu_continuous = dgdu_cont, dgdp_continuous = dgdp_cont,
            sensealg = SundialsAdjoint(),
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        refmix = refgrad .+ refcont
        @test du0 ≈ refmix[1:2] rtol = 1.0e-5
        @test vec(dp) ≈ refmix[3:end] rtol = 1.0e-5
    end
end

@testset "reverse-mode AD of solve" begin
    function loss(p)
        _sol = solve(
            prob, CVODE_BDF(); p = p, saveat = ts,
            abstol = 1.0e-10, reltol = 1.0e-10,
            sensealg = SundialsAdjoint()
        )
        return sum(Array(_sol))
    end
    dp_zygote = Zygote.gradient(loss, p)[1]
    @test dp_zygote ≈ refgrad[3:end] rtol = 1.0e-5

    function loss_u0(u0)
        _sol = solve(
            prob, CVODE_BDF(); u0 = u0, saveat = ts,
            abstol = 1.0e-10, reltol = 1.0e-10,
            sensealg = SundialsAdjoint()
        )
        return sum(Array(_sol))
    end
    du0_zygote = Zygote.gradient(loss_u0, u0)[1]
    @test du0_zygote ≈ refgrad[1:2] rtol = 1.0e-5
end

@testset "informative errors" begin
    sol_tsit = solve(prob, Tsit5(), abstol = 1.0e-10, reltol = 1.0e-10, saveat = ts)
    err = try
        adjoint_sensitivities(
            sol_tsit, Tsit5(); t = ts, dgdu_discrete = dgdu,
            sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP())
        )
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("CVODES/IDAS", err.msg)
    @test occursin("CVODE_BDF", err.msg)

    sol = solve(prob, CVODE_BDF(), abstol = 1.0e-10, reltol = 1.0e-10, saveat = ts)
    @test_throws ErrorException adjoint_sensitivities(
        sol, CVODE_BDF(); t = ts, dgdu_discrete = dgdu, g = (u, p, t) -> sum(u),
        sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP())
    )
    @test_throws ErrorException SundialsAdjoint(interp = :spline)
    @test_throws ErrorException adjoint_sensitivities(
        sol, CVODE_BDF(); t = ts, dgdu_discrete = dgdu,
        sensealg = SundialsAdjoint(autojacvec = true)
    )
end

@testset "GaussAdjoint with CVODE_BDF" begin
    # A `saveat` (non-dense) forward solution auto-enables Gauss checkpointing,
    # whose dense checkpoint re-solves carry a different interpolation type
    # (Hermite) than the original Sundials solution (linear).
    sol_saveat = solve(
        prob, CVODE_BDF(), abstol = 1.0e-10, reltol = 1.0e-10, saveat = ts
    )
    du0, dp = adjoint_sensitivities(
        sol_saveat, CVODE_BDF(); t = ts, dgdu_discrete = dgdu,
        sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP()),
        abstol = 1.0e-8, reltol = 1.0e-8
    )
    @test du0 ≈ refgrad[1:2] rtol = 1.0e-5
    @test vec(dp) ≈ refgrad[3:end] rtol = 1.0e-5

    sol_dense = solve(
        prob, CVODE_BDF(), abstol = 1.0e-10, reltol = 1.0e-10, dense = true
    )
    du0d, dpd = adjoint_sensitivities(
        sol_dense, CVODE_BDF(); t = ts, dgdu_discrete = dgdu,
        sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP()),
        abstol = 1.0e-8, reltol = 1.0e-8
    )
    @test du0d ≈ refgrad[1:2] rtol = 1.0e-5
    @test vec(dpd) ≈ refgrad[3:end] rtol = 1.0e-5
end
