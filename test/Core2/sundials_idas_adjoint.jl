using SciMLSensitivity, Sundials, OrdinaryDiffEq, ForwardDiff, Zygote, Test
using SciMLBase, LinearAlgebra

# Index-1 DAE with known analytic gradient (the Sundials.jl IDAS adjoint test
# problem): F1 = y1' + a*y1, F2 = y2 - y1, y1(0) = y2(0) = 1, so
# y1(t) = y2(t) = exp(-a*t). For G = y1(tf): dG/dy1(0) = exp(-a*tf) and
# dG/da = -tf*exp(-a*tf).
@testset "analytic index-1 DAE" begin
    function lin_dae!(res, du, u, p, t)
        res[1] = du[1] + p[1] * u[1]
        res[2] = u[2] - u[1]
        return nothing
    end
    a = 0.3
    tf = 2.0
    prob = DAEProblem(
        lin_dae!, [-a, -a], [1.0, 1.0], (0.0, tf), [a];
        differential_vars = [true, false]
    )
    sol = solve(prob, IDA(); abstol = 1.0e-10, reltol = 1.0e-10, saveat = [tf])
    @test sol.u[end][1] ≈ exp(-a * tf) rtol = 1.0e-7

    dgdu(out, u, p, t, i) = (out .= 0.0; out[1] = 1.0)
    du0, dp = adjoint_sensitivities(
        sol, IDA(); t = [tf], dgdu_discrete = dgdu,
        sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP()),
        abstol = 1.0e-10, reltol = 1.0e-10
    )
    @test du0[1] ≈ exp(-a * tf) rtol = 1.0e-6
    @test du0[2] == 0.0
    @test dp[1] ≈ -tf * exp(-a * tf) rtol = 1.0e-6
end

function rober_dae!(res, du, u, p, t)
    y1, y2, y3 = u
    k1, k2, k3 = p
    res[1] = du[1] + k1 * y1 - k3 * y2 * y3
    res[2] = du[2] - k1 * y1 + k2 * y2^2 + k3 * y2 * y3
    res[3] = y1 + y2 + y3 - 1
    return nothing
end

p = [0.04, 3.0e7, 1.0e4]
u0 = [1.0, 0.0, 0.0]
du0 = [-0.04, 0.04, 0.0]
tspan = (0.0, 100.0)
ts = [10.0, 50.0, 100.0]
daeprob = DAEProblem(
    rober_dae!, du0, u0, tspan, p; differential_vars = [true, true, false]
)
sol = solve(daeprob, IDA(); abstol = 1.0e-10, reltol = 1.0e-10, saveat = ts)

# G = sum over ts of (u1 + u2). Cost only on the differential variables; the
# gradient w.r.t. (u1(0), u2(0)) is taken along the constraint manifold
# u3 = 1 - u1 - u2, matching what the DAE solution operator differentiates.
dgdu(out, u, p, t, i) = (out .= 0.0; out[1] = 1.0; out[2] = 1.0)

# ForwardDiff reference through the equivalent mass-matrix ODE formulation.
function rober_mm!(du, u, p, t)
    y1, y2, y3 = u
    k1, k2, k3 = p
    du[1] = -k1 * y1 + k3 * y2 * y3
    du[2] = k1 * y1 - k2 * y2^2 - k3 * y2 * y3
    du[3] = y1 + y2 + y3 - 1
    return nothing
end
fmm = ODEFunction(rober_mm!, mass_matrix = Diagonal([1.0, 1.0, 0.0]))
function G(θ)
    _u0 = [θ[1], θ[2], 1 - θ[1] - θ[2]]
    _prob = ODEProblem(fmm, _u0, tspan, θ[3:5])
    _sol = solve(_prob, Rodas5P(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = ts)
    return sum(u[1] + u[2] for u in _sol.u)
end
refgrad = ForwardDiff.gradient(G, [u0[1]; u0[2]; p])

@testset "cost on algebraic variable (index-1 jump correction)" begin
    # G = sum over ts of sum(u.^2)/2, touching the algebraic u3: the jumps take
    # the index-1 constraint-transfer correction.
    dgdu_all(out, u, p, t, i) = (out .= u)
    function Gall(θ)
        _u0 = [θ[1], θ[2], 1 - θ[1] - θ[2]]
        _prob = ODEProblem(fmm, _u0, tspan, θ[3:5])
        _sol = solve(_prob, Rodas5P(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = ts)
        return sum(sum(abs2, u) / 2 for u in _sol.u)
    end
    refall = ForwardDiff.gradient(Gall, [u0[1]; u0[2]; p])
    du0g, dp = adjoint_sensitivities(
        sol, IDA(); t = ts, dgdu_discrete = dgdu_all,
        sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP()),
        abstol = 1.0e-10, reltol = 1.0e-10
    )
    @test du0g[1:2] ≈ refall[1:2] rtol = 1.0e-4
    @test du0g[3] == 0.0
    @test vec(dp) ≈ refall[3:5] rtol = 1.0e-4
end

@testset "parameter-dependent algebraic constraint" begin
    # Constraint u1 + u2 + u3 = s with s = p[4]: exercises the Δλa' ∂F_alg/∂p
    # correction to the parameter gradient at the cost jumps.
    function rober_s!(res, du, u, p, t)
        y1, y2, y3 = u
        k1, k2, k3, s = p
        res[1] = du[1] + k1 * y1 - k3 * y2 * y3
        res[2] = du[2] - k1 * y1 + k2 * y2^2 + k3 * y2 * y3
        res[3] = y1 + y2 + y3 - s
        return nothing
    end
    ps = [0.04, 3.0e7, 1.0e4, 1.0]
    sprob = DAEProblem(
        rober_s!, du0, u0, tspan, ps; differential_vars = [true, true, false]
    )
    ssol = solve(sprob, IDA(); abstol = 1.0e-10, reltol = 1.0e-10, saveat = ts)
    dgdu_all(out, u, p, t, i) = (out .= u)
    function rober_mms!(du, u, p, t)
        y1, y2, y3 = u
        k1, k2, k3, s = p
        du[1] = -k1 * y1 + k3 * y2 * y3
        du[2] = k1 * y1 - k2 * y2^2 - k3 * y2 * y3
        du[3] = y1 + y2 + y3 - s
        return nothing
    end
    fmms = ODEFunction(rober_mms!, mass_matrix = Diagonal([1.0, 1.0, 0.0]))
    function Gs(θ)
        _u0 = [one(eltype(θ)), zero(eltype(θ)), θ[4] - 1]
        _prob = ODEProblem(fmms, _u0, tspan, θ)
        _sol = solve(_prob, Rodas5P(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = ts)
        return sum(sum(abs2, u) / 2 for u in _sol.u)
    end
    refs = ForwardDiff.gradient(Gs, ps)
    du0g, dp = adjoint_sensitivities(
        ssol, IDA(); t = ts, dgdu_discrete = dgdu_all,
        sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP()),
        abstol = 1.0e-10, reltol = 1.0e-10
    )
    @test vec(dp) ≈ refs rtol = 1.0e-4
end

@testset "Robertson DAE vs ForwardDiff" begin
    @testset "vjp = $(vjp)" for vjp in (ReverseDiffVJP(), ReverseDiffVJP(true))
        du0g, dp = adjoint_sensitivities(
            sol, IDA(); t = ts, dgdu_discrete = dgdu,
            sensealg = SundialsAdjoint(autojacvec = vjp),
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test du0g[1:2] ≈ refgrad[1:2] rtol = 1.0e-4
        @test du0g[3] == 0.0
        @test vec(dp) ≈ refgrad[3:5] rtol = 1.0e-5
    end

    @testset "default autojacvec" begin
        du0g, dp = adjoint_sensitivities(
            sol, IDA(); t = ts, dgdu_discrete = dgdu,
            sensealg = SundialsAdjoint(),
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test du0g[1:2] ≈ refgrad[1:2] rtol = 1.0e-4
        @test vec(dp) ≈ refgrad[3:5] rtol = 1.0e-5
    end

    @testset "dgdp_discrete" begin
        # G2 = G + sum over ts of k1, so dG2/dk1 = dG/dk1 + length(ts).
        dgdp(out, u, p, t, i) = (out .= 0.0; out[1] = 1.0)
        du0g, dp = adjoint_sensitivities(
            sol, IDA(); t = ts, dgdu_discrete = dgdu, dgdp_discrete = dgdp,
            sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP()),
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test dp[1] ≈ refgrad[3] + length(ts) rtol = 1.0e-5
        @test vec(dp)[2:3] ≈ refgrad[4:5] rtol = 1.0e-5
    end

    @testset "endpoint only" begin
        du0g, dp = adjoint_sensitivities(
            sol, IDA(); t = [tspan[2]], dgdu_discrete = dgdu,
            sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP()),
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        function Gend(θ)
            _u0 = [θ[1], θ[2], 1 - θ[1] - θ[2]]
            _prob = ODEProblem(fmm, _u0, tspan, θ[3:5])
            _sol = solve(
                _prob, Rodas5P(), abstol = 1.0e-12, reltol = 1.0e-12,
                save_everystep = false
            )
            return _sol.u[end][1] + _sol.u[end][2]
        end
        refend = ForwardDiff.gradient(Gend, [u0[1]; u0[2]; p])
        @test du0g[1:2] ≈ refend[1:2] rtol = 1.0e-4
        @test vec(dp) ≈ refend[3:5] rtol = 1.0e-5
    end
end

@testset "reverse-mode AD of solve" begin
    function loss(p)
        _sol = solve(
            daeprob, IDA(); p = p, saveat = ts,
            abstol = 1.0e-10, reltol = 1.0e-10,
            sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP())
        )
        A = Array(_sol)
        return sum(A[1, :]) + sum(A[2, :])
    end
    dp_zygote = Zygote.gradient(loss, p)[1]
    @test dp_zygote ≈ refgrad[3:5] rtol = 1.0e-5

    function loss_u0(u0)
        _sol = solve(
            daeprob, IDA(); u0 = u0, saveat = ts,
            abstol = 1.0e-10, reltol = 1.0e-10,
            sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP())
        )
        A = Array(_sol)
        return sum(A[1, :]) + sum(A[2, :])
    end
    du0_zygote = Zygote.gradient(loss_u0, u0)[1]
    @test du0_zygote[1:2] ≈ refgrad[1:2] rtol = 1.0e-4
    @test du0_zygote[3] == 0.0
end

@testset "informative errors" begin
    # DAEProblem with a CVODES solver
    err = try
        adjoint_sensitivities(
            sol, CVODE_BDF(); t = ts, dgdu_discrete = dgdu,
            sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP())
        )
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("IDA", err.msg)

    # ODEProblem with IDA
    odeprob = ODEProblem((du, u, p, t) -> (du .= -u), [1.0], (0.0, 1.0))
    odesol = solve(odeprob, CVODE_BDF(); abstol = 1.0e-8, reltol = 1.0e-8, saveat = [1.0])
    err = try
        adjoint_sensitivities(
            odesol, IDA(); t = [1.0],
            dgdu_discrete = (out, u, p, t, i) -> (out .= 1.0),
            sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP())
        )
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("CVODE_BDF", err.msg)

    # unsupported autojacvec
    @test_throws ErrorException adjoint_sensitivities(
        sol, IDA(); t = ts, dgdu_discrete = dgdu,
        sensealg = SundialsAdjoint(autojacvec = EnzymeVJP())
    )

    # unsorted cost times
    @test_throws ErrorException adjoint_sensitivities(
        sol, IDA(); t = reverse(ts), dgdu_discrete = dgdu,
        sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP())
    )

    # continuous cost functionals are not implemented for DAEs
    @test_throws ErrorException adjoint_sensitivities(
        sol, IDA(); g = (u, p, t) -> sum(u),
        sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP())
    )

    # Hessenberg index-2 DAEs are rejected on the IDAS route
    function hess2!(res, du, u, p, t)
        res[1] = p[1] * u[2] + u[3] - du[1]
        res[2] = -p[2] * u[1] - du[2]
        res[3] = u[1] - sin(p[3] * t)
        return nothing
    end
    i2prob = DAEProblem(
        hess2!, [3.0, 0.0, 0.0], [0.0, 1.0, 2.0], (0.0, 1.0), [1.0, 2.0, 3.0];
        differential_vars = [true, true, false]
    )
    # IDA itself may fail on the index-2 forward solve; the structural rejection
    # only needs a solution object, however inaccurate.
    i2sol = try
        solve(i2prob, IDA(); abstol = 1.0e-8, reltol = 1.0e-8, saveat = [1.0], verbose = false)
    catch
        nothing
    end
    if i2sol !== nothing
        err = try
            adjoint_sensitivities(
                i2sol, IDA(); t = [1.0],
                dgdu_discrete = (out, u, p, t, i) -> (out .= [u[1], u[2], 0.0]),
                sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP())
            )
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("index-2", err.msg)
    end

    # state-dependent ∂F/∂du is caught by the runtime screen
    function nonlin_du!(res, du, u, p, t)
        res[1] = u[1] * du[1] + p[1] * u[1]
        res[2] = u[2] - u[1]
        return nothing
    end
    nlprob = DAEProblem(
        nonlin_du!, [-0.3, -0.3], [1.0, 1.0], (0.0, 2.0), [0.3];
        differential_vars = [true, false]
    )
    nlsol = solve(nlprob, IDA(); abstol = 1.0e-8, reltol = 1.0e-8, saveat = [2.0])
    err = try
        adjoint_sensitivities(
            nlsol, IDA(); t = [2.0],
            dgdu_discrete = (out, u, p, t, i) -> (out .= 0.0; out[1] = 1.0),
            sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP())
        )
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("constant Jacobian", err.msg)

    # missing differential_vars
    nodiffvars = DAEProblem(rober_dae!, du0, u0, tspan, p)
    nodiffsol = SciMLBase.build_solution(
        nodiffvars, IDA(), sol.t, sol.u; retcode = sol.retcode
    )
    err = try
        adjoint_sensitivities(
            nodiffsol, IDA(); t = ts, dgdu_discrete = dgdu,
            sensealg = SundialsAdjoint(autojacvec = ReverseDiffVJP())
        )
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("differential_vars", err.msg)
end
