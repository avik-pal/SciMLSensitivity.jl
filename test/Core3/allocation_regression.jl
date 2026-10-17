# Allocation regression tests for the continuous adjoints.
#
# Guards three properties measured on Julia 1.10 / Enzyme 0.13:
#   1. The per-call allocation ceiling of the GaussAdjoint inner loop
#      (`vec_pjac!`) with EnzymeVJP.
#   2. The per-step allocation slope of full adjoint solves: allocations must
#      not grow faster than the recorded slope with integration length —
#      catches any accidentally introduced per-step/per-stage allocation.
#   3. With `dgrad === nothing` (Gauss/Quadrature adjoint rhs) and a
#      runtime-activity mode, the parameter shadow buffer must remain
#      untouched — proves Enzyme receives `Const(p)` and is not computing a
#      discarded parameter gradient on every stage. With the default static
#      mode the shadow must be written (`Duplicated` kept; the demotion is
#      unsound there).
#
# 1 and 2 count allocations, which is meaningless under coverage
# instrumentation, so they are skipped when coverage is active. 3 is a
# behavioral check and always runs.
using SciMLSensitivity, OrdinaryDiffEq, LinearAlgebra, Test
using Enzyme

coverage_on = Base.JLOptions().code_coverage != 0

function lv!(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
    return nothing
end
dgdu(out, u, p, t, i) = (out .= 1.0)

fwdsol(T) = solve(
    ODEProblem(lv!, [1.0, 1.0], (0.0, T), [1.5, 1.0, 3.0, 1.0]),
    Tsit5(); abstol = 1.0e-8, reltol = 1.0e-8, dense = true
)

function adjoint_allocs(T, sa)
    sol = fwdsol(T)
    ts = collect(range(0.0, T, length = 21))
    f() = adjoint_sensitivities(
        sol, Tsit5(); t = ts, dgdu_discrete = dgdu,
        sensealg = sa, abstol = 1.0e-6, reltol = 1.0e-3
    )
    f(); f()  # compile + warm caches
    return minimum(@allocated(f()) for _ in 1:5), length(sol.t)
end

@testset "adjoint allocation regressions" begin
    @testset "per-call ceiling: GaussIntegrand vec_pjac! (EnzymeVJP)" begin
        if coverage_on
            @info "coverage active — skipping vec_pjac! allocation ceiling"
        else
            sol = fwdsol(10.0)
            sa = GaussAdjoint(autojacvec = EnzymeVJP())
            S = SciMLSensitivity.GaussIntegrand(sol, sa, collect(sol.t), nothing)
            out = zeros(4); λ = [1.0, 2.0]; y = copy(sol.u[10]); t = sol.t[10]
            SciMLSensitivity.vec_pjac!(out, λ, y, t, S)
            SciMLSensitivity.vec_pjac!(out, λ, y, t, S)
            a = minimum(@allocated(SciMLSensitivity.vec_pjac!(out, λ, y, t, S)) for _ in 1:5)
            # the warmed vec_pjac! inner loop is allocation-free: the Enzyme
            # call itself is 0 B and `unwrapped_f` is hoisted to construction.
            @test a == 0
        end
    end

    @testset "per-step slope of full adjoint solves" begin
        if coverage_on
            @info "coverage active — skipping adjoint allocation slope"
        else
            # slope = (allocs(T=100) - allocs(T=10)) / (fwd-steps difference).
            # Both inner loops are allocation-free; the ceiling only leaves
            # room for measurement noise.
            for (name, sa, ceiling) in [
                    ("Gauss+EnzymeVJP", GaussAdjoint(autojacvec = EnzymeVJP()), 8.0),
                    ("Interp+EnzymeVJP", InterpolatingAdjoint(autojacvec = EnzymeVJP()), 8.0),
                ]
                a10, n10 = adjoint_allocs(10.0, sa)
                a100, n100 = adjoint_allocs(100.0, sa)
                slope = (a100 - a10) / (n100 - n10)
                @test slope <= ceiling
            end
        end
    end

    @testset "dgrad === nothing skips the parameter gradient (EnzymeVJP)" begin
        # The Gauss/Quadrature adjoint rhs calls vecjacobian! without dgrad on
        # every solver stage. With a runtime-activity mode Enzyme must receive
        # `Const(p)` there: if the parameter shadow were passed as
        # `Duplicated`, it would be zeroed and written each call — the NaN
        # sentinel would not survive. With the default static mode the
        # demotion is unsound (p-derived array references stored into active
        # computation), so `Duplicated` must be kept and the shadow written.
        # Behavioral check, runs under coverage too.
        function sentinel_survives(vjp)
            sol = fwdsol(10.0)
            sa = GaussAdjoint(autojacvec = vjp)
            adj_prob, _, _ = SciMLSensitivity.ODEAdjointProblem(
                sol, sa, Tsit5(),
                SciMLSensitivity.GaussIntegrand(sol, sa, collect(sol.t), nothing),
                nothing, collect(range(0.0, 10.0, length = 21)), dgdu
            )
            S = adj_prob.f.f  # ODEGaussAdjointSensitivityFunction
            tmp2 = S.diffcache.paramjac_config[2]
            dλ = zeros(2); λ = [1.0, 2.0]; y = copy(sol.u[10])
            SciMLSensitivity.vecjacobian!(dλ, y, λ, sol.prob.p, sol.t[10], S)
            fill!(tmp2, NaN)
            SciMLSensitivity.vecjacobian!(dλ, y, λ, sol.prob.p, sol.t[10], S)
            return all(isnan, tmp2), all(isfinite, dλ)
        end
        surv_rta, finite_rta = sentinel_survives(
            EnzymeVJP(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
        )
        @test surv_rta
        @test finite_rta
        surv_static, finite_static = sentinel_survives(EnzymeVJP())
        @test !surv_static  # static mode keeps Duplicated(p, shadow)
        @test finite_static
    end
end
