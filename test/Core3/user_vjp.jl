using SciMLSensitivity, OrdinaryDiffEq, ForwardDiff, LinearAlgebra, Zygote, SciMLBase
using SparseArrays
using Test

# Lotka-Volterra system with user-provided VJP, paramjac, and vjp_p
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    return du[2] = -δ * y + γ * x * y
end

# State VJP: (df/du)^T * v
function lv_vjp!(dλ, λ, u, p, t)
    x, y = u
    α, β, δ, γ = p
    dλ[1] = λ[1] * (α - β * y) + λ[2] * (γ * y)
    return dλ[2] = λ[1] * (-β * x) + λ[2] * (-δ + γ * x)
end

# Full parameter Jacobian: df/dp (n_states × n_params matrix)
function lv_paramjac!(pJ, u, p, t)
    x, y = u
    pJ .= 0
    pJ[1, 1] = x
    pJ[1, 2] = -x * y
    pJ[2, 3] = -y
    return pJ[2, 4] = x * y
end

# Parameter VJP: (df/dp)^T * v (vector of length n_params)
function lv_vjp_p!(Jpv, v, u, p, t)
    x, y = u
    Jpv[1] = x * v[1]
    Jpv[2] = -x * y * v[1]
    Jpv[3] = -y * v[2]
    return Jpv[4] = x * y * v[2]
end

p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
solver_kwargs = (abstol = 1.0e-12, reltol = 1.0e-12)

# ForwardDiff baseline
function loss_fwd(p)
    prob = ODEProblem(lotka_volterra!, u0, tspan, p)
    sol = solve(prob, Tsit5(); saveat = 0.1, solver_kwargs...)
    return sum(sol)
end
grad_fwd = ForwardDiff.gradient(loss_fwd, p)

# Helper to compute gradient for a given ODEFunction and sensealg
function compute_grad(f, sensealg)
    prob = ODEProblem(f, u0, tspan, p)
    loss = p -> sum(
        solve(
            prob, Tsit5(); p = p, saveat = 0.1, solver_kwargs...,
            sensealg = sensealg
        )
    )
    return Zygote.gradient(loss, p)[1]
end

@testset "User-provided VJP dispatch" begin
    @testset "VJP + vjp_p: $name" for (name, sensealg) in [
            ("GaussAdjoint", GaussAdjoint(autojacvec = EnzymeVJP())),
            (
                "QuadratureAdjoint",
                QuadratureAdjoint(
                    autojacvec = EnzymeVJP(), abstol = 1.0e-12, reltol = 1.0e-12
                ),
            ),
            ("InterpolatingAdjoint", InterpolatingAdjoint(autojacvec = EnzymeVJP())),
            ("BacksolveAdjoint", BacksolveAdjoint(autojacvec = EnzymeVJP())),
        ]
        f = ODEFunction(lotka_volterra!; vjp = lv_vjp!, vjp_p = lv_vjp_p!)
        grad = compute_grad(f, sensealg)
        @test isapprox(grad, grad_fwd, rtol = 1.0e-5)
    end

    @testset "VJP + paramjac: $name" for (name, sensealg) in [
            ("GaussAdjoint", GaussAdjoint(autojacvec = EnzymeVJP())),
            (
                "QuadratureAdjoint",
                QuadratureAdjoint(
                    autojacvec = EnzymeVJP(), abstol = 1.0e-12, reltol = 1.0e-12
                ),
            ),
            ("InterpolatingAdjoint", InterpolatingAdjoint(autojacvec = EnzymeVJP())),
            ("BacksolveAdjoint", BacksolveAdjoint(autojacvec = EnzymeVJP())),
        ]
        f = ODEFunction(lotka_volterra!; vjp = lv_vjp!, paramjac = lv_paramjac!)
        grad = compute_grad(f, sensealg)
        @test isapprox(grad, grad_fwd, rtol = 1.0e-5)
    end

    @testset "vjp_p matches paramjac exactly: $name" for (name, sensealg) in [
            ("GaussAdjoint", GaussAdjoint(autojacvec = EnzymeVJP())),
            (
                "QuadratureAdjoint",
                QuadratureAdjoint(
                    autojacvec = EnzymeVJP(), abstol = 1.0e-12, reltol = 1.0e-12
                ),
            ),
            ("InterpolatingAdjoint", InterpolatingAdjoint(autojacvec = EnzymeVJP())),
            ("BacksolveAdjoint", BacksolveAdjoint(autojacvec = EnzymeVJP())),
        ]
        f_vjpp = ODEFunction(lotka_volterra!; vjp = lv_vjp!, vjp_p = lv_vjp_p!)
        f_pjac = ODEFunction(lotka_volterra!; vjp = lv_vjp!, paramjac = lv_paramjac!)
        grad_vjpp = compute_grad(f_vjpp, sensealg)
        grad_pjac = compute_grad(f_pjac, sensealg)
        @test isapprox(grad_vjpp, grad_pjac, rtol = 1.0e-10)
    end

    @testset "vjp_p takes priority over paramjac" begin
        calls_vjp_p = Ref(0)
        calls_paramjac = Ref(0)

        function counting_vjp_p!(Jpv, v, u, p, t)
            calls_vjp_p[] += 1
            lv_vjp_p!(Jpv, v, u, p, t)
        end
        function counting_paramjac!(pJ, u, p, t)
            calls_paramjac[] += 1
            lv_paramjac!(pJ, u, p, t)
        end

        f = ODEFunction(
            lotka_volterra!;
            vjp = lv_vjp!,
            paramjac = counting_paramjac!,
            vjp_p = counting_vjp_p!
        )

        grad = compute_grad(f, GaussAdjoint(autojacvec = EnzymeVJP()))
        @test isapprox(grad, grad_fwd, rtol = 1.0e-5)
        @test calls_vjp_p[] > 0
        @test calls_paramjac[] == 0
    end

    @testset "VJP-only (no paramjac or vjp_p): $name" for (name, sensealg) in [
            ("GaussAdjoint", GaussAdjoint(autojacvec = ReverseDiffVJP())),
            ("InterpolatingAdjoint", InterpolatingAdjoint(autojacvec = ReverseDiffVJP())),
        ]
        f = ODEFunction(lotka_volterra!; vjp = lv_vjp!)
        grad = compute_grad(f, sensealg)
        @test isapprox(grad, grad_fwd, rtol = 1.0e-3)
    end
end

# Analytical Jacobian for the Lotka-Volterra system: df/du
function lv_jac!(J, u, p, t)
    x, y = u
    α, β, δ, γ = p
    J[1, 1] = α - β * y
    J[1, 2] = -β * x
    J[2, 1] = γ * y
    return J[2, 2] = -δ + γ * x
end

@testset "Adjoint Jacobian passthrough (user jac → adjoint solver)" begin
    # Dense jac_prototype
    @testset "Dense jac_prototype: $name" for (name, sensealg) in [
            ("GaussAdjoint", GaussAdjoint(autojacvec = EnzymeVJP())),
            (
                "QuadratureAdjoint",
                QuadratureAdjoint(
                    autojacvec = EnzymeVJP(), abstol = 1.0e-12, reltol = 1.0e-12
                ),
            ),
        ]
        jp = zeros(2, 2)
        lv_jac!(jp, u0, p, 0.0)
        f = ODEFunction(
            lotka_volterra!;
            vjp = lv_vjp!, vjp_p = lv_vjp_p!,
            jac = lv_jac!, jac_prototype = zeros(2, 2)
        )
        grad = compute_grad(f, sensealg)
        @test isapprox(grad, grad_fwd, rtol = 1.0e-5)
    end

    # Sparse jac_prototype
    @testset "Sparse jac_prototype: $name" for (name, sensealg) in [
            ("GaussAdjoint", GaussAdjoint(autojacvec = EnzymeVJP())),
            (
                "QuadratureAdjoint",
                QuadratureAdjoint(
                    autojacvec = EnzymeVJP(), abstol = 1.0e-12, reltol = 1.0e-12
                ),
            ),
        ]
        # Build sparse prototype from actual Jacobian structure
        jp_dense = zeros(2, 2)
        lv_jac!(jp_dense, u0, p, 0.0)
        jp_sparse = sparse(jp_dense)
        f = ODEFunction(
            lotka_volterra!;
            vjp = lv_vjp!, vjp_p = lv_vjp_p!,
            jac = lv_jac!, jac_prototype = jp_sparse
        )
        grad = compute_grad(f, sensealg)
        @test isapprox(grad, grad_fwd, rtol = 1.0e-5)
    end

    # Verify adjoint jac is actually called when provided.
    # Must use an implicit solver so the adjoint solver needs a Jacobian.
    @testset "Adjoint jac is called (implicit solver)" begin
        jac_calls = Ref(0)
        function counting_jac!(J, u, p, t)
            jac_calls[] += 1
            lv_jac!(J, u, p, t)
        end
        f = ODEFunction(
            lotka_volterra!;
            vjp = lv_vjp!, vjp_p = lv_vjp_p!,
            jac = counting_jac!, jac_prototype = zeros(2, 2)
        )
        prob = ODEProblem(f, u0, tspan, p)
        loss = p -> sum(
            solve(
                prob, Rodas5P(); p = p, saveat = 0.1, solver_kwargs...,
                sensealg = GaussAdjoint(autojacvec = EnzymeVJP())
            )
        )
        grad = Zygote.gradient(loss, p)[1]
        @test isapprox(grad, grad_fwd, rtol = 1.0e-3)
        @test jac_calls[] > 0
    end
end

@testset "AD fallback adjoint Jacobian (no user jac)" begin
    f_plain = ODEFunction(lotka_volterra!)
    prob = ODEProblem(f_plain, u0, tspan, p)
    sol = solve(prob, Rodas5P(); solver_kwargs...)

    # Unit test: the fallback builds -(df/du)^T at the interpolated forward state
    for sensealg in (GaussAdjoint(), GaussAdjoint(autodiff = false))
        jacfn = SciMLSensitivity.build_adjoint_jac(
            sol, sensealg, Rodas5P(), sol.prob.f, reverse(tspan)
        )
        @test jacfn !== nothing
        J_adj = zeros(2, 2)
        jacfn(J_adj, nothing, p, 4.2)
        J_ref = zeros(2, 2)
        lv_jac!(J_ref, sol(4.2), p, 4.2)
        @test isapprox(J_adj, -J_ref', rtol = 1.0e-6)
    end

    # No fallback for explicit adjoint solvers (Jacobian never used) or
    # finite-difference implicit solvers (rhs may not be Dual-safe there).
    @test SciMLSensitivity.build_adjoint_jac(
        sol, GaussAdjoint(), Tsit5(), sol.prob.f, reverse(tspan)
    ) === nothing
    @test SciMLSensitivity.build_adjoint_jac(
        sol, GaussAdjoint(), Rodas5P(autodiff = AutoFiniteDiff()), sol.prob.f,
        reverse(tspan)
    ) === nothing

    # End-to-end: gradients with an implicit adjoint solver match ForwardDiff
    ts_disc = collect(0.5:0.5:10.0)
    dgdu_disc = (out, u, p, t, i) -> (out .= 1)
    function loss_fd(p)
        _sol = solve(
            remake(prob, p = p), Rodas5P(); saveat = ts_disc, solver_kwargs...
        )
        return sum(sum(u) for u in _sol.u)
    end
    grad_ref = ForwardDiff.gradient(loss_fd, p)
    @testset "$name / Rodas5P" for (name, sensealg) in [
            ("GaussAdjoint(EnzymeVJP)", GaussAdjoint(autojacvec = EnzymeVJP())),
            ("GaussAdjoint(ReverseDiffVJP)", GaussAdjoint(autojacvec = ReverseDiffVJP())),
            (
                "QuadratureAdjoint(EnzymeVJP)",
                QuadratureAdjoint(
                    autojacvec = EnzymeVJP(), abstol = 1.0e-10, reltol = 1.0e-10
                ),
            ),
        ]
        du0, dp = adjoint_sensitivities(
            sol, Rodas5P(); t = ts_disc, dgdu_discrete = dgdu_disc,
            sensealg = sensealg, abstol = 1.0e-8, reltol = 1.0e-8
        )
        @test isapprox(vec(dp), grad_ref, rtol = 1.0e-4)
    end
end

@testset "Sparse user jac + continuous cost (regression: ftranspose! into dense)" begin
    # Before the adjoint_jac_prototype fix, the sparse-jac closure was passed to
    # the adjoint solver without its sparse prototype for continuous costs, so
    # `ftranspose!` hit a dense destination and threw a MethodError. The sparse
    # setup must run and agree with the dense-prototype setup.
    jp_dense = zeros(2, 2)
    lv_jac!(jp_dense, u0, p, 0.0)
    dgdu_cont = (out, u, p, t) -> (out .= 1)
    results = map((sparse(jp_dense), copy(jp_dense))) do jp
        f = ODEFunction(lotka_volterra!; jac = lv_jac!, jac_prototype = jp)
        prob = ODEProblem(f, u0, tspan, p)
        sol = solve(prob, Rodas5P(); solver_kwargs...)
        map(
            (
                GaussAdjoint(autojacvec = ReverseDiffVJP()),
                QuadratureAdjoint(
                    autojacvec = ReverseDiffVJP(), abstol = 1.0e-10, reltol = 1.0e-10
                ),
            )
        ) do sa
            du0, dp = adjoint_sensitivities(
                sol, Rodas5P(); dgdu_continuous = dgdu_cont,
                sensealg = sa, abstol = 1.0e-8, reltol = 1.0e-8
            )
            vec(dp)
        end
    end
    dp_sparse, dp_dense = results
    for (dps, dpd) in zip(dp_sparse, dp_dense)
        @test isapprox(dps, dpd, rtol = 1.0e-6)
    end
end
