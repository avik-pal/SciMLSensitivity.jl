module SciMLSensitivitySundialsExt

using SciMLSensitivity: SciMLSensitivity, SciMLBase, SciMLLogging,
    SundialsAdjoint, QuadratureAdjoint, GaussAdjoint,
    ODEQuadratureAdjointSensitivityFunction, GaussIntegrand,
    vecjacobian!, vec_pjac!, accumulate_cost!, inplace_vjp,
    ReverseDiffVJP, mutable_zeros, compile_tape, ReverseDiff,
    canonicalize, Tunable, isscimlstructure, unwrapped_f,
    get_dae_paramjac_config, dae_du_jacobian, dae_u_jacobian,
    dae_adjoint_structure
import SciMLSensitivity: _adjoint_sensitivities
using SciMLBase: ODEProblem, DAEProblem, isinplace
using Sundials: Sundials, N_Vector, NVector, realtype
using LinearAlgebra: I, UniformScaling, pinv, norm, mul!

_cvodes_lmm(::Sundials.CVODE_BDF) = Sundials.CV_BDF
_cvodes_lmm(::Sundials.CVODE_Adams) = Sundials.CV_ADAMS
_cvodes_method(::Sundials.SundialsODEAlgorithm{M, LS}) where {M, LS} = M
_cvodes_linear_solver(::Sundials.SundialsODEAlgorithm{M, LS}) where {M, LS} = LS

function _check_cvodes_flag(flag, fname)
    return flag >= Sundials.CV_SUCCESS ||
        error("SundialsAdjoint: $fname failed with error code = $flag")
end

# User data passed through the CVODES C callbacks via `CVodeSetUserData(B)`.
# `S` carries the state vjp machinery (`vecjacobian!`) and `integrand` the
# parameter vjp machinery (`vec_pjac!`); both are the standard SciMLSensitivity
# caches so all `autojacvec` choices work unchanged.
mutable struct CVODESAdjointUserData{ND, F, P, S, GI, DL, PQ, GQ}
    const f::F
    const p::P
    const sz::NTuple{ND, Int}
    const n::Int
    const S::S
    const integrand::GI
    const dλ::DL
    const pquad::PQ
    const continuous_cost::Bool
    const gquad::GQ
end

function _cvodes_forward_rhs(
        t::realtype, y_nv::N_Vector, ydot_nv::N_Vector,
        data::CVODESAdjointUserData{ND}
    ) where {ND}
    y = unsafe_wrap(Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(y_nv), data.sz)
    dy = unsafe_wrap(
        Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(ydot_nv),
        data.sz
    )
    data.f(dy, y, data.p, t)
    return Sundials.CV_SUCCESS
end

# yB' = -(df/du)^T yB, evaluated at the CVODES checkpoint-interpolated forward
# state `y` which the backward integrator hands to this callback.
function _cvodes_adjoint_rhs(
        t::realtype, y_nv::N_Vector, yB_nv::N_Vector,
        yBdot_nv::N_Vector, data::CVODESAdjointUserData{ND}
    ) where {ND}
    y = unsafe_wrap(Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(y_nv), data.sz)
    λ = unsafe_wrap(Vector{Float64}, Sundials.N_VGetArrayPointer_Serial(yB_nv), data.n)
    out = unsafe_wrap(
        Vector{Float64}, Sundials.N_VGetArrayPointer_Serial(yBdot_nv),
        data.n
    )
    vecjacobian!(data.dλ, y, λ, data.p, t, data.S)
    @. out = -data.dλ
    data.continuous_cost && accumulate_cost!(out, y, data.p, t, data.S)
    return Sundials.CV_SUCCESS
end

# qB' = -((df/dp)^T yB + dg/dp). CVODES integrates qB backward from tf with
# qB(tf) = 0, so the value returned at t0 is +∫_{t0}^{tf} (λ^T df/dp + dg/dp) dt.
function _cvodes_quad_rhs(
        t::realtype, y_nv::N_Vector, yB_nv::N_Vector,
        qBdot_nv::N_Vector, data::CVODESAdjointUserData{ND}
    ) where {ND}
    y = unsafe_wrap(Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(y_nv), data.sz)
    λ = unsafe_wrap(Vector{Float64}, Sundials.N_VGetArrayPointer_Serial(yB_nv), data.n)
    out = unsafe_wrap(
        Vector{Float64}, Sundials.N_VGetArrayPointer_Serial(qBdot_nv),
        length(data.pquad)
    )
    vec_pjac!(data.pquad, λ, y, t, data.integrand)
    pq = vec(data.pquad)
    @. out = -pq
    if data.continuous_cost
        # `accumulate_cost!` writes `-dg/dp` into `gquad` (the dλ contribution
        # goes into the scratch `dλ` buffer and is discarded).
        fill!(data.gquad, false)
        accumulate_cost!(data.dλ, y, data.p, t, data.S, data.gquad)
        out .+= data.gquad
    end
    return Sundials.CV_SUCCESS
end

function _forward_cfunction(::T) where {T <: CVODESAdjointUserData}
    return @cfunction(
        _cvodes_forward_rhs, Cint,
        (realtype, N_Vector, N_Vector, Ref{T})
    )
end

function _adjoint_cfunction(::T) where {T <: CVODESAdjointUserData}
    return @cfunction(
        _cvodes_adjoint_rhs, Cint,
        (realtype, N_Vector, N_Vector, N_Vector, Ref{T})
    )
end

function _quad_cfunction(::T) where {T <: CVODESAdjointUserData}
    return @cfunction(
        _cvodes_quad_rhs, Cint,
        (realtype, N_Vector, N_Vector, N_Vector, Ref{T})
    )
end

# Attaches the nonlinear/linear solver combination implied by `alg` via the
# supplied setter closures and returns the created objects so the caller can
# keep them rooted for the duration of the integration.
function _cvodes_set_solvers(ls_setter, nls_setter, alg, unv, n, ctx)
    linear_solver = _cvodes_linear_solver(alg)
    if _cvodes_method(alg) == :Functional
        nls = Sundials.SUNNonlinSol_FixedPoint(unv, Cint(0), ctx)
        nls_handle = Sundials.NonLinSolHandle(nls, Sundials.FixedPoint())
        _check_cvodes_flag(nls_setter(nls), "CVodeSetNonlinearSolver")
        return nls_handle, nothing, nothing
    elseif linear_solver == :Dense
        A = Sundials.SUNDenseMatrix(n, n, ctx)
        LS = Sundials.SUNLinSol_Dense(unv, A, ctx)
        _check_cvodes_flag(ls_setter(LS, A), "CVodeSetLinearSolver")
        return nothing, LS, A
    elseif linear_solver == :Band
        A = Sundials.SUNBandMatrix(n, alg.jac_upper, alg.jac_lower, ctx)
        LS = Sundials.SUNLinSol_Band(unv, A, ctx)
        _check_cvodes_flag(ls_setter(LS, A), "CVodeSetLinearSolver")
        return nothing, LS, A
    elseif linear_solver == :GMRES
        krylov_dim = alg.krylov_dim == 0 ? 5 : alg.krylov_dim
        LS = Sundials.SUNLinSol_SPGMR(unv, Cint(Sundials.PREC_NONE), Cint(krylov_dim), ctx)
        # A typed null pointer is required for dispatch to reach the ccall method.
        nullmat = Sundials.SUNMatrix(C_NULL)
        _check_cvodes_flag(ls_setter(LS, nullmat), "CVodeSetLinearSolver")
        return nothing, LS, nothing
    else
        error(
            "SundialsAdjoint currently supports the `:Dense`, `:Band`, and `:GMRES` " *
                "linear solvers (or `method = :Functional`), got `$(linear_solver)`."
        )
    end
end

function _adjoint_sensitivities(
        sol, sensealg::SundialsAdjoint{CS, AD, FDT},
        alg::Union{Sundials.CVODE_BDF, Sundials.CVODE_Adams};
        t = nothing,
        dgdu_discrete = nothing,
        dgdp_discrete = nothing,
        dgdu_continuous = nothing,
        dgdp_continuous = nothing,
        g = nothing, no_start = false,
        abstol = 1.0e-6, reltol = 1.0e-3,
        maxiters = Int(1.0e5),
        verbose = SciMLLogging.Standard(),
        kwargs...
    ) where {CS, AD, FDT}
    continuous_cost = dgdu_continuous !== nothing || dgdp_continuous !== nothing ||
        g !== nothing
    if continuous_cost && dgdu_continuous === nothing && g === nothing
        error("SundialsAdjoint requires `dgdu_continuous` or `g` alongside `dgdp_continuous`.")
    end
    if g !== nothing && t !== nothing
        error(
            "SundialsAdjoint does not support the discrete scalar cost form " *
                "(`g` together with `t`). Provide the discrete cost derivative " *
                "directly via `dgdu_discrete`."
        )
    end
    if t !== nothing && dgdu_discrete === nothing
        error("SundialsAdjoint requires `dgdu_discrete` when `t` is specified.")
    end
    if t === nothing && !continuous_cost
        error(
            "SundialsAdjoint requires a cost functional: either `t` with " *
                "`dgdu_discrete`, or a continuous cost via `dgdu_continuous`/" *
                "`dgdp_continuous` or `g`."
        )
    end

    prob = sol.prob
    prob isa ODEProblem ||
        error(
        "SundialsAdjoint with `CVODE_BDF`/`CVODE_Adams` only supports `ODEProblem`s. " *
            "For a `DAEProblem`, use `IDA()` as the solver instead."
    )
    isinplace(prob) ||
        error("SundialsAdjoint currently only supports in-place (mutating) `ODEProblem`s.")
    mm = prob.f.mass_matrix
    (mm isa UniformScaling && mm == I) ||
        error("SundialsAdjoint does not support mass matrices (a CVODES restriction).")
    u0 = prob.u0
    eltype(u0) === Float64 && eltype(prob.tspan) === Float64 ||
        error("SundialsAdjoint requires `Float64` state and time (a SUNDIALS restriction).")
    t0, tf = prob.tspan
    t0 < tf || error("SundialsAdjoint requires a forward time span with `tspan[1] < tspan[2]`.")

    ts = t === nothing ? Float64[] : collect(Float64, t)
    issorted(ts) || error("SundialsAdjoint requires the cost times `t` to be sorted in ascending order.")
    if !isempty(ts) && !(first(ts) >= t0 && last(ts) <= tf)
        error("SundialsAdjoint requires all cost times `t` to lie within the problem `tspan`.")
    end

    p = prob.p
    has_p = !(p === nothing || p isa SciMLBase.NullParameters)
    if has_p
        if isscimlstructure(p) && !(p isa AbstractArray)
            tunables, repack, _ = canonicalize(Tunable(), p)
        elseif p isa AbstractArray
            tunables, repack = p, identity
        else
            error(
                "SundialsAdjoint requires the parameters to be an `AbstractArray` or a " *
                    "SciMLStructures-compatible struct, got `$(typeof(p))`."
            )
        end
    else
        tunables, repack = nothing, identity
    end

    # The standard automatic vjp choice; `adjoint_sensitivities` already applies
    # this before dispatching here, so this only triggers on direct calls.
    vjp = if sensealg.autojacvec === nothing
        has_p ? inplace_vjp(prob, u0, p, verbose, repack) : ReverseDiffVJP()
    else
        sensealg.autojacvec
    end
    vjp === true &&
        error(
        "SundialsAdjoint does not support `autojacvec = true`. Use `autojacvec = false` " *
            "(Jacobian construction controlled by `autodiff`) or a vjp choice such as " *
            "`ReverseDiffVJP()` or `EnzymeVJP()`."
    )

    # Reuse the existing quadrature/Gauss adjoint vjp caches: `S` computes the
    # state vjp `(df/du)^T λ` via `vecjacobian!` (plus the continuous-cost
    # `dg/du` term via `accumulate_cost!`) and `integrand` the parameter vjp
    # `(df/dp)^T λ` via `vec_pjac!`.
    S = ODEQuadratureAdjointSensitivityFunction(
        g,
        QuadratureAdjoint(
            chunk_size = CS, autodiff = AD, diff_type = FDT,
            autojacvec = vjp
        ),
        !continuous_cost, sol, dgdu_continuous, dgdp_continuous, alg
    )
    integrand = has_p ?
        GaussIntegrand(
            sol,
            GaussAdjoint(
                chunk_size = CS, autodiff = AD, diff_type = FDT,
                autojacvec = vjp
            ),
            nothing
        ) : nothing

    n = length(u0)
    np = has_p ? length(tunables) : 0
    f = unwrapped_f(prob.f)
    data = CVODESAdjointUserData(
        f, p, size(u0), n, S, integrand,
        zeros(n), has_p ? mutable_zeros(tunables) : nothing,
        continuous_cost, continuous_cost && has_p ? zeros(np) : nothing
    )
    fwd_cfun = _forward_cfunction(data)
    adj_cfun = _adjoint_cfunction(data)
    quad_cfun = has_p ? _quad_cfunction(data) : nothing

    interp = sensealg.interp === :hermite ? Sundials.CV_HERMITE : Sundials.CV_POLYNOMIAL
    lmm = _cvodes_lmm(alg)

    ctx_ref = Ref{Sundials.SUNContext}(C_NULL)
    Sundials.SUNContext_Create(
        C_NULL,
        Base.unsafe_convert(Ptr{Sundials.SUNContext}, ctx_ref)
    )
    ctx = ctx_ref[]
    mem = Sundials.Handle(Sundials.CVodeCreate(lmm, ctx))

    ufwd = collect(vec(copy(u0)))
    ufwd_nv = NVector(ufwd, ctx)
    yret = similar(ufwd)
    yret_nv = NVector(yret, ctx)
    λ = zeros(n)
    λ_nv = NVector(λ, ctx)
    qB = zeros(np)
    qB_nv = has_p ? NVector(qB, ctx) : nothing
    gu = zeros(n)
    gp = dgdp_discrete === nothing ? nothing : zeros(np)
    dp = has_p ? zeros(np) : nothing
    tret = [t0]
    ncheck = Ref{Cint}(0)
    which = Ref{Cint}(0)

    # The solver objects created inside the `try` must stay referenced until the
    # `finally` cleanup: `NonLinSolHandle` carries a finalizer that frees the
    # SUNDIALS nonlinear solver, so letting the binding go dead mid-integration
    # lets GC free memory CVODES is still using (segfault); the linear solver and
    # matrix have no finalizers and would leak without the explicit frees.
    local nlsf = nothing
    local LSf = nothing
    local Af = nothing
    local nlsB = nothing
    local LSB = nothing
    local AB = nothing

    GC.@preserve data ufwd_nv yret_nv λ_nv qB_nv begin
        try
            # Forward pass with checkpointing.
            _check_cvodes_flag(
                Sundials.CVodeInit(mem, fwd_cfun, t0, ufwd_nv),
                "CVodeInit"
            )
            _check_cvodes_flag(Sundials.CVodeSetUserData(mem, data), "CVodeSetUserData")
            _check_cvodes_flag(
                Sundials.CVodeSStolerances(mem, reltol, abstol),
                "CVodeSStolerances"
            )
            _check_cvodes_flag(
                Sundials.CVodeSetMaxNumSteps(mem, maxiters),
                "CVodeSetMaxNumSteps"
            )
            nlsf, LSf,
                Af = _cvodes_set_solvers(
                (LS, A) -> Sundials.CVodeSetLinearSolver(mem, LS, A),
                NLS -> Sundials.CVodeSetNonlinearSolver(mem, NLS),
                alg, ufwd_nv, n, ctx
            )
            _check_cvodes_flag(
                Sundials.CVodeAdjInit(mem, sensealg.steps, interp),
                "CVodeAdjInit"
            )
            _check_cvodes_flag(
                Sundials.CVodeF(mem, tf, yret_nv, tret, Sundials.CV_NORMAL, ncheck),
                "CVodeF"
            )

            # Backward (adjoint) problem. λ(tf) collects the cost jumps at tf.
            cur_time = length(ts)
            if cur_time >= 1 && ts[cur_time] == tf
                y_f = sol(tf)
                while cur_time >= 1 && ts[cur_time] == tf
                    if !(no_start && cur_time == 1)
                        fill!(gu, false)
                        dgdu_discrete(gu, y_f, p, tf, cur_time)
                        λ .+= gu
                        if dgdp_discrete !== nothing
                            fill!(gp, false)
                            dgdp_discrete(gp, y_f, p, tf, cur_time)
                            dp .+= gp
                        end
                    end
                    cur_time -= 1
                end
            end

            _check_cvodes_flag(Sundials.CVodeCreateB(mem, lmm, which), "CVodeCreateB")
            _check_cvodes_flag(
                Sundials.CVodeInitB(mem, which[], adj_cfun, tf, λ_nv),
                "CVodeInitB"
            )
            _check_cvodes_flag(
                Sundials.CVodeSetUserDataB(mem, which[], data),
                "CVodeSetUserDataB"
            )
            _check_cvodes_flag(
                Sundials.CVodeSStolerancesB(mem, which[], reltol, abstol),
                "CVodeSStolerancesB"
            )
            _check_cvodes_flag(
                Sundials.CVodeSetMaxNumStepsB(mem, which[], maxiters),
                "CVodeSetMaxNumStepsB"
            )
            nlsB, LSB,
                AB = _cvodes_set_solvers(
                (LS, A) -> Sundials.CVodeSetLinearSolverB(mem, which[], LS, A),
                NLS -> Sundials.CVodeSetNonlinearSolverB(mem, which[], NLS),
                alg, λ_nv, n, ctx
            )
            if has_p
                _check_cvodes_flag(
                    Sundials.CVodeQuadInitB(mem, which[], quad_cfun, qB_nv),
                    "CVodeQuadInitB"
                )
                _check_cvodes_flag(
                    Sundials.CVodeQuadSStolerancesB(mem, which[], reltol, abstol),
                    "CVodeQuadSStolerancesB"
                )
                _check_cvodes_flag(
                    Sundials.CVodeSetQuadErrConB(
                        mem, which[],
                        sensealg.quad_error_control ? 1 : 0
                    ),
                    "CVodeSetQuadErrConB"
                )
            end

            # Integrate backward, stopping at every discrete cost time to add the
            # jump `λ += dg/du` and reinitialize the backward problem.
            tcur = tf
            while cur_time >= 1
                s = ts[cur_time]
                if s < tcur && s > t0
                    _check_cvodes_flag(
                        Sundials.CVodeB(mem, s, Sundials.CV_NORMAL),
                        "CVodeB"
                    )
                    _check_cvodes_flag(
                        Sundials.CVodeGetB(mem, which[], tret, λ_nv),
                        "CVodeGetB"
                    )
                    if has_p
                        _check_cvodes_flag(
                            Sundials.CVodeGetQuadB(mem, which[], tret, qB_nv),
                            "CVodeGetQuadB"
                        )
                    end
                    tcur = s
                    y_s = sol(s)
                    while cur_time >= 1 && ts[cur_time] == s
                        if !(no_start && cur_time == 1)
                            fill!(gu, false)
                            dgdu_discrete(gu, y_s, p, s, cur_time)
                            λ .+= gu
                            if dgdp_discrete !== nothing
                                fill!(gp, false)
                                dgdp_discrete(gp, y_s, p, s, cur_time)
                                dp .+= gp
                            end
                        end
                        cur_time -= 1
                    end
                    _check_cvodes_flag(
                        Sundials.CVodeReInitB(mem, which[], s, λ_nv),
                        "CVodeReInitB"
                    )
                    if has_p
                        _check_cvodes_flag(
                            Sundials.CVodeQuadReInitB(mem, which[], qB_nv),
                            "CVodeQuadReInitB"
                        )
                    end
                else
                    break
                end
            end

            # Final leg down to t0 and any jump exactly at t0.
            if tcur > t0
                _check_cvodes_flag(
                    Sundials.CVodeB(mem, t0, Sundials.CV_NORMAL),
                    "CVodeB"
                )
                _check_cvodes_flag(
                    Sundials.CVodeGetB(mem, which[], tret, λ_nv),
                    "CVodeGetB"
                )
                if has_p
                    _check_cvodes_flag(
                        Sundials.CVodeGetQuadB(mem, which[], tret, qB_nv),
                        "CVodeGetQuadB"
                    )
                end
            end
            while cur_time >= 1
                s = ts[cur_time]
                s == t0 ||
                    error("SundialsAdjoint: internal error, unprocessed cost time $(s).")
                if !(no_start && cur_time == 1)
                    y_s = sol(s)
                    fill!(gu, false)
                    dgdu_discrete(gu, y_s, p, s, cur_time)
                    λ .+= gu
                    if dgdp_discrete !== nothing
                        fill!(gp, false)
                        dgdp_discrete(gp, y_s, p, s, cur_time)
                        dp .+= gp
                    end
                end
                cur_time -= 1
            end

            has_p && (dp .+= qB)
        finally
            empty!(mem)
            nlsf === nothing || empty!(nlsf)
            nlsB === nothing || empty!(nlsB)
            LSf === nothing || Sundials.SUNLinSolFree(LSf)
            LSB === nothing || Sundials.SUNLinSolFree(LSB)
            Af === nothing || Sundials.SUNMatDestroy(Af)
            AB === nothing || Sundials.SUNMatDestroy(AB)
            Sundials.SUNContext_Free(ctx)
        end
    end

    du0 = copy(λ)
    return du0, has_p ? dp' : nothing
end

# ===== IDAS adjoint for fully implicit DAEProblems =====
#
# Backward problem in residual form (Cao–Li–Petzold; IDAS user guide):
#
#     resB = (∂F/∂du)ᵀ ypB − (∂F/∂u)ᵀ yB
#
# which is the adjoint DAE  d/dt[(∂F/∂du)ᵀ λ] − (∂F/∂u)ᵀ λ = 0  under the
# assumption that ∂F/∂du is CONSTANT (F linear in `du` with state- and
# time-independent coefficients), so the d/dt(∂F/∂du)ᵀ λ term vanishes.
# The parameter gradient is the backward quadrature  qB' = (∂F/∂p)ᵀ λ  with
# qB(tf) = 0, giving  dG/dp = qB(t0) = −∫ λᵀ (∂F/∂p) dt, and the gradient
# with respect to the initial state is  (∂F/∂du)ᵀ λ |_{t0}  (plus any cost
# jump exactly at t0). These conventions reproduce the analytic gradients of
# the SUNDIALS `idas` adjoint examples.

function _check_idas_flag(flag, fname)
    return flag >= Sundials.IDA_SUCCESS ||
        error("SundialsAdjoint: $fname failed with error code = $flag")
end

_idas_linear_solver(::Sundials.SundialsDAEAlgorithm{LS}) where {LS} = LS

function _idas_set_solvers(ls_setter, alg, unv, n, ctx)
    linear_solver = _idas_linear_solver(alg)
    if linear_solver == :Dense
        A = Sundials.SUNDenseMatrix(n, n, ctx)
        LS = Sundials.SUNLinSol_Dense(unv, A, ctx)
        _check_idas_flag(ls_setter(LS, A), "IDASetLinearSolver")
        return LS, A
    elseif linear_solver == :Band
        A = Sundials.SUNBandMatrix(n, alg.jac_upper, alg.jac_lower, ctx)
        LS = Sundials.SUNLinSol_Band(unv, A, ctx)
        _check_idas_flag(ls_setter(LS, A), "IDASetLinearSolver")
        return LS, A
    elseif linear_solver == :GMRES
        krylov_dim = alg.krylov_dim == 0 ? 5 : alg.krylov_dim
        LS = Sundials.SUNLinSol_SPGMR(unv, Cint(Sundials.PREC_NONE), Cint(krylov_dim), ctx)
        # A typed null pointer is required for dispatch to reach the ccall method.
        nullmat = Sundials.SUNMatrix(C_NULL)
        _check_idas_flag(ls_setter(LS, nullmat), "IDASetLinearSolver")
        return LS, nothing
    else
        error(
            "SundialsAdjoint with `IDA` currently supports the `:Dense`, `:Band`, and " *
                "`:GMRES` linear solvers, got `$(linear_solver)`."
        )
    end
end

# User data passed through the IDAS C callbacks via `IDASetUserData(B)`. With
# `compile = true` the recorded `tape` is reused (which freezes control flow,
# the standard `ReverseDiffVJP(true)` restriction); otherwise a fresh tape is
# recorded at every evaluation point, matching the `vecjacobian!` convention.
# Tapes are always recorded over `(du, u, p, t)` via the shared
# `get_dae_paramjac_config` builder, with an empty parameter input when the
# problem has no parameters.
mutable struct IDASAdjointUserData{ND, F, P, RP, TU, T}
    const f::F
    const p::P
    const repack::RP
    const tunables::TU
    const has_p::Bool
    const sz::NTuple{ND, Int}
    const n::Int
    const tape::T
    const dλ::Vector{Float64}
    const ddu::Vector{Float64}
    const tscratch::Vector{Float64}
end

# Computes vector-Jacobian products of the DAE residual at the forward point
# `(yp, y, t)`: seeds `yB` to get `(∂F/∂u)ᵀ yB` into `data.dλ` (and, when
# `dgrad !== nothing`, `(∂F/∂p)ᵀ yB` into `dgrad`), and seeds `ypB` to get
# `(∂F/∂du)ᵀ ypB` into `data.ddu`. Both seeds share one tape forward pass;
# ReverseDiff clears instruction-output derivatives during the reverse pass,
# so only the input hooks need unseeding between the two reverse passes.
function _idas_res_vjps!(data::IDASAdjointUserData, y, yp, t, yB, ypB, dgrad)
    tape = data.tape === nothing ?
        get_dae_paramjac_config(
            ReverseDiffVJP(), data.p, data.f,
            collect(Float64, y), collect(Float64, yp),
            data.tunables, t; isinplace = true
        ) : data.tape
    tdu, tu, tp, tt = ReverseDiff.input_hook(tape)
    output = ReverseDiff.output_hook(tape)
    ReverseDiff.unseed!(tdu)
    ReverseDiff.unseed!(tu)
    ReverseDiff.unseed!(tp)
    ReverseDiff.unseed!(tt)
    ReverseDiff.value!(tdu, yp)
    ReverseDiff.value!(tu, y)
    data.has_p && ReverseDiff.value!(tp, data.tunables)
    data.tscratch[1] = t
    ReverseDiff.value!(tt, data.tscratch)
    ReverseDiff.forward_pass!(tape)
    if yB !== nothing
        ReverseDiff.increment_deriv!(output, yB)
        ReverseDiff.reverse_pass!(tape)
        copyto!(vec(data.dλ), ReverseDiff.deriv(tu))
        dgrad === nothing || copyto!(vec(dgrad), ReverseDiff.deriv(tp))
        if ypB !== nothing
            ReverseDiff.unseed!(tdu)
            ReverseDiff.unseed!(tu)
            ReverseDiff.unseed!(tp)
            ReverseDiff.unseed!(tt)
        end
    end
    if ypB !== nothing
        ReverseDiff.increment_deriv!(output, ypB)
        ReverseDiff.reverse_pass!(tape)
        copyto!(vec(data.ddu), ReverseDiff.deriv(tdu))
    end
    return nothing
end

function _idas_forward_res(
        t::realtype, yy_nv::N_Vector, yp_nv::N_Vector, rr_nv::N_Vector,
        data::IDASAdjointUserData{ND}
    ) where {ND}
    y = unsafe_wrap(Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(yy_nv), data.sz)
    dy = unsafe_wrap(Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(yp_nv), data.sz)
    res = unsafe_wrap(
        Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(rr_nv),
        data.sz
    )
    data.f(res, dy, y, data.p, t)
    return Sundials.IDA_SUCCESS
end

function _idas_adjoint_res(
        t::realtype, yy_nv::N_Vector, yp_nv::N_Vector, yyB_nv::N_Vector,
        ypB_nv::N_Vector, rrB_nv::N_Vector, data::IDASAdjointUserData{ND}
    ) where {ND}
    y = unsafe_wrap(Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(yy_nv), data.sz)
    dy = unsafe_wrap(Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(yp_nv), data.sz)
    λ = unsafe_wrap(Vector{Float64}, Sundials.N_VGetArrayPointer_Serial(yyB_nv), data.n)
    λp = unsafe_wrap(Vector{Float64}, Sundials.N_VGetArrayPointer_Serial(ypB_nv), data.n)
    out = unsafe_wrap(Vector{Float64}, Sundials.N_VGetArrayPointer_Serial(rrB_nv), data.n)
    _idas_res_vjps!(data, y, dy, t, λ, λp, nothing)
    @. out = data.ddu - data.dλ
    return Sundials.IDA_SUCCESS
end

function _idas_quad_rhs(
        t::realtype, yy_nv::N_Vector, yp_nv::N_Vector, yyB_nv::N_Vector,
        ypB_nv::N_Vector, qBdot_nv::N_Vector, data::IDASAdjointUserData{ND}
    ) where {ND}
    y = unsafe_wrap(Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(yy_nv), data.sz)
    dy = unsafe_wrap(Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(yp_nv), data.sz)
    λ = unsafe_wrap(Vector{Float64}, Sundials.N_VGetArrayPointer_Serial(yyB_nv), data.n)
    out = unsafe_wrap(
        Vector{Float64}, Sundials.N_VGetArrayPointer_Serial(qBdot_nv),
        length(data.tunables)
    )
    _idas_res_vjps!(data, y, dy, t, λ, nothing, out)
    return Sundials.IDA_SUCCESS
end

function _idas_forward_cfunction(::T) where {T <: IDASAdjointUserData}
    return @cfunction(
        _idas_forward_res, Cint,
        (realtype, N_Vector, N_Vector, N_Vector, Ref{T})
    )
end

function _idas_adjoint_cfunction(::T) where {T <: IDASAdjointUserData}
    return @cfunction(
        _idas_adjoint_res, Cint,
        (realtype, N_Vector, N_Vector, N_Vector, N_Vector, N_Vector, Ref{T})
    )
end

function _idas_quad_cfunction(::T) where {T <: IDASAdjointUserData}
    return @cfunction(
        _idas_quad_rhs, Cint,
        (realtype, N_Vector, N_Vector, N_Vector, N_Vector, N_Vector, Ref{T})
    )
end

# Transfers a discrete cost jump `gu` onto the adjoint variable w = (∂F/∂du)ᵀλ
# at a cost time, applying the index-1 consistency correction shared with the
# native `ReverseLossCallback` machinery when the cost touches algebraic
# variables:
#
#     Δλa = -(dhda' \ gu[algvars]),   Δw[diffvars] = gu[diffvars] + dhdd' Δλa
#
# with the parameter-gradient correction Δλa' ∂F_algeqs/∂p accumulated into
# `dp_corr` (when non-`nothing`). Returns the jump Δw in w-space and the λ jump
# solving `(∂F/∂du)ᵀ Δλ = Δw`.
function _idas_cost_jump!(dp_corr, data, structure, Mtpinv, Mt, gu, y_s, yp_s, s)
    diffvar_idxs, algevar_idxs, diffeq_idxs, algeeq_idxs = structure
    Δw = gu
    if !isempty(algevar_idxs)
        J_u = dae_u_jacobian(data.f, true, yp_s, y_s, data.p, s)
        dhdd = J_u[algeeq_idxs, diffvar_idxs]
        dhda = J_u[algeeq_idxs, algevar_idxs]
        Δλa = -(dhda' \ gu[algevar_idxs])
        Δw = zeros(length(gu))
        Δw[diffvar_idxs] .= @view(gu[diffvar_idxs])
        Δw[diffvar_idxs] .+= dhdd' * Δλa
        if dp_corr !== nothing
            iλ = zeros(length(gu))
            iλ[algeeq_idxs] .= Δλa
            out = similar(dp_corr)
            _idas_res_vjps!(data, y_s, yp_s, s, iλ, nothing, out)
            dp_corr .+= out
        end
    end
    Δλ = Mtpinv * Δw
    resid = Mt * Δλ .- Δw
    if norm(resid) > max(1.0e-10, 1.0e-8 * norm(Δw))
        error(
            "SundialsAdjoint with `IDA` could not transfer the discrete cost jump " *
                "onto the adjoint at t = $s: the (index-1 corrected) jump is not in " *
                "the range of `(∂F/∂du)ᵀ`. This typically indicates a higher-index " *
                "or structurally degenerate DAE; use the native " *
                "`InterpolatingAdjoint` DAEProblem support instead."
        )
    end
    return Δλ, Δw
end

function _adjoint_sensitivities(
        sol, sensealg::SundialsAdjoint{CS, AD, FDT},
        alg::Sundials.IDA;
        t = nothing,
        dgdu_discrete = nothing,
        dgdp_discrete = nothing,
        dgdu_continuous = nothing,
        dgdp_continuous = nothing,
        g = nothing, no_start = false,
        abstol = 1.0e-6, reltol = 1.0e-3,
        maxiters = Int(1.0e5),
        verbose = SciMLLogging.Standard(),
        kwargs...
    ) where {CS, AD, FDT}
    prob = sol.prob
    prob isa DAEProblem ||
        error(
        "SundialsAdjoint with `IDA` only supports `DAEProblem`s. For an " *
            "`ODEProblem`, use `CVODE_BDF()` or `CVODE_Adams()` as the solver instead."
    )
    isinplace(prob) ||
        error("SundialsAdjoint currently only supports in-place (mutating) `DAEProblem`s.")
    u0 = prob.u0
    du0 = prob.du0
    eltype(u0) === Float64 && eltype(prob.tspan) === Float64 ||
        error("SundialsAdjoint requires `Float64` state and time (a SUNDIALS restriction).")
    t0, tf = prob.tspan
    t0 < tf || error("SundialsAdjoint requires a forward time span with `tspan[1] < tspan[2]`.")
    diffvars = prob.differential_vars
    diffvars === nothing &&
        error(
        "SundialsAdjoint with `IDA` requires the `DAEProblem` to specify " *
            "`differential_vars` (needed for consistent initialization of the " *
            "backward problem via `IDACalcICB`)."
    )

    if dgdu_continuous !== nothing || dgdp_continuous !== nothing || g !== nothing
        error(
            "SundialsAdjoint with `IDA` currently only supports discrete cost " *
                "functionals (`t` with `dgdu_discrete`). Continuous cost functionals " *
                "(`g`/`dgdu_continuous`/`dgdp_continuous`) are not yet implemented " *
                "for `DAEProblem`s."
        )
    end
    t !== nothing && dgdu_discrete !== nothing ||
        error(
        "SundialsAdjoint with `IDA` requires a discrete cost functional: pass " *
            "the cost times `t` together with `dgdu_discrete`."
    )

    ts = collect(Float64, t)
    issorted(ts) || error("SundialsAdjoint requires the cost times `t` to be sorted in ascending order.")
    if !isempty(ts) && !(first(ts) >= t0 && last(ts) <= tf)
        error("SundialsAdjoint requires all cost times `t` to lie within the problem `tspan`.")
    end

    p = prob.p
    has_p = !(p === nothing || p isa SciMLBase.NullParameters)
    if has_p
        if isscimlstructure(p) && !(p isa AbstractArray)
            tunables, repack, _ = canonicalize(Tunable(), p)
        elseif p isa AbstractArray
            tunables, repack = p, identity
        else
            error(
                "SundialsAdjoint requires the parameters to be an `AbstractArray` or a " *
                    "SciMLStructures-compatible struct, got `$(typeof(p))`."
            )
        end
    else
        tunables, repack = Float64[], identity
    end

    # `adjoint_sensitivities` resolves `autojacvec === nothing` via
    # `inplace_vjp`, which returns `ReverseDiffVJP()` for DAEProblems; the
    # `nothing` branch here only triggers on direct `_adjoint_sensitivities`
    # calls.
    vjp = sensealg.autojacvec === nothing ? ReverseDiffVJP() : sensealg.autojacvec
    vjp isa ReverseDiffVJP ||
        error(
        "SundialsAdjoint with `IDA` currently only supports " *
            "`autojacvec = ReverseDiffVJP()` for the DAE residual vector-Jacobian " *
            "products, got `$(typeof(vjp))`."
    )

    n = length(u0)
    np = length(tunables)
    f = unwrapped_f(prob.f)
    u0v = collect(Float64, vec(u0))
    du0v = collect(Float64, vec(du0))
    tape = compile_tape(vjp) ?
        get_dae_paramjac_config(
            vjp, p, f, u0v, du0v, has_p ? collect(Float64, tunables) : Float64[],
            t0; isinplace = true
        ) : nothing
    data = IDASAdjointUserData(
        f, p, repack, has_p ? mutable_zeros(tunables) : Float64[], has_p,
        size(u0), n, tape, zeros(n), zeros(n), [t0]
    )
    has_p && copyto!(data.tunables, tunables)
    fwd_cfun = _idas_forward_cfunction(data)
    adj_cfun = _idas_adjoint_cfunction(data)
    quad_cfun = has_p ? _idas_quad_cfunction(data) : nothing

    # `Mt = (∂F/∂du)ᵀ`, constant by assumption; used to transfer discrete cost
    # jumps onto the adjoint variables and for the `du0` boundary term.
    Mt = collect(dae_du_jacobian(f, true, du0v, u0v, p, t0)')
    Mtpinv = pinv(Mt)

    # Algebraic equation/variable structure for the index-1 cost-jump
    # correction (shared with the native DAE adjoint machinery). Hessenberg
    # index-2 systems are rejected on this route: `IDACalcICB` cannot compute
    # consistent backward initial conditions for them.
    diffvar_idxs, algevar_idxs, diffeq_idxs, algeeq_idxs,
        index2 = dae_adjoint_structure(
        f, true, du0v, u0v, p, t0, diffvars
    )
    index2 &&
        error(
        "SundialsAdjoint with `IDA` does not support Hessenberg index-2 DAEs. " *
            "Use the native `InterpolatingAdjoint`/`QuadratureAdjoint`/`GaussAdjoint` " *
            "DAEProblem support instead."
    )
    structure = (diffvar_idxs, algevar_idxs, diffeq_idxs, algeeq_idxs)

    # Best-effort screen for the constant-∂F/∂du assumption: compare the vjp at
    # a second, perturbed point. Skipped if the residual cannot be evaluated
    # there (e.g. domain errors).
    let w = [1.0 + i / n for i in 1:n]
        try
            _idas_res_vjps!(
                data, u0v .+ 0.5, du0v .+ 1.0, (t0 + tf) / 2, nothing, w,
                nothing
            )
            if !isapprox(data.ddu, Mt * w; rtol = 1.0e-6, atol = 1.0e-10)
                error(
                    "SundialsAdjoint with `IDA` requires the DAE residual " *
                        "`F(du, u, p, t)` to have a constant Jacobian `∂F/∂du` (i.e. " *
                        "F linear in `du` with state- and time-independent " *
                        "coefficients); a state/time-dependent `∂F/∂du` was detected."
                )
            end
        catch err
            err isa ErrorException && startswith(err.msg, "SundialsAdjoint") && rethrow()
        end
    end

    interp = sensealg.interp === :hermite ? Sundials.IDA_HERMITE : Sundials.IDA_POLYNOMIAL

    ctx_ref = Ref{Sundials.SUNContext}(C_NULL)
    Sundials.SUNContext_Create(
        C_NULL,
        Base.unsafe_convert(Ptr{Sundials.SUNContext}, ctx_ref)
    )
    ctx = ctx_ref[]
    mem = Sundials.Handle(Sundials.IDACreate(ctx))

    ufwd = copy(u0v)
    dufwd = copy(du0v)
    ufwd_nv = NVector(ufwd, ctx)
    dufwd_nv = NVector(dufwd, ctx)
    yret = similar(ufwd)
    ypret = similar(dufwd)
    yret_nv = NVector(yret, ctx)
    ypret_nv = NVector(ypret, ctx)
    yinterp = similar(ufwd)
    ypinterp = similar(dufwd)
    yinterp_nv = NVector(yinterp, ctx)
    ypinterp_nv = NVector(ypinterp, ctx)
    λ = zeros(n)
    λp = zeros(n)
    λ_nv = NVector(λ, ctx)
    λp_nv = NVector(λp, ctx)
    qB = zeros(np)
    qB_nv = has_p ? NVector(qB, ctx) : nothing
    id = Float64.(collect(diffvars))
    id_nv = NVector(id, ctx)
    gu = zeros(n)
    gp = dgdp_discrete === nothing ? nothing : zeros(np)
    dp = has_p ? zeros(np) : nothing
    du0_grad = zeros(n)
    tret = [t0]
    ncheck = Ref{Cint}(0)
    which = Ref{Cint}(0)

    # The forward/backward linear solvers and matrices are raw SUNDIALS pointers
    # that IDAS uses for the whole integration; they carry no finalizer, so they
    # must stay bound (pre-declared here, assigned inside the `try`) and be freed
    # explicitly in the `finally` after `IDAFree` (via `empty!(mem)`) and before
    # `SUNContext_Free`, otherwise they leak. `:GMRES` returns a `nothing` matrix.
    local LSf = nothing
    local Af = nothing
    local LSB = nothing
    local AB = nothing

    GC.@preserve data ufwd_nv dufwd_nv yret_nv ypret_nv yinterp_nv ypinterp_nv λ_nv λp_nv qB_nv id_nv begin
        try
            # Forward pass with checkpointing. The provided `(u0, du0)` are
            # assumed consistent (as for the forward `IDA` solve).
            _check_idas_flag(
                Sundials.IDAInit(mem, fwd_cfun, t0, ufwd_nv, dufwd_nv),
                "IDAInit"
            )
            _check_idas_flag(Sundials.IDASetUserData(mem, data), "IDASetUserData")
            _check_idas_flag(
                Sundials.IDASStolerances(mem, reltol, abstol),
                "IDASStolerances"
            )
            _check_idas_flag(
                Sundials.IDASetMaxNumSteps(mem, maxiters),
                "IDASetMaxNumSteps"
            )
            _check_idas_flag(Sundials.IDASetId(mem, id_nv), "IDASetId")
            LSf, Af = _idas_set_solvers(
                (LS, A) -> Sundials.IDASetLinearSolver(mem, LS, A),
                alg, ufwd_nv, n, ctx
            )
            _check_idas_flag(
                Sundials.IDAAdjInit(mem, sensealg.steps, interp),
                "IDAAdjInit"
            )
            _check_idas_flag(
                Sundials.IDASolveF(
                    mem, tf, tret, yret_nv, ypret_nv, Sundials.IDA_NORMAL,
                    ncheck
                ),
                "IDASolveF"
            )

            # Backward (adjoint) problem. λ(tf) collects the cost jumps at tf,
            # transferred through `(∂F/∂du)ᵀ Δλ = Δw` with the index-1
            # correction; `yret`/`ypret` hold the forward terminal state from
            # `IDASolveF`.
            cur_time = length(ts)
            if cur_time >= 1 && ts[cur_time] == tf
                y_f = sol(tf)
                while cur_time >= 1 && ts[cur_time] == tf
                    if !(no_start && cur_time == 1)
                        fill!(gu, false)
                        dgdu_discrete(gu, y_f, p, tf, cur_time)
                        Δλ, _ = _idas_cost_jump!(
                            dp, data, structure, Mtpinv, Mt, gu, yret, ypret, tf
                        )
                        λ .+= Δλ
                        if dgdp_discrete !== nothing
                            fill!(gp, false)
                            dgdp_discrete(gp, y_f, p, tf, cur_time)
                            dp .+= gp
                        end
                    end
                    cur_time -= 1
                end
            end

            _check_idas_flag(Sundials.IDACreateB(mem, which), "IDACreateB")
            fill!(λp, false)
            _check_idas_flag(
                Sundials.IDAInitB(mem, which[], adj_cfun, tf, λ_nv, λp_nv),
                "IDAInitB"
            )
            _check_idas_flag(
                Sundials.IDASetUserDataB(mem, which[], data),
                "IDASetUserDataB"
            )
            _check_idas_flag(
                Sundials.IDASStolerancesB(mem, which[], reltol, abstol),
                "IDASStolerancesB"
            )
            _check_idas_flag(
                Sundials.IDASetMaxNumStepsB(mem, which[], maxiters),
                "IDASetMaxNumStepsB"
            )
            _check_idas_flag(Sundials.IDASetIdB(mem, which[], id_nv), "IDASetIdB")
            LSB, AB = _idas_set_solvers(
                (LS, A) -> Sundials.IDASetLinearSolverB(mem, which[], LS, A),
                alg, λ_nv, n, ctx
            )
            if has_p
                _check_idas_flag(
                    Sundials.IDAQuadInitB(mem, which[], quad_cfun, qB_nv),
                    "IDAQuadInitB"
                )
                _check_idas_flag(
                    Sundials.IDAQuadSStolerancesB(mem, which[], reltol, abstol),
                    "IDAQuadSStolerancesB"
                )
                _check_idas_flag(
                    Sundials.IDASetQuadErrConB(
                        mem, which[],
                        sensealg.quad_error_control ? 1 : 0
                    ),
                    "IDASetQuadErrConB"
                )
            end

            # Consistent backward initial conditions: the algebraic components
            # of λ and the differential components of λ' are computed by IDAS
            # from the differential λ set above (IDACalcICB takes the forward
            # solution at tf as input).
            next_stop = cur_time >= 1 && ts[cur_time] > t0 && ts[cur_time] < tf ?
                ts[cur_time] : t0
            _check_idas_flag(
                Sundials.IDACalcICB(mem, which[], next_stop, yret_nv, ypret_nv),
                "IDACalcICB"
            )

            # Integrate backward, stopping at every discrete cost time to add
            # the jump and reinitialize the backward problem.
            tcur = tf
            while cur_time >= 1
                s = ts[cur_time]
                if s < tcur && s > t0
                    _check_idas_flag(
                        Sundials.IDASolveB(mem, s, Sundials.IDA_NORMAL),
                        "IDASolveB"
                    )
                    _check_idas_flag(
                        Sundials.IDAGetB(mem, which[], tret, λ_nv, λp_nv),
                        "IDAGetB"
                    )
                    if has_p
                        _check_idas_flag(
                            Sundials.IDAGetQuadB(mem, convert(Cint, which[]), tret, qB_nv),
                            "IDAGetQuadB"
                        )
                    end
                    tcur = s
                    y_s = sol(s)
                    # The forward `(y, y')` at the stop time, for the jump
                    # correction Jacobians and for `IDACalcICB` below.
                    _check_idas_flag(
                        Sundials.IDAGetAdjY(mem, s, yinterp_nv, ypinterp_nv),
                        "IDAGetAdjY"
                    )
                    while cur_time >= 1 && ts[cur_time] == s
                        if !(no_start && cur_time == 1)
                            fill!(gu, false)
                            dgdu_discrete(gu, y_s, p, s, cur_time)
                            Δλ, _ = _idas_cost_jump!(
                                dp, data, structure, Mtpinv, Mt, gu, yinterp,
                                ypinterp, s
                            )
                            λ .+= Δλ
                            if dgdp_discrete !== nothing
                                fill!(gp, false)
                                dgdp_discrete(gp, y_s, p, s, cur_time)
                                dp .+= gp
                            end
                        end
                        cur_time -= 1
                    end
                    _check_idas_flag(
                        Sundials.IDAReInitB(mem, which[], s, λ_nv, λp_nv),
                        "IDAReInitB"
                    )
                    if has_p
                        _check_idas_flag(
                            Sundials.IDAQuadReInitB(mem, convert(Cint, which[]), qB_nv),
                            "IDAQuadReInitB"
                        )
                    end
                    next_stop = cur_time >= 1 && ts[cur_time] > t0 && ts[cur_time] < s ?
                        ts[cur_time] : t0
                    _check_idas_flag(
                        Sundials.IDACalcICB(
                            mem, which[], next_stop, yinterp_nv,
                            ypinterp_nv
                        ),
                        "IDACalcICB"
                    )
                else
                    break
                end
            end

            # Final leg down to t0.
            if tcur > t0
                _check_idas_flag(
                    Sundials.IDASolveB(mem, t0, Sundials.IDA_NORMAL),
                    "IDASolveB"
                )
                _check_idas_flag(
                    Sundials.IDAGetB(mem, which[], tret, λ_nv, λp_nv),
                    "IDAGetB"
                )
                if has_p
                    _check_idas_flag(
                        Sundials.IDAGetQuadB(mem, convert(Cint, which[]), tret, qB_nv),
                        "IDAGetQuadB"
                    )
                end
            end

            # Gradient w.r.t. the initial state: the adjoint boundary term
            # `(∂F/∂du)ᵀ λ(t0)`, plus any cost jump exactly at t0 (which enters
            # `dG/du0` in w-space directly, with the index-1 correction).
            mul!(du0_grad, Mt, λ)
            while cur_time >= 1
                s = ts[cur_time]
                s == t0 ||
                    error("SundialsAdjoint: internal error, unprocessed cost time $(s).")
                if !(no_start && cur_time == 1)
                    y_s = sol(s)
                    fill!(gu, false)
                    dgdu_discrete(gu, y_s, p, s, cur_time)
                    _, Δw = _idas_cost_jump!(
                        dp, data, structure, Mtpinv, Mt, gu, u0v, du0v, s
                    )
                    du0_grad .+= Δw
                    if dgdp_discrete !== nothing
                        fill!(gp, false)
                        dgdp_discrete(gp, y_s, p, s, cur_time)
                        dp .+= gp
                    end
                end
                cur_time -= 1
            end

            has_p && (dp .+= qB)
        finally
            empty!(mem)
            LSf === nothing || Sundials.SUNLinSolFree(LSf)
            LSB === nothing || Sundials.SUNLinSolFree(LSB)
            Af === nothing || Sundials.SUNMatDestroy(Af)
            AB === nothing || Sundials.SUNMatDestroy(AB)
            Sundials.SUNContext_Free(ctx)
        end
    end

    return du0_grad, has_p ? dp' : nothing
end

end
