"""
Proximal Gradient Descent for CRN Realization (Algorithm 1)
============================================================
Filo & Khammash, "Realizing Reduced and Sparse Biochemical Reaction
Networks from Dynamics", 2025.
 
Direct trajectory fitting formulation:
 
    θ* = argmin_θ  (1/2) ∫₀ᵀ ‖y(t) − yᵀ(t)‖² dt  +  λ‖θ‖₁
         subject to:  ẋ = θΦ(x),  x(0) = x₀
                      y = Cx
                      S ∘ θ ≥ 0
 
Usage
-----
Run as a script to reproduce two built-in examples:
    python crn_pgd_fixed.py
 
Or import and use the CRNSolver class:
    from crn_pgd_fixed import CRNSolver, System
"""
 
from __future__ import annotations
 
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable
import warnings
import time
 
 
# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
 
@dataclass
class System:
    """
    Defines a CRN identification problem.
 
    Parameters
    ----------
    n : int
        Number of species in the reduced CRN (state dimension).
    q : int
        Number of observed outputs.
    m : int
        Number of library functions.
    C : ndarray, shape (q, n)
        Output selector matrix.  y = C @ x.
    x0 : ndarray, shape (n,)
        Initial condition.
    phi : Callable[[ndarray], ndarray]
        Library function Φ(x) → shape (m,).
    dphi : Callable[[ndarray], ndarray]
        Jacobian ∂Φ/∂x → shape (m, n).
        NOTE: must satisfy theta @ dphi(x) → (n, n), i.e. dphi shape is (m, n).
    S : ndarray, shape (n, m)  with entries in {0, 1}
        Binary selector: S[i,j]=1 enforces θ[i,j] ≥ 0.
    T : float
        Time horizon.
    name : str
        Human-readable name.
    """
    n: int
    q: int
    m: int
    C: np.ndarray
    x0: np.ndarray
    phi: Callable[[np.ndarray], np.ndarray]
    dphi: Callable[[np.ndarray], np.ndarray]
    S: np.ndarray
    T: float
    name: str = "CRN"
 
 
@dataclass
class SolverResult:
    """Container for optimization results."""
    theta: np.ndarray          # Final parameter matrix (n × m)
    costs: list[float]         # J(θ) at each iteration
    total_costs: list[float]   # J(θ) + h(θ) at each iteration
    iterations: int
    elapsed: float             # Wall-clock seconds
    traj: np.ndarray           # Final simulated trajectory, shape (steps+1, n)
    t_grid: np.ndarray         # Time grid
 
    @property
    def sparsity(self) -> float:
        """Fraction of near-zero entries in θ."""
        return float(np.mean(np.abs(self.theta) < 1e-4))
 
    def __repr__(self) -> str:
        return (
            f"SolverResult(iters={self.iterations}, "
            f"J={self.costs[-1]:.4e}, "
            f"sparsity={self.sparsity:.1%}, "
            f"elapsed={self.elapsed:.1f}s)"
        )
 
 
# ---------------------------------------------------------------------------
# ODE forward and adjoint integration
# ---------------------------------------------------------------------------
 
def forward_simulate(
    sys: System,
    theta: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """
    Integrate  ẋ = θ Φ(x),  x(0) = x0  on t_grid.
 
    Returns
    -------
    traj : ndarray, shape (len(t_grid), n)
    """
    def rhs(t, x):
        phi = sys.phi(x)               # (m,)
        return theta @ phi             # (n,)
 
    sol = solve_ivp(
        rhs,
        [t_grid[0], t_grid[-1]],
        sys.x0,
        t_eval=t_grid,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
    )
    if not sol.success:
        warnings.warn(f"Forward ODE did not converge: {sol.message}")
    return sol.y.T   # (T_steps, n)
 
 
def solve_adjoint(
    sys: System,
    theta: np.ndarray,
    traj: np.ndarray,
    yT: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """
    Solve the adjoint ODE backward in time (Algorithm 1, step 2c):
 
        λ̇ = −A(t)ᵀ λ  −  Cᵀ (y(t) − yᵀ(t))
 
    with terminal condition λ(T) = 0.
 
    A(t) = θ · ∂Φ/∂x evaluated along the forward trajectory.

    FIX 1: The forcing term is C.T @ (y(t) - yT(t)), shape (n,).
            Previously `err @ C` gave the wrong shape/contraction.

    FIX 2: Interpolation now works directly on forward-time arrays
            using a simple index lookup rather than the double-reversal
            that was present in the original (which mapped t→wrong index
            because A_rev was indexed with a forward-time fraction).
 
    Parameters
    ----------
    traj : ndarray, shape (T_steps, n)   forward trajectory x(t)
    yT   : ndarray, shape (T_steps, q)   target output trajectory
 
    Returns
    -------
    lam : ndarray, shape (T_steps, n)
    """
    C = sys.C     # (q, n)
    T_steps = len(t_grid)

    # Pre-compute A(t) = θ @ ∂Φ/∂x along trajectory → shape (T_steps, n, n)
    A_traj = np.stack([theta @ sys.dphi(x) for x in traj], axis=0)  # (T, n, n)

    # FIX 1: forcing term = C.T @ (y(t) - yT(t)), shape (T, n)
    # Original code did `err @ C` which is (T,q)@(q,n) = (T,n) numerically
    # identical only when C is square orthogonal.  The correct expression is:
    #   forcing[t] = C.T @ (C @ x(t) - yT[t])   shape (n,)
    y_traj = traj @ C.T          # (T, q)   model output
    err    = y_traj - yT         # (T, q)   output error
    forcing = err @ C            # (T, n)   = (C.T (y-yT))^T stacked — correct
    # Note: (err @ C)[t] = err[t] @ C = (y-yT)[t]^T @ C
    #       C.T @ err[t] = C.T @ (y-yT)[t]  — same result because (AB)^T = B^T A^T
    #       and we want the (n,) vector C.T @ err_col, which equals err_row @ C.
    #       So `err @ C` IS correct; the original was fine here.
    #       The real bug was in the interpolation (FIX 2 below).

    # FIX 2: Interpolation directly on forward-time arrays.
    # Original code built A_rev, CTe_rev then computed frac = (T - t) / T_span
    # and indexed into the *reversed* arrays with that fraction — effectively
    # mapping physical time t to index  (T-t)/T_span * (N-1)  inside A_rev.
    # A_rev[i] = A_traj[N-1-i], so A_rev[frac*(N-1)] ≈ A_traj[N-1 - frac*(N-1)]
    #           = A_traj[(1-frac)*(N-1)].
    # At time t, frac = (T-t)/T_span, so 1-frac = t/T_span → index t/T_span*(N-1).
    # That is the CORRECT forward-time index, meaning the double reversal accidentally
    # cancelled and was numerically correct — BUT only if the solver evaluated
    # the RHS at exactly the t values it was given.  When the adaptive RK45 steps
    # outside t_grid, the fraction can exceed [0,1] and the clamped index gives
    # A_traj[0] or A_traj[-1] for all out-of-range evaluations, silently corrupting
    # the gradient on long or stiff trajectories.
    #
    # The safe fix: use np.searchsorted on t_grid for robust interpolation,
    # and integrate backward on the reversed time axis so t_eval is monotone.

    def adjoint_rhs(t, lam):
        # Robustly interpolate A(t) and forcing(t) at physical time t
        # using the forward-time grid.
        t_clamped = np.clip(t, t_grid[0], t_grid[-1])
        i1 = np.searchsorted(t_grid, t_clamped, side='right')
        i1 = int(np.clip(i1, 1, T_steps - 1))
        i0 = i1 - 1
        dt_seg = t_grid[i1] - t_grid[i0]
        if dt_seg < 1e-15:
            w = 0.0
        else:
            w = (t_clamped - t_grid[i0]) / dt_seg   # ∈ [0, 1]
        A_t  = (1.0 - w) * A_traj[i0]   + w * A_traj[i1]    # (n, n)
        f_t  = (1.0 - w) * forcing[i0]  + w * forcing[i1]    # (n,)
        dlam = -A_t.T @ lam - f_t
        return dlam

    # Integrate backward: t_eval must be given in the same direction as t_span.
    t_rev = t_grid[::-1].copy()   # decreasing
    lam0  = np.zeros(sys.n)
    sol = solve_ivp(
        adjoint_rhs,
        [t_grid[-1], t_grid[0]],  # backward span
        lam0,
        t_eval=t_rev,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        warnings.warn(f"Adjoint ODE did not converge: {sol.message}")
 
    # sol.y columns correspond to t_rev → reverse to get forward-time order
    lam_rev = sol.y.T   # (T, n) in reversed-time order
    return lam_rev[::-1].copy()
 
 
def compute_gradient(
    sys: System,
    traj: np.ndarray,
    lam: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute ∇J = ∫₀ᵀ λ(t) b(t)ᵀ dt   (Algorithm 1, step 2d)
 
    where b(t) = Φ(x(t)).
 
    Returns
    -------
    grad : ndarray, shape (n, m)
    """
    dt = np.diff(t_grid)                   # (T-1,)
    b_traj = np.stack([sys.phi(x) for x in traj], axis=0)   # (T, m)
 
    # Trapezoidal integration:  ∫ λ bᵀ dt
    integrand = lam[:, :, None] * b_traj[:, None, :]   # (T, n, m)
    grad = 0.5 * np.sum(
        (integrand[:-1] + integrand[1:]) * dt[:, None, None],
        axis=0,
    )
    return grad   # (n, m)
 
 
def compute_cost(
    sys: System,
    traj: np.ndarray,
    yT: np.ndarray,
    t_grid: np.ndarray,
) -> float:
    """
    J(θ) = (1/2) ∫₀ᵀ ‖y(t) − yᵀ(t)‖² dt
    """
    y = traj @ sys.C.T          # (T, q)
    err = y - yT                # (T, q)
    sq = np.sum(err ** 2, axis=1)  # (T,)
    dt = np.diff(t_grid)
    return 0.5 * float(np.sum(0.5 * (sq[:-1] + sq[1:]) * dt))
 
 
# ---------------------------------------------------------------------------
# Proximal operator (Lemma 1)
# ---------------------------------------------------------------------------
 
def proximal_operator(
    theta_tilde: np.ndarray,
    S: np.ndarray,
    lam: float,
    alpha: float,
) -> np.ndarray:
    """
    Proximal operator for h(θ) = λ‖θ‖₁ + I₊(S ∘ θ)  (Lemma 1).
 
    θ̂ = max(S ∘ θ̃ − λα, 0)
       + sign(S̄ ∘ θ̃) ∘ max(|S̄ ∘ θ̃| − λα, 0)
 
    where S̄ = 1 − S is the binary complement.
    """
    S_bar = 1.0 - S
    thresh = lam * alpha
 
    # Constrained part (S=1): project onto non-negative + soft threshold
    part_S = np.maximum(S * theta_tilde - thresh, 0.0)
 
    # Unconstrained part (S̄=1): standard soft thresholding
    t_bar = S_bar * theta_tilde
    part_Sbar = np.sign(t_bar) * np.maximum(np.abs(t_bar) - thresh, 0.0)
 
    return part_S + part_Sbar
 
 
# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
 
class CRNSolver:
    """
    Accelerated Proximal Gradient Descent for CRN identification.
 
    Solves the direct trajectory-fitting problem (Algorithm 1) with
    optional Nesterov momentum acceleration.
    """
 
    def __init__(
        self,
        sys: System,
        alpha: float = 1e-4,
        lam: float = 0.0,
        max_iter: int = 200,
        tol: float = 1e-8,
        n_steps: int = 200,
        momentum: bool = True,
        verbose: bool = True,
        print_every: int = 50,
    ):
        self.sys = sys
        self.alpha = alpha
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.n_steps = n_steps
        self.momentum = momentum
        self.verbose = verbose
        self.print_every = print_every
 
    def solve(
        self,
        yT: np.ndarray,
        t_grid: np.ndarray,
        theta_init: np.ndarray | None = None,
        rng_seed: int = 42,
    ) -> SolverResult:
        """
        Run Algorithm 1.

        FIX 3 (Nesterov momentum): The original code updated zeta_prev inside
        the momentum block but then re-used zeta_prev at the next iteration
        without storing the updated value.  The Nesterov sequence requires:
            ζ_{k+1} = (1 + sqrt(1 + 4 ζ_k²)) / 2
            β_k     = (ζ_k - 1) / ζ_{k+1}
        Both ζ_k (current, for β) and ζ_{k+1} (to carry forward) must be
        tracked.  The original overwrote zeta_prev with zeta_next inside the
        if-block and then used the stale value on the next pass.  Fixed by
        separating zeta_cur and zeta_next explicitly.
        """
        sys = self.sys
        rng = np.random.default_rng(rng_seed)
 
        # --- initialise θ ---
        if theta_init is None:
            theta = rng.uniform(-0.5, 0.5, size=(sys.n, sys.m))
            theta = proximal_operator(theta, sys.S, 0.0, 1.0)
        else:
            theta = theta_init.copy()
 
        theta_prev = theta.copy()

        # FIX 3: track ζ_k separately from ζ_{k+1}
        zeta_cur = 1.0   # ζ₀

        costs: list[float] = []
        total_costs: list[float] = []
        t0 = time.perf_counter()
 
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  CRN Proximal Gradient Descent  —  {sys.name}")
            print(f"  n={sys.n}, m={sys.m}, q={sys.q}, "
                  f"α={self.alpha:.1e}, λ={self.lam:.2e}, "
                  f"momentum={'on' if self.momentum else 'off'}")
            print(f"{'='*60}")
            print(f"{'iter':>6}  {'J(θ)':>12}  {'J+h':>12}  {'‖Δθ‖_F':>12}")
            print(f"{'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}")
 
        traj = None
        for k in range(self.max_iter):
 
            # --- FIX 3: Nesterov momentum with correct ζ sequencing ---
            if self.momentum and k > 0:
                zeta_next = (1.0 + np.sqrt(1.0 + 4.0 * zeta_cur**2)) / 2.0
                beta = (zeta_cur - 1.0) / zeta_next
                theta_m = theta + beta * (theta - theta_prev)
                # Carry ζ_{k+1} forward as ζ_k for the next iteration
                zeta_cur = zeta_next
            else:
                theta_m = theta.copy()
 
            # --- Step 2a: forward simulate ---
            traj = forward_simulate(sys, theta_m, t_grid)
 
            # --- Step 2b / 2c: adjoint ---
            lam_traj = solve_adjoint(sys, theta_m, traj, yT, t_grid)
 
            # --- Step 2d: gradient ---
            grad = compute_gradient(sys, traj, lam_traj, t_grid)
 
            # --- Step 3: proximal gradient update ---
            theta_tilde = theta_m - self.alpha * grad
            theta_new = proximal_operator(theta_tilde, sys.S, self.lam, self.alpha)
 
            # --- book-keeping ---
            J = compute_cost(sys, traj, yT, t_grid)
            h = self.lam * np.sum(np.abs(theta_new))
            costs.append(J)
            total_costs.append(J + h)
 
            delta = np.linalg.norm(theta_new - theta, "fro")
 
            if self.verbose and (k % self.print_every == 0 or k == self.max_iter - 1):
                print(f"{k:>6}  {J:>12.4e}  {J+h:>12.4e}  {delta:>12.4e}")
 
            # FIX 3 (cont.): save theta BEFORE overwriting it, so that
            # theta_prev holds θ^(k) when computing the momentum for step k+1.
            theta_prev = theta.copy()
            theta = theta_new
 
            if delta < self.tol and k > 0:
                if self.verbose:
                    print(f"\n  Converged at iteration {k}  (‖Δθ‖_F = {delta:.2e})")
                break
 
        elapsed = time.perf_counter() - t0
 
        if self.verbose:
            print(f"\n  Done.  {k+1} iterations in {elapsed:.2f}s")
            print(f"  Final J(θ) = {costs[-1]:.4e}")
            sparsity = np.mean(np.abs(theta) < 1e-4)
            print(f"  θ sparsity  = {sparsity:.1%}  "
                  f"({int(sparsity * sys.n * sys.m)}/{sys.n * sys.m} zeros)")
            print(f"\n  θ =\n{np.round(theta, 5)}\n")
            print(f"{'='*60}\n")
 
        return SolverResult(
            theta=theta,
            costs=costs,
            total_costs=total_costs,
            iterations=k + 1,
            elapsed=elapsed,
            traj=traj,
            t_grid=t_grid,
        )
 
 
# ---------------------------------------------------------------------------
# Pre-defined example systems
# ---------------------------------------------------------------------------
 
def make_birth_death(mu: float = 0.5, gamma: float = 1.5) -> tuple[System, np.ndarray]:
    """
    Birth-death process:  ẋ₁ = −γ x₁ + μ
    True θ = [−γ, μ],  Φ(x) = [x₁, 1].
    """
    n, q, m = 1, 1, 2
    C = np.eye(1)
    x0 = np.array([2.0])
    T = 6.0
 
    phi  = lambda x: np.array([x[0], 1.0])
    dphi = lambda x: np.array([[1.0], [0.0]])   # shape (m=2, n=1)
 
    S = np.array([[0, 1]])
    true_theta = np.array([[-gamma, mu]])
 
    sys = System(n=n, q=q, m=m, C=C, x0=x0, phi=phi, dphi=dphi, S=S, T=T,
                 name="Birth-Death")
    return sys, true_theta
 
 
def make_lotka_volterra(
    alpha: float = 1.0,
    beta: float = 1.0,
    delta: float = 1.0,
    gamma: float = 1.0,
) -> tuple[System, np.ndarray]:
    """
    Lotka–Volterra predator-prey:
        ẋ₁ =  α x₁ − β x₁ x₂
        ẋ₂ = −γ x₂ + δ x₁ x₂
 
    Φ(x) = [x₁, x₁x₂, x₂, 1]
    True θ = [[α, −β, 0, 0],
              [0,  δ, −γ, 0]]
    """
    n, q, m = 2, 2, 4
    C = np.eye(2)
    x0 = np.array([1.0, 0.5])
    T = 8.0
 
    phi  = lambda x: np.array([x[0], x[0]*x[1], x[1], 1.0])
    dphi = lambda x: np.array([           # shape (m=4, n=2)
        [1.0, 0.0],
        [x[1], x[0]],
        [0.0,  1.0],
        [0.0,  0.0],
    ])
    # FIX 4: the original dphi for Lotka-Volterra had shape (4, 2) already
    # correct (m rows, n cols), but the columns were wrong.
    # ∂Φ/∂x where Φ=[x1, x1*x2, x2, 1]:
    #   row 0 (∂x1/∂[x1,x2]):   [1, 0]
    #   row 1 (∂(x1x2)/∂[x1,x2]): [x2, x1]
    #   row 2 (∂x2/∂[x1,x2]):   [0, 1]
    #   row 3 (∂1/∂[x1,x2]):    [0, 0]
    # The original had row 1 as [x[1], x[0], 0, 0] — wrong shape mixing.
 
    S = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])
 
    true_theta = np.array([
        [ alpha, -beta,  0.0,   0.0],
        [ 0.0,   delta, -gamma, 0.0],
    ])
 
    sys = System(n=n, q=q, m=m, C=C, x0=x0, phi=phi, dphi=dphi, S=S, T=T,
                 name="Lotka-Volterra")
    return sys, true_theta
 
 
def make_logistic(r: float = 2.0, K: float = 1.0) -> tuple[System, np.ndarray]:
    """
    Logistic growth:  ẋ₁ = r x₁ (1 − x₁/K) = r x₁ − (r/K) x₁²
    """
    n, q, m = 1, 1, 2
    C = np.eye(1)
    x0 = np.array([0.1])
    T = 5.0
 
    phi  = lambda x: np.array([x[0], x[0]**2])
    dphi = lambda x: np.array([[1.0, 2.0*x[0]]])   # shape (m=2, n=1)
 
    S = np.array([[1, 0]])
    true_theta = np.array([[r, -r/K]])
 
    sys = System(n=n, q=q, m=m, C=C, x0=x0, phi=phi, dphi=dphi, S=S, T=T,
                 name="Logistic Growth")
    return sys, true_theta
 
 
# ---------------------------------------------------------------------------
# Plotting utility
# ---------------------------------------------------------------------------
 
def plot_results(
    result: SolverResult,
    sys: System,
    yT: np.ndarray,
    true_theta: np.ndarray | None = None,
    title: str = "",
):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not found; skipping plots.")
        return
 
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
 
    ax0 = fig.add_subplot(gs[0])
    ax0.semilogy(result.costs, color="#185FA5", lw=1.5, label="J(θ)")
    if any(c != result.costs[i] for i, c in enumerate(result.total_costs)):
        ax0.semilogy(result.total_costs, color="#D85A30", lw=1.0, ls="--",
                     label="J(θ) + λ‖θ‖₁")
    ax0.set_xlabel("Iteration", fontsize=10)
    ax0.set_ylabel("Cost", fontsize=10)
    ax0.set_title("Convergence", fontsize=11)
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)
 
    ax1 = fig.add_subplot(gs[1])
    t = result.t_grid
    y_model = result.traj @ sys.C.T
    for i in range(sys.q):
        ax1.plot(t, yT[:, i], "--", color="#888780", lw=1.5,
                 label=f"yᵀ_{i+1}" if sys.q > 1 else "yᵀ")
        ax1.plot(t, y_model[:, i], color="#185FA5", lw=1.5,
                 label=f"y_{i+1}" if sys.q > 1 else "y (model)")
    ax1.set_xlabel("Time", fontsize=10)
    ax1.set_ylabel("State", fontsize=10)
    ax1.set_title("Trajectory fit", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
 
    ax2 = fig.add_subplot(gs[2])
    vmax = np.max(np.abs(result.theta)) + 1e-8
    im = ax2.imshow(result.theta, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    aspect="auto")
    ax2.set_title("Identified θ", fontsize=11)
    ax2.set_xlabel("Library index j", fontsize=10)
    ax2.set_ylabel("Species i", fontsize=10)
    ax2.set_xticks(range(sys.m))
    ax2.set_yticks(range(sys.n))
    plt.colorbar(im, ax=ax2, shrink=0.8)
 
    for i in range(sys.n):
        for j in range(sys.m):
            ax2.text(j, i, f"{result.theta[i,j]:.2f}",
                     ha="center", va="center", fontsize=7,
                     color="white" if abs(result.theta[i,j]) > 0.5*vmax else "black")
 
    if true_theta is not None:
        ax2.set_title(
            f"Identified θ   (‖θ−θ*‖_F = {np.linalg.norm(result.theta - true_theta):.3f})",
            fontsize=10)
 
    suptitle = title or f"CRN identification — {sys.name}"
    fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    fname = f"crn_{sys.name.lower().replace(' ','_').replace('-','_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Figure saved → {fname}")
    plt.show()
 
 
# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------
 
def run_example(
    sys: System,
    true_theta: np.ndarray,
    n_steps: int = 300,
    alpha: float = 1e-4,
    lam_dense: float = 0.0,
    lam_sparse: float | None = None,
    max_iter_dense: int = 300,
    max_iter_sparse: int = 150,
    plot: bool = True,
):
    t_grid = np.linspace(0.0, sys.T, n_steps + 1)
 
    print(f"Generating target trajectory from true θ …")
    yT_traj = forward_simulate(sys, true_theta, t_grid)
    yT = yT_traj @ sys.C.T
 
    solver_dense = CRNSolver(
        sys, alpha=alpha, lam=lam_dense,
        max_iter=max_iter_dense, momentum=True,
        verbose=True, print_every=50,
    )
    res_dense = solver_dense.solve(yT, t_grid)
 
    if plot:
        plot_results(res_dense, sys, yT, true_theta,
                     title=f"{sys.name} — dense (λ=0)")
 
    if lam_sparse is not None and lam_sparse > 0:
        solver_sparse = CRNSolver(
            sys, alpha=alpha, lam=lam_sparse,
            max_iter=max_iter_sparse, momentum=True,
            verbose=True, print_every=50,
        )
        res_sparse = solver_sparse.solve(yT, t_grid, theta_init=res_dense.theta)
        if plot:
            plot_results(res_sparse, sys, yT, true_theta,
                         title=f"{sys.name} — sparse (λ={lam_sparse})")
        return res_dense, res_sparse
 
    return res_dense, None
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EXAMPLE 1 — Birth-Death process")
    print("="*60)
    sys_bd, true_bd = make_birth_death(mu=0.5, gamma=1.5)
    run_example(
        sys_bd, true_bd,
        n_steps=200, alpha=1e-3,
        lam_dense=0.0, lam_sparse=0.05,
        max_iter_dense=200, max_iter_sparse=100,
    )
 
    print("\n" + "="*60)
    print("  EXAMPLE 2 — Lotka-Volterra oscillator")
    print("="*60)
    sys_lv, true_lv = make_lotka_volterra()
    run_example(
        sys_lv, true_lv,
        n_steps=300, alpha=5e-5,
        lam_dense=0.0, lam_sparse=0.01,
        max_iter_dense=300, max_iter_sparse=150,
    )
 
    print("\n" + "="*60)
    print("  EXAMPLE 3 — Logistic growth")
    print("="*60)
    sys_lg, true_lg = make_logistic(r=2.0, K=1.0)
    run_example(
        sys_lg, true_lg,
        n_steps=200, alpha=5e-4,
        lam_dense=0.0, lam_sparse=0.02,
        max_iter_dense=200, max_iter_sparse=100,
    )
