"""
ImplicitSINDy — Implicit Sparse Identification of Nonlinear Dynamics
---------------------------------------------------------------------
Identifies implicit ODEs of the form:  F(x, ẋ) = 0

Library:  Θ(x, ẋ) — all monomials  x1^p1 · … · xn^pn · ẋ1^q1 · … · ẋn^qn
          where sum(pᵢ + qᵢ) ≤ max_degree  (total-degree library)
          →  C(2n + d, d) columns  (avoids the 6^(2n) multicollinearity explosion)

Key design choices vs. naïve implementation
--------------------------------------------
1. Total-degree library (not per-variable): avoids exponential bloat and the
   severe multicollinearity that breaks STLSQ when per-variable degree is used.

2. Norm-only column scaling (no mean subtraction): StandardScaler's mean
   removal destroys the zero-intercept assumption implicit in SINDy and
   corrupts coefficient recovery. We divide each column by its L2 norm only.

3. Model selection prefers compound LHS: candidates whose LHS is a raw state
   variable xᵢ correspond to trivial explicit forms. A penalty is added to
   their AIC/BIC so the algorithm prefers true implicit (mixed) terms as LHS.

4. Lambda sweep via AIC: sweeps a log-spaced grid of sparsity thresholds and
   selects the model minimising AIC = m·log(RSS/m) + 2·nnz.

Usage
-----
    sindy = ImplicitSINDy(max_degree=2)
    sindy.fit(X, dt=0.05)
    sindy.print_equations()
"""

import numpy as np
import itertools
from math import comb
from joblib import Parallel, delayed


# ──────────────────────────────────────────────────────────────
# Numerical differentiation
# ──────────────────────────────────────────────────────────────

def finite_diff(X: np.ndarray, dt: float, order: int = 4) -> np.ndarray:
    """
    4th-order central finite differences (interior), 2nd-order at edges.
    X : (m, n)  — rows = time samples, cols = state variables
    """
    m, n = X.shape
    Xdot = np.zeros_like(X)
    if order == 4 and m >= 5:
        Xdot[2:-2] = (-X[4:] + 8*X[3:-1] - 8*X[1:-3] + X[:-4]) / (12 * dt)
        # Index 1 (use points 0, 1, 2, 3, 4)
        Xdot[1] = (-3*X[0] - 10*X[1] + 18*X[2] - 6*X[3] + X[4]) / (12 * dt)
        # Index -2 (use points -5, -4, -3, -2, -1)
        Xdot[-2] = (3*X[-1] + 10*X[-2] - 18*X[-3] + 6*X[-4] - X[-5]) / (12 * dt)
        
        # Absolute edges (remain 2nd order one-sided)
        Xdot[0]  = (-3*X[0]  + 4*X[1]  - X[2])  / (2 * dt)
        Xdot[-1] = ( 3*X[-1] - 4*X[-2] + X[-3]) / (2 * dt)
    else:
        Xdot[1:-1] = (X[2:] - X[:-2]) / (2 * dt)
        Xdot[0]    = (-3*X[0]  + 4*X[1]  - X[2])  / (2 * dt)
        Xdot[-1]   = ( 3*X[-1] - 4*X[-2] + X[-3]) / (2 * dt)
    return Xdot

_finite_diff = finite_diff  # back-compat alias


# ──────────────────────────────────────────────────────────────
# Lambda grid factory
# ──────────────────────────────────────────────────────────────

def lambda_grid(lo: float = 1e-3, hi: float = 1.0, n: int = 20) -> list:
    """Log-spaced grid of STLSQ threshold values."""
    return list(np.logspace(np.log10(lo), np.log10(hi), n))


# ──────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────

class ImplicitSINDy:
    """
    Parameters
    ----------
    max_degree      : maximum *total* degree of each monomial (default 2)
                      Library size = C(2n + d, d) where n = number of states
    threshold       : STLSQ threshold lambda — float or list for sweep
    max_iter        : STLSQ inner iterations
    n_val_frac      : fraction of data held out for validation
    implicit_bonus  : AIC penalty added to candidates whose LHS is a raw
                      state variable xi (encourages implicit/mixed LHS)
    """

    def __init__(
        self,
        max_degree: int = 2,
        threshold: float | list = 0.05,
        max_iter: int = 30,
        n_val_frac: float = 0.2,
        implicit_bonus: float = 500.0,
    ):
        self.max_degree     = max_degree
        self.threshold      = threshold
        self.max_iter       = max_iter
        self.n_val_frac     = n_val_frac
        self.implicit_bonus = implicit_bonus

        # populated after fit()
        self.feature_names_ = None
        self.exponents_     = None
        self._col_norms     = None
        self._n_states      = None
        self.results_       = None
        self.best_          = None
        self.Xi_            = None

    # ── Library ─────────────────────────────────────────────

    def _build_library(self, X: np.ndarray, Xdot: np.ndarray) -> np.ndarray:
        """
        Build Theta(X, Xdot) using all monomials with total degree <= max_degree.
        Returns raw (unscaled) Theta of shape (m, n_funcs).
        """
        m, n = X.shape
        self._n_states = n
        combined = np.hstack([X, Xdot])   # (m, 2n)
        n_vars   = 2 * n
        d        = self.max_degree

        self.exponents_ = [
            combo
            for combo in itertools.product(range(d + 1), repeat=n_vars)
            if sum(combo) <= d
        ]

        n_funcs = len(self.exponents_)
        Theta   = np.ones((m, n_funcs), dtype=np.float64)
        for i, combo in enumerate(self.exponents_):
            for v, p in enumerate(combo):
                if p > 0:
                    Theta[:, i] *= combined[:, v] ** p

        x_names  = [f"x{k+1}"  for k in range(n)]
        xd_names = [f"xd{k+1}" for k in range(n)]
        all_names = x_names + xd_names
        self.feature_names_ = []
        for combo in self.exponents_:
            parts = [
                (f"{nm}^{p}" if p > 1 else nm)
                for nm, p in zip(all_names, combo) if p > 0
            ]
            self.feature_names_.append("*".join(parts) if parts else "1")

        expected = comb(2*n + d, d)
        assert n_funcs == expected, f"Library size mismatch: {n_funcs} vs {expected}"
        return Theta

    def _scale(self, Theta: np.ndarray) -> np.ndarray:
        """
        Norm-only column scaling: divide each column by its L2 norm.
        No mean subtraction — that would corrupt the algebraic structure.
        """
        norms = np.linalg.norm(Theta, axis=0, keepdims=True)
        norms[norms < 1e-10] = 1.0
        self._col_norms = norms.ravel()
        return Theta / norms

    # ── STLSQ ───────────────────────────────────────────────

    def _stlsq(self, Phi: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        """Sequentially Thresholded Least Squares."""
        coef, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
        for _ in range(self.max_iter):
            small = np.abs(coef) < lam
            coef[small] = 0.0
            big = ~small
            if not np.any(big):
                break
            coef[big], _, _, _ = np.linalg.lstsq(Phi[:, big], y, rcond=None)
        return coef

    # ── Candidate solver (parallel worker) ──────────────────

    def _solve_candidate(
        self,
        Theta_tr: np.ndarray,
        Theta_val: np.ndarray,
        j: int,
        lam: float,
    ) -> dict:
        """
        Move column j to LHS; solve theta_j = Theta' xi_j via STLSQ.
        Diagonal is zero by construction (prevents trivial solution Xi = I).
        """
        y_tr   = Theta_tr[:, j]
        Phi_tr = np.delete(Theta_tr, j, axis=1)

        xi_r = self._stlsq(Phi_tr, y_tr, lam)
        xi   = np.insert(xi_r, j, 0.0)

        y_val   = Theta_val[:, j]
        Phi_val = np.delete(Theta_val, j, axis=1)
        res     = y_val - Phi_val @ xi_r
        norm_y  = np.linalg.norm(y_val) + 1e-12
        val_err = np.linalg.norm(res) / norm_y
        nnz     = int(np.sum(xi != 0))

        m_val = len(y_val)
        rss   = float(np.sum(res**2)) + 1e-30
        aic   = m_val * np.log(rss / m_val) + 2.0 * nnz

        # Penalise raw state variable LHS (xi^1 only) — those are trivial
        # explicit rearrangements, not genuine implicit equations.
        # A "raw state" monomial has total degree 1 AND its single power is on
        # an x variable (index < n_states), not an xdot variable.
        exp_j = self.exponents_[j]
        n = self._n_states
        # Check if the term contains any derivative (xdot) variables
        has_xdot = any(exp_j[i] > 0 for i in range(n, 2*n))
        # Check if it's a "pure" term (degree > 0, but no derivative)
        is_pure_state = (not has_xdot) and (sum(exp_j) > 0)
        # This determines if the term is a simple x1, x2, etc. (for the flag)
        is_raw_state = sum(exp_j) == 1 and not has_xdot

        if is_pure_state:
            aic += self.implicit_bonus

        return {
            "xi":           xi,
            "val_error":    val_err,
            "aic":          aic,
            "nnz":          nnz,
            "j":            j,
            "lam":          lam,
            "is_raw_state": is_raw_state,
        }

    # ── Batch simultaneous formulation ──────────────────────

    def _batch_solve(
        self,
        Theta_tr: np.ndarray,
        Theta_val: np.ndarray,
        lam: float,
    ) -> dict:
        """
        Solve  min ||Theta - Theta Xi||_F  s.t. diag(Xi) = 0
        via column-by-column STLSQ with diagonal zeroed.
        """
        n_f = Theta_tr.shape[1]
        Xi  = np.zeros((n_f, n_f))
        for j in range(n_f):
            y_tr   = Theta_tr[:, j]
            Phi_tr = np.delete(Theta_tr, j, axis=1)
            xi_r   = self._stlsq(Phi_tr, y_tr, lam)
            xi     = np.insert(xi_r, j, 0.0)
            Xi[:, j] = xi

        res_F   = np.linalg.norm(Theta_val - Theta_val @ Xi, "fro")
        norm_F  = np.linalg.norm(Theta_val, "fro") + 1e-12
        val_err = res_F / norm_F
        nnz     = int(np.sum(Xi != 0))
        m_val   = Theta_val.shape[0]
        aic     = m_val * np.log(res_F**2 / m_val + 1e-30) + 2 * nnz
        return {"Xi": Xi, "val_error": val_err, "aic": aic, "nnz": nnz, "lam": lam}

    # ── Main fit ─────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        X_dot: np.ndarray | None = None,
        dt: float | None = None,
        n_jobs: int = -1,
        lambdas: list | None = None,
        batch: bool = False,
    ) -> "ImplicitSINDy":
        """
        Parameters
        ----------
        X      : (m, n) state matrix (rows = time samples)
        X_dot  : (m, n) derivative matrix; computed from X via FD if None
        dt     : time step, required when X_dot is None
        n_jobs : parallel workers (-1 = all cores)
        lambdas: STLSQ thresholds to sweep; defaults to 20-point log grid
        batch  : also run the simultaneous batch formulation
        """
        if X_dot is None:
            if dt is None:
                raise ValueError("Provide X_dot or dt.")
            X_dot = finite_diff(X, dt)

        m, n = X.shape
        d    = self.max_degree
        print(f"[SINDy] Data:    {m} samples x {n} states")

        Theta_raw = self._build_library(X, X_dot)
        Theta     = self._scale(Theta_raw)
        n_funcs   = Theta.shape[1]
        print(f"[SINDy] Library: {n_funcs} monomials  "
              f"(total-degree <= {d},  C(2*{n}+{d},{d}) = {n_funcs})")

        n_val = max(1, int(m * self.n_val_frac))
        # Use a fixed generator to ensure the validation split is always the same
        rng = np.random.default_rng(42) 
        idx = rng.permutation(m)
        tr, val = idx[n_val:], idx[:n_val]
        Theta_tr, Theta_val = Theta[tr], Theta[val]

        if lambdas is None:
            if isinstance(self.threshold, (list, tuple, np.ndarray)):
                lambdas = list(self.threshold)
            else:
                lambdas = lambda_grid(lo=self.threshold * 0.1,
                                      hi=self.threshold * 10,
                                      n=20)

        print(f"[SINDy] Sweeping {len(lambdas)} threshold(s): "
              f"[{min(lambdas):.3g} ... {max(lambdas):.3g}]")

        all_results = Parallel(n_jobs=n_jobs)(
            delayed(self._solve_candidate)(Theta_tr, Theta_val, j, lam)
            for lam in lambdas
            for j in range(n_funcs)
        )
        self.results_ = all_results

        valid = [r for r in all_results if r["nnz"] >= 2]
        if not valid:
            valid = all_results
        best = min(valid, key=lambda r: r["aic"])
        self.best_ = best
        self.Xi_   = best["xi"]

        flag = "  [raw state = explicit form]" if best["is_raw_state"] else ""
        print(
            f"[SINDy] Best LHS : {self.feature_names_[best['j']]}{flag}\n"
            f"         lambda={best['lam']:.4g}  val_err={best['val_error']:.4e}"
            f"  AIC={best['aic']:.4g}  nnz={best['nnz']}"
        )

        if batch:
            batch_res  = [self._batch_solve(Theta_tr, Theta_val, lam) for lam in lambdas]
            self.batch_results_ = batch_res
            best_batch = min(batch_res, key=lambda r: r["aic"])
            self.best_batch_ = best_batch
            print(f"[SINDy] Batch:   lambda={best_batch['lam']:.4g}"
                  f"  val_err={best_batch['val_error']:.4e}  nnz={best_batch['nnz']}")

        return self

    # ── Reporting ────────────────────────────────────────────

    def _unscale_coefficients(self, xi_scaled: np.ndarray, j: int) -> np.ndarray:
        """
        Convert scaled-space coefficients back to physical units.
        In scaled space:  theta_j/norm_j = sum_k xi_k * theta_k/norm_k
        In physical space: theta_j = sum_k (xi_k * norm_j/norm_k) * theta_k
        """
        norms = self._col_norms
        return xi_scaled * norms[j] / norms

    def print_equations(self) -> None:
        """Print the identified implicit equation F(x, xdot) = 0."""
        if self.best_ is None:
            raise RuntimeError("Call fit() first.")
        j        = self.best_["j"]
        xi_phys  = self._unscale_coefficients(self.Xi_, j)
        lhs      = self.feature_names_[j]
        terms = [
            f"({c:+.4g})*{self.feature_names_[k]}"
            for k, c in enumerate(xi_phys)
            if k != j and c != 0.0
        ]
        rhs = "  +  ".join(terms) if terms else "0"
        print("\n-- Identified Implicit Equation ---------------------------")
        print(f"  {lhs}  =  {rhs}")
        print("-----------------------------------------------------------\n")

    def get_active_terms(self) -> list[tuple[str, float]]:
        """Returns [(feature_name, physical_coefficient)] for all non-zero terms."""
        if self.Xi_ is None:
            raise RuntimeError("Call fit() first.")
        j       = self.best_["j"]
        xi_phys = self._unscale_coefficients(self.Xi_, j)
        return [
            (self.feature_names_[k], float(c))
            for k, c in enumerate(xi_phys)
            if c != 0.0
        ]

    def top_candidates(self, n: int = 5) -> list[dict]:
        """Top-n candidate models ranked by AIC."""
        if self.results_ is None:
            raise RuntimeError("Call fit() first.")
        valid  = [r for r in self.results_ if r["nnz"] >= 2]
        ranked = sorted(valid, key=lambda r: r["aic"])[:n]
        for rank, r in enumerate(ranked, 1):
            flag = "*" if r["is_raw_state"] else " "
            print(
                f"  #{rank}{flag} LHS={self.feature_names_[r['j']]:14s}"
                f"  lambda={r['lam']:.4g}  val_err={r['val_error']:.4e}"
                f"  AIC={r['aic']:.4g}  nnz={r['nnz']}"
            )
        print("  (* = raw state variable LHS -- trivial explicit rearrangement)")
        return ranked


# ──────────────────────────────────────────────────────────────
# Example — Michaelis-Menten kinetics
# ──────────────────────────────────────────────────────────────
# True dynamics:   xdot = 0.6 - 1.5x / (0.3 + x)
# Implicit form:   multiply through by (0.3 + x):
#                  x*xdot = -0.3*xdot + 0.18 - 0.9*x
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scipy.integrate import solve_ivp

    np.random.seed(42)

    def mm(t, y):
        x = y[0]
        return [0.6 - 1.5 * x / (0.3 + x)]

    X_all, Xdot_all = [], []
    dt = 0.05
    for x0 in np.linspace(0.5, 3.0, 8):
        sol = solve_ivp(mm, [0, 15], [x0], max_step=dt, dense_output=True)
        t   = np.arange(0, 15, dt)
        xarr = sol.sol(t).T
        X_all.append(xarr)
        Xdot_all.append(finite_diff(xarr, dt))

    X    = np.vstack(X_all)
    Xdot = np.vstack(Xdot_all)
    print(f"Data: {X.shape[0]} samples from {len(X_all)} trajectories\n")

    sindy = ImplicitSINDy(max_degree=2, implicit_bonus=500.0)
    sindy.fit(X, X_dot=Xdot, n_jobs=-1,
              lambdas=lambda_grid(lo=0.001, hi=0.3, n=20))
    sindy.print_equations()

    print("Top-5 candidates:")
    sindy.top_candidates(5)

    print("\nExpected:  x1*xd1  =  (+0.18)*1  +  (-0.30)*xd1  +  (-0.90)*x1")
