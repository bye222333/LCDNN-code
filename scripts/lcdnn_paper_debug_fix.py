AUTO_INSTALL = True

if AUTO_INSTALL:
    import sys, subprocess, importlib
    from typing import Optional

    def ensure(pkg: str, import_name: Optional[str] = None):
        name = import_name or pkg
        try:
            importlib.import_module(name)
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

    ensure("numpy")
    ensure("scipy")
    ensure("pandas")
    ensure("scikit-learn", "sklearn")
    ensure("tqdm")
    ensure("lingam")

import numpy as np, pandas as pd, scipy
import sklearn
import lingam
print("numpy", np.__version__)
print("scipy", scipy.__version__)
print("pandas", pd.__version__)
print("sklearn", sklearn.__version__)
print("lingam", lingam.__version__)

import os
import math
import time
import json
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Set, Iterable

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from scipy import linalg
from scipy.stats import gamma

from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")

# =========================
# 1) Reproducibility
# =========================
def set_all_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True, ddof=1)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


# ============================================================
# 2) HSIC independence test (Gamma approximation; fast, no perms)
# ============================================================
def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    G = np.sum(X * X, axis=1, keepdims=True)
    return G + G.T - 2.0 * (X @ X.T)

def _median_sigma(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    D = _pairwise_sq_dists(X)
    tri = D[np.triu_indices(n, k=1)]
    tri = tri[tri > 0]
    if tri.size == 0:
        return 1.0
    return float(np.sqrt(0.5 * np.median(tri)))

def _rbf_kernel(X: np.ndarray, sigma: float) -> np.ndarray:
    D = _pairwise_sq_dists(X)
    K = np.exp(-D / (2.0 * sigma * sigma))
    return K


def _center_gram(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=float)
    mean_col = K.mean(axis=0, keepdims=True)
    mean_row = K.mean(axis=1, keepdims=True)
    mean_all = float(K.mean())
    return K - mean_col - mean_row + mean_all

def _median_sigma_subsample(X: np.ndarray, max_points: int = 200, seed: int = 0) -> float:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    if max_points is not None and n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        X = X[idx]
    return _median_sigma(X)

def hsic_statistic(X: np.ndarray, Y: np.ndarray, *, max_sigma_points: int = 200, seed: int = 0) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n = X.shape[0]
    assert Y.shape[0] == n,

    sig_x = _median_sigma_subsample(X, max_points=max_sigma_points, seed=seed)
    sig_y = _median_sigma_subsample(Y, max_points=max_sigma_points, seed=seed + 1)

    K = _rbf_kernel(X, sig_x)
    L = _rbf_kernel(Y, sig_y)

    Kc = _center_gram(K)
    Lc = _center_gram(L)

    # Biased HSIC estimator (constant factors cancel in permutation tests)
    hsic = float(np.sum(Kc * Lc) / (n * n))
    return hsic, K, L, Kc

def hsic_gamma_pvalue(X: np.ndarray, Y: np.ndarray, *, max_sigma_points: int = 200, seed: int = 0) -> Tuple[float, float]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n = X.shape[0]
    assert Y.shape[0] == n,

    test_stat, K, L, Kc = hsic_statistic(X, Y, max_sigma_points=max_sigma_points, seed=seed)
    Lc = _center_gram(L)

    # Variance estimate (same style as common reference implementations)
    var_hsic = (Kc * Lc / 6.0) ** 2
    var_hsic = float((np.sum(var_hsic) - np.trace(var_hsic)) / (n * (n - 1)))
    if n > 5:
        var_hsic = var_hsic * 72.0 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3))
    else:
        var_hsic = max(var_hsic, 1e-12)

    K_off = K - np.diag(np.diag(K))
    L_off = L - np.diag(np.diag(L))
    one = np.ones((n, 1))
    mu_x = float((one.T @ K_off @ one) / (n * (n - 1)))
    mu_y = float((one.T @ L_off @ one) / (n * (n - 1)))
    m_hsic = float((1.0 + mu_x * mu_y - mu_x - mu_y) / n)

    # Guards
    if (not np.isfinite(var_hsic)) or (not np.isfinite(m_hsic)) or var_hsic <= 1e-12 or m_hsic <= 1e-12:
        return test_stat, 1.0

    alpha = (m_hsic ** 2) / var_hsic
    beta = var_hsic / m_hsic  # scale
    pval = float(1.0 - gamma.cdf(test_stat, alpha, scale=beta))
    pval = max(min(pval, 1.0), 0.0)
    return test_stat, pval

def hsic_perm_pvalue(X: np.ndarray, Y: np.ndarray, *, n_perm: int = 200, max_sigma_points: int = 200, seed: int = 0) -> Tuple[float, float]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n = X.shape[0]
    assert Y.shape[0] == n,

    stat_obs, K, L, Kc = hsic_statistic(X, Y, max_sigma_points=max_sigma_points, seed=seed)
    # Pre-center K once; permute L
    rng = np.random.default_rng(seed + 12345)
    stats = []
    for _ in range(int(n_perm)):
        perm = rng.permutation(n)
        Lp = L[np.ix_(perm, perm)]
        Lpc = _center_gram(Lp)
        stats.append(float(np.sum(Kc * Lpc) / (n * n)))
    stats = np.asarray(stats, dtype=float)
    pval = float((np.sum(stats >= stat_obs) + 1.0) / (len(stats) + 1.0))
    return stat_obs, pval

def hsic_pvalue(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    method: str = "perm",
    n_perm: int = 200,
    max_sigma_points: int = 200,
    seed: int = 0,
) -> Tuple[float, float]:
    method = str(method).lower()
    if method in ["perm", "permutation"]:
        return hsic_perm_pvalue(X, Y, n_perm=n_perm, max_sigma_points=max_sigma_points, seed=seed)
    if method in ["gamma", "gaussian", "approx"]:
        return hsic_gamma_pvalue(X, Y, max_sigma_points=max_sigma_points, seed=seed)
    raise ValueError(f"Unknown HSIC method: {method}")




# ------------------------------------------------------------
# HSIC speed/compatibility shim
# ------------------------------------------------------------
_HSIC_PVALUE_ORIG = hsic_pvalue

def enable_fast_hsic(max_sigma_points: int = 200, seed: int = 0) -> None:
    global hsic_pvalue

    def _wrapped(
        X: np.ndarray,
        Y: np.ndarray,
        *,
        method: str = "perm",
        n_perm: int = 200,
        max_sigma_points: int = max_sigma_points,
        seed: int = seed,
    ) -> Tuple[float, float]:
        return _HSIC_PVALUE_ORIG(
            X,
            Y,
            method=method,
            n_perm=n_perm,
            max_sigma_points=max_sigma_points,
            seed=seed,
        )

    hsic_pvalue = _wrapped

def disable_fast_hsic() -> None:
    global hsic_pvalue
    hsic_pvalue = _HSIC_PVALUE_ORIG

# ============================================================
# 3) DL-GIN (Phase I): nullspace projection + HSIC
# ============================================================

def estimate_rank_by_relative_threshold(svals: np.ndarray, rel_tol: float) -> int:
    svals = np.asarray(svals, dtype=float)
    if svals.size == 0:
        return 0
    mx = float(svals[0])
    if mx <= 1e-12:
        return 0
    return int(np.sum(svals >= rel_tol * mx))

def estimate_rank_by_noise_edge(svals: np.ndarray, n: int, n_rows: int, n_cols: int, k: float = 1.5) -> int:

    svals = np.asarray(svals, dtype=float)
    if svals.size == 0:
        return 0
    tau = float(k * (np.sqrt(n_rows) + np.sqrt(n_cols)) / np.sqrt(max(n - 1, 1)))
    return int(np.sum(svals > tau))

def left_nullspace_basis(
    M: np.ndarray,
    *,
    n: int,
    method: str = "noise",
    rel_tol: float = 0.3,
    noise_k: float = 1.5,
) -> np.ndarray:

    U, S, _ = np.linalg.svd(M, full_matrices=True)
    method = str(method).lower()
    if method in ["noise", "noise_edge", "rmt"]:
        rank = estimate_rank_by_noise_edge(S, n=n, n_rows=M.shape[0], n_cols=M.shape[1], k=noise_k)
    elif method in ["relative", "rel"]:
        rank = estimate_rank_by_relative_threshold(S, rel_tol=rel_tol)
    else:
        raise ValueError(f"Unknown rank method: {method}")
    rank = min(rank, M.shape[0])
    return U[:, rank:]  # left nullspace basis

def dl_gin_pvalue(
    X: np.ndarray,
    C: Iterable[int],
    *,
    alpha_hsic: float = 0.05,
    rank_method: str = "noise",
    rel_tol: float = 0.3,
    noise_k: float = 1.5,
    hsic_method: str = "perm",
    hsic_n_perm: int = 200,
    hsic_max_sigma_points: int = 200,
    seed: int = 0,
) -> float:
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    C = sorted(set(C))
    R = [j for j in range(p) if j not in set(C)]
    if len(R) == 0:
        return 0.0  # do not merge into a single cluster; R empty makes the test vacuous

    Xc = X[:, C]
    Xr = X[:, R]
    Xc0 = Xc - Xc.mean(axis=0, keepdims=True)
    Xr0 = Xr - Xr.mean(axis=0, keepdims=True)

    # Sample cross-covariance (ddof=1 scaling matches most numerical practice; paper uses 1/n)
    Sigma_cr = (Xc0.T @ Xr0) / max(n - 1, 1)

    W = left_nullspace_basis(Sigma_cr, n=n, method=rank_method, rel_tol=rel_tol, noise_k=noise_k)
    if W.shape[1] == 0:
        # no usable projection -> cannot certify "GIN", so reject merge
        return 0.0

    Z = Xc0 @ W  # n x d
    _, pval = hsic_pvalue(
        Z,
        Xr0,
        method=hsic_method,
        n_perm=hsic_n_perm,
        max_sigma_points=hsic_max_sigma_points,
        seed=seed,
    )
    return float(pval)

def agglomerative_gin_clustering(
    X: np.ndarray,
    *,
    alpha_gin: float,
    rank_method: str = "noise",
    rel_tol: float = 0.3,
    noise_k: float = 1.5,
    hsic_method: str = "perm",
    hsic_n_perm: int = 200,
    hsic_max_sigma_points: int = 200,
    max_merges: Optional[int] = None,
    max_pair_checks: Optional[int] = None,
    seed: int = 0,
) -> List[Set[int]]:
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    clusters: List[Set[int]] = [{i} for i in range(p)]
    merges = 0

    while True:
        merged = False

        clusters_sorted = sorted(clusters, key=lambda s: (len(s), min(s)))

        pairs = []
        for a in range(len(clusters_sorted)):
            for b in range(a + 1, len(clusters_sorted)):
                Ca = clusters_sorted[a]
                Cb = clusters_sorted[b]
                pairs.append((len(Ca) + len(Cb), min(Ca), min(Cb), Ca, Cb))
        pairs.sort(key=lambda t: (t[0], t[1], t[2]))

        checks = 0
        for _, _, _, Ca, Cb in pairs:
            C = Ca | Cb
            if len(C) == p:
                continue  # never merge into a single cluster (R would be empty)
            pval = dl_gin_pvalue(
                X,
                C,
                alpha_hsic=alpha_gin,
                rank_method=rank_method,
                rel_tol=rel_tol,
                noise_k=noise_k,
                hsic_method=hsic_method,
                hsic_n_perm=hsic_n_perm,
                hsic_max_sigma_points=hsic_max_sigma_points,
                seed=seed + merges * 10000 + checks,
            )
            checks += 1
            if pval > alpha_gin:
                new_clusters = [S for S in clusters if S != Ca and S != Cb]
                new_clusters.append(C)
                clusters = new_clusters
                merged = True
                merges += 1
                break

            if (max_pair_checks is not None) and (checks >= max_pair_checks):
                break

        if not merged:
            break
        if (max_merges is not None) and (merges >= max_merges):
            break

    return [set(sorted(list(c))) for c in clusters]


# ============================================================
# 4) Phase II: CA-RCD = (RCD baseline) + (cluster-aware pruning)
# ============================================================

def ols_fit_and_residual(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = y.shape[0]
    if X.size == 0:
        resid = y - y.mean()
        beta = np.array([y.mean()])  # intercept only
        return beta, resid

    # add intercept
    X1 = np.column_stack([X, np.ones(n)])
    # solve least squares
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    return beta, resid

def refit_linear_sem(X: np.ndarray, parents: List[List[int]]) -> np.ndarray:
    n, p = X.shape
    B = np.zeros((p, p), dtype=float)
    for j in range(p):
        Pj = parents[j]
        if len(Pj) == 0:
            continue
        beta, _ = ols_fit_and_residual(X[:, j], X[:, Pj])
        coef = beta[:-1]  # exclude intercept
        for k, i in enumerate(Pj):
            B[i, j] = coef[k]
    return B

def compute_residuals_from_B(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    n, p = X.shape
    R = np.zeros_like(X)
    for j in range(p):
        Pj = list(np.where(np.abs(B[:, j]) > 0)[0])
        if len(Pj) == 0:
            _, resid = ols_fit_and_residual(X[:, j], np.empty((n, 0)))
        else:
            _, resid = ols_fit_and_residual(X[:, j], X[:, Pj])
        R[:, j] = resid
    return R

def run_direct_lingam(X: np.ndarray) -> np.ndarray:
    from lingam import DirectLiNGAM
    model = DirectLiNGAM()
    model.fit(X)

    W_to_from = np.asarray(model.adjacency_matrix_, dtype=float)
    W_from_to = W_to_from.T  # convert to our convention

    A = (np.abs(W_from_to) > 1e-6).astype(int)
    np.fill_diagonal(A, 0)
    return A

def parse_lingam_rcd_output(adj: np.ndarray, w_threshold: float = 1e-8):
    adj = np.asarray(adj, dtype=float)
    p = adj.shape[0]

    conf_pairs = set()
    B = np.zeros((p, p), dtype=float)

    for to in range(p):
        for frm in range(p):
            if to == frm:
                continue

            a = adj[to, frm]     # frm -> to (lingam convention)
            b = adj[frm, to]     # to  -> frm (lingam convention)

            # mark conf/unknown if either direction is nan
            if np.isnan(a) or np.isnan(b):
                conf_pairs.add((min(frm, to), max(frm, to)))

            # keep directed edge if finite
            if np.isfinite(a) and abs(a) > w_threshold:
                B[frm, to] = a    # convert to our convention (from,to)

    return B, conf_pairs


def prune_edges_by_hsic(
    X: np.ndarray,
    B: np.ndarray,
    *,
    alpha: float = 0.05,
    hsic_method: str = "perm",
    hsic_n_perm: int = 200,
    hsic_max_sigma_points: int = 200,
    max_rounds: int = 1,
    seed: int = 0,
) -> np.ndarray:

    X = np.asarray(X, dtype=float)
    n, p = X.shape
    Bp = B.copy()

    for rr in range(int(max_rounds)):
        changed = False
        for j in range(p):
            parents = list(np.where(np.abs(Bp[:, j]) > 0)[0])
            if len(parents) == 0:
                continue
            for i in parents:
                P_wo = [k for k in parents if k != i]
                if len(P_wo) == 0:
                    _, resid = ols_fit_and_residual(X[:, j], np.empty((n, 0)))
                else:
                    _, resid = ols_fit_and_residual(X[:, j], X[:, P_wo])
                _, pval = hsic_pvalue(
                    resid,
                    X[:, i],
                    method=hsic_method,
                    n_perm=hsic_n_perm,
                    max_sigma_points=hsic_max_sigma_points,
                    seed=seed + 100000 * rr + 1000 * j + i,
                )
                if pval > alpha:
                    Bp[i, j] = 0.0
                    changed = True
        if not changed:
            break
    return Bp

def _make_rcd(**kwargs):
    import inspect
    from lingam import RCD
    sig = inspect.signature(RCD)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return RCD(**filtered)

def ca_rcd(
    X: np.ndarray,
    clusters: List[Set[int]],
    *,
    # RCD hyperparameters (IMPORTANT: defaults in lingam are often too restrictive)
    rcd_max_explanatory_num: Optional[int] = None,
    rcd_cor_alpha: float = 0.05,
    rcd_ind_alpha: float = 0.05,
    rcd_shapiro_alpha: float = 0.05,
    rcd_MLHSICR: bool = False,
    rcd_bw_method: str = "mdbs",
    rcd_independence: str = "hsic",
    # Post-processing
    w_threshold: float = 1e-6,
    do_prune: bool = False,
    prune_alpha: float = 0.05,
    prune_hsic_method: str = "perm",
    prune_hsic_n_perm: int = 200,
    prune_hsic_max_sigma_points: int = 200,
    prune_max_rounds: int = 1,
    verbose: bool = False,
    seed: int = 0,
) -> Tuple[np.ndarray, Set[Tuple[int, int]]]:

    X = np.asarray(X, dtype=float)
    n, p = X.shape

    if rcd_max_explanatory_num is None:
        # Practical default: allow moderate in-degree but keep runtime reasonable
        rcd_max_explanatory_num = min(6, p - 1)

    model = _make_rcd(
        max_explanatory_num=int(rcd_max_explanatory_num),
        cor_alpha=float(rcd_cor_alpha),
        ind_alpha=float(rcd_ind_alpha),
        shapiro_alpha=float(rcd_shapiro_alpha),
        MLHSICR=bool(rcd_MLHSICR),
        bw_method=str(rcd_bw_method),
        independence=str(rcd_independence),
    )
    model.fit(X)

    B_init, conf_pairs = parse_lingam_rcd_output(model.adjacency_matrix_, w_threshold=w_threshold)

    # Cluster map
    cl_of = np.empty(p, dtype=int)
    for ci, C in enumerate(clusters):
        for v in C:
            cl_of[v] = ci

    # Remove within-cluster directed edges (forbidden)
    for i in range(p):
        for j in range(p):
            if i != j and cl_of[i] == cl_of[j]:
                B_init[i, j] = 0.0

    # Remove within-cluster conf pairs (local latents explain these; we only keep cross-cluster)
    conf_pairs = {pair for pair in conf_pairs if cl_of[pair[0]] != cl_of[pair[1]]}

    # Optional extra pruning (can hurt recall if too aggressive)
    B_work = B_init
    if do_prune and prune_max_rounds > 0:
        B_work = prune_edges_by_hsic(
            X, B_work,
            alpha=prune_alpha,
            hsic_method=prune_hsic_method,
            hsic_n_perm=prune_hsic_n_perm,
            hsic_max_sigma_points=prune_hsic_max_sigma_points,
            max_rounds=prune_max_rounds,
            seed=seed,
        )

    # Refit by OLS given parent sets
    parents = [list(np.where(np.abs(B_work[:, j]) > 0)[0]) for j in range(p)]
    B_refit = refit_linear_sem(X, parents)

    if verbose:
        nnz = int(np.sum(np.abs(B_refit) > 0))
        print(f"CA-RCD: directed edges kept = {nnz}, confounded pairs kept = {len(conf_pairs)} (max_explanatory_num={rcd_max_explanatory_num})")

    return B_refit, conf_pairs





# ============================================================
# 5) Phase III: Typing via inter-span estimation + projection
# ============================================================

def estimate_inter_span_from_residuals(
    R: np.ndarray,
    clusters: List[Set[int]],
    *,
    max_rank: int = 10,
    eigengap_min_rel: float = 0.05,
    eigengap_min_abs: float = 1e-3,
) -> Tuple[np.ndarray, int, np.ndarray]:

    R = np.asarray(R, dtype=float)
    n, p = R.shape
    Sigma = np.cov(R, rowvar=False, ddof=1)  # p x p

    Sigma_off = Sigma.copy()
    for C in clusters:
        idx = list(C)
        Sigma_off[np.ix_(idx, idx)] = 0.0

    U, s, _ = np.linalg.svd(Sigma_off, full_matrices=False)

    if s.size == 0 or float(s[0]) <= 1e-10:
        return np.zeros((p, 0)), 0, s

    m = min(int(max_rank), int(s.size - 1))
    if m <= 0:
        return np.zeros((p, 0)), 0, s

    gaps = s[:m] - s[1:m + 1]
    k = int(np.argmax(gaps)) + 1
    gap_max = float(gaps[k - 1])

    # Guard against flat spectra (common when r=0 or n is small)
    if (gap_max <= max(eigengap_min_abs, eigengap_min_rel * float(s[0]))):
        r_hat = 0
    else:
        r_hat = k

    U_hat = U[:, :r_hat] if r_hat > 0 else np.zeros((p, 0))
    return U_hat, int(r_hat), s

def project_orthogonal(R: np.ndarray, U: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    if U.size == 0:
        return R
    P = np.eye(U.shape[0]) - U @ U.T
    return R @ P

def type_confounded_pairs(
    X: np.ndarray,
    B_hat: np.ndarray,
    conf_pairs: Set[Tuple[int, int]],
    clusters: List[Set[int]],
    *,
    alpha_hsic: float = 0.05,
    max_rank: int = 10,
    hsic_method: str = "perm",
    hsic_n_perm: int = 200,
    hsic_max_sigma_points: int = 200,
    seed: int = 0,
) -> Tuple[Dict[Tuple[int, int], str], Dict[str, float]]:
    X = np.asarray(X, dtype=float)
    R = compute_residuals_from_B(X, B_hat)
    U_hat, r_hat, svals = estimate_inter_span_from_residuals(R, clusters, max_rank=max_rank)
    R_perp = project_orthogonal(R, U_hat)

    labels: Dict[Tuple[int, int], str] = {}
    inter_count = 0
    total = 0

    for (i, j) in conf_pairs:
        total += 1
        _, pval = hsic_pvalue(
            R_perp[:, i],
            R_perp[:, j],
            method=hsic_method,
            n_perm=hsic_n_perm,
            max_sigma_points=hsic_max_sigma_points,
            seed=seed + 1000 * i + j,
        )
        if pval > alpha_hsic:
            labels[(i, j)] = "inter"
            inter_count += 1
        else:
            labels[(i, j)] = "intra_or_mixed"

    inter_fraction = inter_count / total if total > 0 else 0.0
    info = {"r_hat": float(r_hat), "inter_fraction": float(inter_fraction)}
    return labels, info






# ============================================================
# 6) Synthetic NoN generator (Appendix A.3 style)
# ============================================================

@dataclass
class SynthConfig:
    p: int = 30
    m: int = 6
    r_int: int = 3
    n: int = 1000
    edge_prob: float = 0.6
    interface_size: int = 1  
    noise_dist: str = "laplace"  
    scales: Tuple[float, float, float, float, float] = (0.6, 0.9, 1.3, 0.7, 1.0)  
    seed0: int = 123

   
    inter_latents_per_cluster: int = 1
    avoid_confounded_directed: bool = False   
    min_directed_edges: int = 1          
    max_resample_dag: int = 50         


def _draw_noise(rng: np.random.Generator, shape: Tuple[int, ...], dist: str) -> np.ndarray:
    if dist == "laplace":
        return rng.laplace(size=shape)
    if dist == "student":
        return rng.standard_t(df=3, size=shape)
    if dist == "gaussian":
        return rng.normal(size=shape)
    # default
    return rng.laplace(size=shape)

def generate_base_sem(cfg: SynthConfig, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    p, m, r_int = cfg.p, cfg.m, cfg.r_int

    
    sizes = [p // m] * m
    for i in range(p % m):
        sizes[i] += 1
    rng.shuffle(sizes)

    clusters: List[List[int]] = []
    idx = 0
    for s in sizes:
        clusters.append(list(range(idx, idx + s)))
        idx += s

   
    interface_nodes: List[List[int]] = []
    for nodes in clusters:
        k = min(cfg.interface_size, len(nodes))
        interface_nodes.append(list(rng.choice(nodes, size=k, replace=False)))

   
    per_cluster = max(1, min(int(cfg.inter_latents_per_cluster), r_int))
    Kc: List[Set[int]] = []
    for _ in range(m):
        Kc.append(set(rng.choice(r_int, size=per_cluster, replace=False)))

   
    Gamma = np.zeros((p, r_int), dtype=float)
    for c in range(m):
        for j in interface_nodes[c]:
            for k in Kc[c]:
                Gamma[j, k] = float(rng.uniform(0.5, 1.6) * rng.choice([-1, 1]))

  
    q = m
    Lambda = np.zeros((p, q), dtype=float)
    for c, nodes in enumerate(clusters):
        for j in nodes:
            Lambda[j, c] = float(rng.uniform(0.5, 1.6) * rng.choice([-1, 1]))

   
    def _sample_directed_B(allow_confounded: bool) -> np.ndarray:
        Btmp = np.zeros((p, p), dtype=float)
        for ca in range(m):
            for cb in range(ca + 1, m):
          
                if (not allow_confounded) and cfg.avoid_confounded_directed and (len(Kc[ca].intersection(Kc[cb])) > 0):
                    continue
                src = interface_nodes[ca]
                tgt = interface_nodes[cb]
                for i in src:
                    for j in tgt:
                        if rng.random() < cfg.edge_prob:
                            coef = float(rng.uniform(0.3, 0.9) * rng.choice([-1, 1]))
                            Btmp[i, j] = coef  # i -> j
        return Btmp

  
    B = _sample_directed_B(allow_confounded=False)
    if cfg.min_directed_edges > 0:
        n_edges = int(np.sum(np.abs(B) > 0))
        tries = 0
        while n_edges < cfg.min_directed_edges and tries < cfg.max_resample_dag:
            tries += 1
            B = _sample_directed_B(allow_confounded=False)
            n_edges = int(np.sum(np.abs(B) > 0))

        if n_edges < cfg.min_directed_edges and cfg.avoid_confounded_directed:
            tries = 0
            while n_edges < cfg.min_directed_edges and tries < cfg.max_resample_dag:
                tries += 1
                B = _sample_directed_B(allow_confounded=True)
                n_edges = int(np.sum(np.abs(B) > 0))

    target_rank = min(r_int, sum(len(v) for v in interface_nodes))
    for _ in range(10):
        if np.linalg.matrix_rank(Gamma) >= target_rank:
            break
  
        Kc = []
        for _c in range(m):
            Kc.append(set(rng.choice(r_int, size=per_cluster, replace=False)))
        Gamma[:] = 0.0
        for c in range(m):
            for j in interface_nodes[c]:
                for k in Kc[c]:
                    Gamma[j, k] = float(rng.uniform(0.5, 1.6) * rng.choice([-1, 1]))

    return {
        "clusters": [set(c) for c in clusters],
        "interface_nodes": interface_nodes,
        "B": B,
        "Lambda": Lambda,
        "Gamma": Gamma,
        "Kc": Kc,
    }


def simulate_from_sem(base: Dict, cfg: SynthConfig, scale: float, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    B = base["B"]
    Lambda = base["Lambda"]
    Gamma = base["Gamma"] * float(scale)
    clusters = base["clusters"]

    n, p = cfg.n, cfg.p
    q = Lambda.shape[1]
    r_int = Gamma.shape[1]

    L_loc = _draw_noise(rng, (n, q), cfg.noise_dist)
    L_int = _draw_noise(rng, (n, r_int), cfg.noise_dist)
    E = _draw_noise(rng, (n, p), cfg.noise_dist)


    A = np.linalg.inv(np.eye(p) - B.T)
    S = L_loc @ Lambda.T + L_int @ Gamma.T + E
    X = S @ A.T
    X = standardize(X)

    A_true = (np.abs(B) > 0).astype(int)

   
    Gamma_bin = (np.abs(Gamma) > 1e-12).astype(int)
    inter_conf = np.zeros((p, p), dtype=int)
    for k in range(r_int):
        idx = list(np.where(Gamma_bin[:, k] == 1)[0])
        for i in idx:
            for j in idx:
                if i != j:
                    inter_conf[i, j] = 1

    cl_id = np.empty(p, dtype=int)
    for ci, C in enumerate(clusters):
        for v in C:
            cl_id[v] = ci

    local_conf = (cl_id.reshape(-1, 1) == cl_id.reshape(1, -1)).astype(int)
    np.fill_diagonal(local_conf, 0)


    conf_any = ((local_conf + inter_conf) > 0).astype(int)


    true_inter = ((inter_conf == 1) & (local_conf == 0)).astype(int)

    return {
        "X": X,
        "A_true": A_true,
        "clusters_true": clusters,
        "cl_id_true": cl_id,
        "conf_any": conf_any,
        "true_inter": true_inter,
        "Gamma": Gamma,
    }


# ============================================================
# 7) Baselines (PC, DirectLiNGAM, NOTEARS, GraN-DAG, FCI, RFCI, RCD)
# ============================================================
def directed_metrics(A_true: np.ndarray, A_pred: np.ndarray) -> Dict[str, float]:

    A_true = (A_true > 0).astype(int)
    A_pred = (A_pred > 0).astype(int)
    tp = int(np.sum((A_true == 1) & (A_pred == 1)))
    fp = int(np.sum((A_true == 0) & (A_pred == 1)))
    fn = int(np.sum((A_true == 1) & (A_pred == 0)))

    total_true = int(np.sum(A_true))
    total_pred = int(np.sum(A_pred))


    if total_true == 0 and total_pred == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}

def oracle_align_from_skeleton(
    skel: np.ndarray,
    oriented: np.ndarray,
    A_true: np.ndarray
) -> np.ndarray:
    p = skel.shape[0]
    A = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i + 1, p):
            if skel[i, j] == 0:
                continue
            if oriented[i, j] == 1:
                A[i, j] = 1
            elif oriented[j, i] == 1:
                A[j, i] = 1
            else:
                # use oracle truth
                if A_true[i, j] == 1:
                    A[i, j] = 1
                elif A_true[j, i] == 1:
                    A[j, i] = 1
                else:
                    # arbitrary for extra edges
                    A[i, j] = 1
    return A

def causallearn_graph_to_skel_oriented(g, node_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    import re
    from causallearn.graph.Endpoint import Endpoint

    p = len(node_names)
    name_to_idx = {name: k for k, name in enumerate(node_names)}

    graph_nums = []
    try:
        for n in g.get_nodes():
            m = re.match(r"^X(\d+)$", str(n.get_name()))
            if m:
                graph_nums.append(int(m.group(1)))
    except Exception:
        graph_nums = []

    graph_min = min(graph_nums) if graph_nums else None
    graph_max = max(graph_nums) if graph_nums else None

    def _name_to_index(name: str) -> int:
        if name in name_to_idx:
            return name_to_idx[name]

        # fallback: parse Xk
        m = re.match(r"^X(\d+)$", str(name))
        if m:
            k = int(m.group(1))
            # If graph looks like X1..Xp, map Xk -> k-1
            if graph_min == 1 and graph_max == p:
                idx = k - 1
                if 0 <= idx < p:
                    return idx
            # If graph looks like X0..X(p-1), map Xk -> k
            if graph_min == 0 and graph_max == p - 1:
                idx = k
                if 0 <= idx < p:
                    return idx
                    
            for idx in (k, k - 1):
                if 0 <= idx < p:
                    return idx

        raise KeyError(
            f"Graph node name '{name}' not found in node_names (len={p}). "
            "This is usually caused by a 0/1-based naming mismatch (e.g., X0.. vs X1..)."
        )

    skel = np.zeros((p, p), dtype=int)
    oriented = np.zeros((p, p), dtype=int)

    for e in g.get_graph_edges():
        n1 = e.get_node1().get_name()
        n2 = e.get_node2().get_name()
        i = _name_to_index(n1)
        j = _name_to_index(n2)
        skel[i, j] = 1
        skel[j, i] = 1

        ep1 = e.get_endpoint1()
        ep2 = e.get_endpoint2()

        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            oriented[i, j] = 1
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            oriented[j, i] = 1

    return skel, oriented

def run_pc(X: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    from causallearn.search.ConstraintBased.PC import pc
    node_names = [f"X{i+1}" for i in range(X.shape[1])]
    cg = pc(X, alpha=alpha, indep_test="fisherz", stable=True, node_names=node_names, show_progress=False)
    skel, oriented = causallearn_graph_to_skel_oriented(cg.G, node_names)
    return skel, oriented

def run_fci(X: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    from causallearn.search.ConstraintBased.FCI import fci
    node_names = [f"X{i+1}" for i in range(X.shape[1])]
    g, _ = fci(X, independence_test_method="fisherz", alpha=alpha, verbose=False)
    skel, oriented = causallearn_graph_to_skel_oriented(g, node_names)
    return skel, oriented, g

# Flag to avoid spamming warnings inside loops
_RFCI_FALLBACK_WARNED = False

from functools import lru_cache

@lru_cache(maxsize=1)
def _get_rfci_callable():
    candidates = [
        ("causallearn.search.ConstraintBased.RFCI", "rfci"),  # expected (some forks/versions)
        ("causallearn.search.ConstraintBased.FCI", "rfci"),   # sometimes bundled with FCI
        ("causallearn.search.FC", "rfci"),                    # older API alias used in the wild
    ]
    for mod, fn_name in candidates:
        try:
            module = __import__(mod, fromlist=[fn_name])
            fn = getattr(module, fn_name, None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None

def run_rfci(X: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    global _RFCI_FALLBACK_WARNED

    node_names = [f"X{i+1}" for i in range(X.shape[1])]

    rfci_fn = _get_rfci_callable()
    if rfci_fn is None:
        if not _RFCI_FALLBACK_WARNED:
            import warnings
            warnings.warn(
                "RFCI is not available in this causal-learn installation. Falling back to FCI "
                "so the experiments can run end-to-end. If you need true RFCI, install a build "
                "of causal-learn that includes RFCI (or remove RFCI from the method list). ",
                RuntimeWarning,
            )
            _RFCI_FALLBACK_WARNED = True

        from causallearn.search.ConstraintBased.FCI import fci
        g, _ = fci(X, independence_test_method="fisherz", alpha=alpha, verbose=False)
    else:
        g, _ = rfci_fn(X, independence_test_method="fisherz", alpha=alpha, verbose=False)

    skel, oriented = causallearn_graph_to_skel_oriented(g, node_names)
    return skel, oriented, g

def notears_linear(
    X: np.ndarray,
    lambda1: float = 0.02,
    lambda2: float = 0.005,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.3
) -> np.ndarray:
    from scipy.optimize import minimize

    X = np.asarray(X, dtype=float)
    n, d = X.shape

    def _loss(W):
        M = X @ W
        R = X - M
        loss = 0.5 / n * np.sum(R ** 2)
        G = -1.0 / n * (X.T @ R)
        return loss, G

    def _h(W):
        E = linalg.expm(W * W)
        h = np.trace(E) - d
        G = (E.T) * W * 2
        return h, G

    def _obj(w, rho, alpha):
        W = w.reshape(d, d)
        np.fill_diagonal(W, 0.0)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        # smooth l1 to keep L-BFGS happy
        eps = 1e-8
        l1 = np.sum(np.sqrt(W * W + eps))
        obj = loss + lambda1 * l1 + 0.5 * lambda2 * np.sum(W * W) + alpha * h + 0.5 * rho * h * h
        G_smooth = lambda1 * (W / np.sqrt(W * W + eps))
        G = G_loss + G_smooth + lambda2 * W + (alpha + rho * h) * G_h
        np.fill_diagonal(G, 0.0)
        return obj, G.ravel()

    w_est = np.zeros(d * d, dtype=float)
    rho, alpha = 1.0, 0.0
    h_new = np.inf

    bounds = []
    for i in range(d):
        for j in range(d):
            if i == j:
                bounds.append((0.0, 0.0))
            else:
                bounds.append((None, None))

    for _ in range(max_iter):
        sol = minimize(
            fun=lambda w: _obj(w, rho, alpha)[0],
            x0=w_est,
            jac=lambda w: _obj(w, rho, alpha)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-12}
        )
        w_est = sol.x
        W = w_est.reshape(d, d)
        np.fill_diagonal(W, 0.0)
        h_new, _ = _h(W)
        if h_new <= h_tol or rho >= rho_max:
            break
        alpha += rho * h_new
        rho *= 10.0

    W = w_est.reshape(d, d)
    np.fill_diagonal(W, 0.0)
    A = (np.abs(W) > w_threshold).astype(int)
    np.fill_diagonal(A, 0)
    return A

# "GraN-DAG" baseline: gradient-based neural DAG learning with acyclicity penalty.
# This is a faithful NOTEARS-style neural-SEM implementation (often used as GraN-DAG-like baseline in practice).
def grandag_nonlinear(
    X: np.ndarray,
    hidden: int = 32,
    lambda1: float = 0.01,
    rho_max: float = 1e4,
    h_tol: float = 1e-8,
    max_outer: int = 10,
    inner_epochs: int = 2000,
    lr: float = 5e-3,
    w_threshold: float = 0.3,
    device: Optional[str] = None
) -> np.ndarray:
    import torch
    import torch.nn as nn

    Xnp = np.asarray(X, dtype=float)
    n, d = Xnp.shape

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_t = torch.tensor(Xnp, dtype=torch.float32, device=device)

    class NodeMLP(nn.Module):
        def __init__(self, d_in, hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )
        def forward(self, z):
            return self.net(z)

    mlps = nn.ModuleList([NodeMLP(d, hidden) for _ in range(d)]).to(device)

    # adjacency logits (d x d), diag fixed to -inf effectively
    A_logits = torch.zeros((d, d), dtype=torch.float32, device=device, requires_grad=True)

    def mask_matrix():
        W = torch.sigmoid(A_logits)
        W = W * (1.0 - torch.eye(d, device=device))  # zero diag
        return W

    def acyclicity(W):
        # NOTEARS constraint: h(W) = tr(exp(W ⊙ W)) - d
        E = torch.matrix_exp(W * W)
        return torch.trace(E) - d

    rho = 1.0
    alpha = 0.0

    params = list(mlps.parameters()) + [A_logits]
    opt = torch.optim.Adam(params, lr=lr)

    for _ in range(max_outer):
        for _e in range(inner_epochs):
            opt.zero_grad()
            W = mask_matrix()

            # predict each node
            preds = []
            for j in range(d):
                m_j = W[:, j].reshape(1, -1)  # (1,d)
                X_masked = X_t * m_j  # broadcast: (n,d)
                yhat = mlps[j](X_masked)  # (n,1)
                preds.append(yhat)

            Yhat = torch.cat(preds, dim=1)  # (n,d)
            loss_mse = torch.mean((X_t - Yhat) ** 2)

            l1 = torch.sum(torch.abs(W))
            h = acyclicity(W)

            loss = loss_mse + lambda1 * l1 + alpha * h + 0.5 * rho * h * h
            loss.backward()
            opt.step()

        with torch.no_grad():
            W = mask_matrix()
            h_val = float(acyclicity(W).cpu().item())
            if h_val <= h_tol:
                break
            alpha += rho * h_val
            rho = min(rho * 10.0, rho_max)

    W_final = mask_matrix().detach().cpu().numpy()
    A = (np.abs(W_final) > w_threshold).astype(int)
    np.fill_diagonal(A, 0)
    return A


def run_rcd(
    X: np.ndarray,
    *,
    max_explanatory_num: Optional[int] = None,
    cor_alpha: float = 0.05,
    ind_alpha: float = 0.05,
    shapiro_alpha: float = 0.05,
    MLHSICR: bool = False,
    bw_method: str = "mdbs",
    independence: str = "hsic",
    w_threshold: float = 1e-6,
) -> Tuple[np.ndarray, Set[Tuple[int, int]]]:
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if max_explanatory_num is None:
        max_explanatory_num = min(6, p - 1)
    model = _make_rcd(
        max_explanatory_num=int(max_explanatory_num),
        cor_alpha=float(cor_alpha),
        ind_alpha=float(ind_alpha),
        shapiro_alpha=float(shapiro_alpha),
        MLHSICR=bool(MLHSICR),
        bw_method=str(bw_method),
        independence=str(independence),
    )
    model.fit(X)
    B_init, conf = parse_lingam_rcd_output(model.adjacency_matrix_, w_threshold=w_threshold)
    A = (np.abs(B_init) > w_threshold).astype(int)
    np.fill_diagonal(A, 0)
    return A, conf






# ============================================================
# 8) LCDNN wrapper (Phases I–III)
# ============================================================

@dataclass
class LCDNNConfig:
    # -------------------
    # Phase I (DL-GIN)
    # -------------------
    alpha_gin: float = 0.05
    dlgin_rank_method: str = "noise"   # "noise" (recommended) or "relative"
    dlgin_rel_tol: float = 0.3         # only used when dlgin_rank_method="relative"
    dlgin_noise_k: float = 1.5         # only used when dlgin_rank_method="noise"
    dlgin_hsic_method: str = "perm"    # "perm" (recommended) or "gamma"
    dlgin_hsic_n_perm: int = 200
    dlgin_hsic_max_sigma_points: int = 200

    # Standard GIN baseline (optional, for Appendix ARI table)
    stdgin_rank_method: str = "relative"
    stdgin_rel_tol: float = 0.01
    stdgin_hsic_method: str = "perm"
    stdgin_hsic_n_perm: int = 200
    stdgin_hsic_max_sigma_points: int = 200

    # -------------------
    # Phase II (CA-RCD)
    # -------------------
    rcd_max_explanatory_num: Optional[int] = None  # if None -> min(6, p-1)
    rcd_cor_alpha: float = 0.05
    rcd_ind_alpha: float = 0.05
    rcd_shapiro_alpha: float = 0.05
    rcd_MLHSICR: bool = False
    rcd_bw_method: str = "mdbs"
    rcd_independence: str = "hsic"

    ca_do_prune: bool = False
    ca_prune_alpha: float = 0.05
    ca_prune_hsic_method: str = "perm"
    ca_prune_hsic_n_perm: int = 200
    ca_prune_hsic_max_sigma_points: int = 200
    ca_prune_max_rounds: int = 1
    # Thresholding (IMPORTANT for precision in finite samples)
    ca_w_threshold: float = 0.05   # keep RCD coefficients with |w| > this
    edge_threshold: float = 0.05  # binarize B_hat into adjacency using this threshold


    # -------------------
    # Phase III (Typing)
    # -------------------
    alpha_type: float = 0.05
    typing_hsic_method: str = "perm"
    typing_hsic_n_perm: int = 200
    typing_hsic_max_sigma_points: int = 200
    max_rank: int = 10

    seed: int = 0

def run_lcdnn(X: np.ndarray, cfg: LCDNNConfig) -> Dict:
    # Phase I: DL-GIN clustering
    clusters_hat = agglomerative_gin_clustering(
        X,
        alpha_gin=cfg.alpha_gin,
        rank_method=cfg.dlgin_rank_method,
        rel_tol=cfg.dlgin_rel_tol,
        noise_k=cfg.dlgin_noise_k,
        hsic_method=cfg.dlgin_hsic_method,
        hsic_n_perm=cfg.dlgin_hsic_n_perm,
        hsic_max_sigma_points=cfg.dlgin_hsic_max_sigma_points,
        seed=cfg.seed,
    )

    # Phase II: CA-RCD
    B_hat, conf_pairs = ca_rcd(
        X,
        clusters_hat,
        rcd_max_explanatory_num=cfg.rcd_max_explanatory_num,
        rcd_cor_alpha=cfg.rcd_cor_alpha,
        rcd_ind_alpha=cfg.rcd_ind_alpha,
        rcd_shapiro_alpha=cfg.rcd_shapiro_alpha,
        rcd_MLHSICR=cfg.rcd_MLHSICR,
        rcd_bw_method=cfg.rcd_bw_method,
        rcd_independence=cfg.rcd_independence,
        w_threshold=cfg.ca_w_threshold,
        do_prune=cfg.ca_do_prune,
        prune_alpha=cfg.ca_prune_alpha,
        prune_hsic_method=cfg.ca_prune_hsic_method,
        prune_hsic_n_perm=cfg.ca_prune_hsic_n_perm,
        prune_hsic_max_sigma_points=cfg.ca_prune_hsic_max_sigma_points,
        prune_max_rounds=cfg.ca_prune_max_rounds,
        seed=cfg.seed,
    )

    # Phase III: typing
    labels, info = type_confounded_pairs(
        X,
        B_hat,
        conf_pairs,
        clusters_hat,
        alpha_hsic=cfg.alpha_type,
        max_rank=cfg.max_rank,
        hsic_method=cfg.typing_hsic_method,
        hsic_n_perm=cfg.typing_hsic_n_perm,
        hsic_max_sigma_points=cfg.typing_hsic_max_sigma_points,
        seed=cfg.seed,
    )

    A_hat = (np.abs(B_hat) > cfg.edge_threshold).astype(int)
    np.fill_diagonal(A_hat, 0)

    return {
        "clusters_hat": clusters_hat,
        "A_hat": A_hat,
        "B_hat": B_hat,
        "conf_pairs": conf_pairs,
        "type_labels": labels,
        "type_info": info,
    }






# ============================================================
# 9) Table formatting (UAI-style caption ABOVE table)
# ============================================================
def fmt_num(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (not np.isfinite(x))):
        return "--"
    return f"{x:.{nd}f}"

def make_ranked_format_table(df: pd.DataFrame, higher_is_better_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in higher_is_better_cols:
        vals = df[col].astype(float).values
        order = np.argsort(-vals)  # descending
        best = order[0]
        second = order[1] if len(order) > 1 else None
        for i in range(len(df)):
            s = fmt_num(float(df.iloc[i][col]), 2)
            if i == best:
                s = r"\textbf{" + s + "}"
            elif second is not None and i == second:
                s = r"\underline{" + s + "}"
            out.iloc[i, out.columns.get_loc(col)] = s
    return out

def df_to_uai_table_tex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    table_env: str = "table",
    footnotesize: bool = True
) -> str:
    lines = []
    lines.append(fr"\begin{{{table_env}}}[t]")
    lines.append(r"\centering")
    if footnotesize:
        lines.append(r"\footnotesize")
    lines.append(fr"\caption{{{caption}}}")
    lines.append(fr"\label{{{label}}}")
    lines.append(df.to_latex(index=False, escape=False, booktabs=True))
    lines.append(fr"\end{{{table_env}}}")
    return "\n".join(lines)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# 10) Synthetic experiments runner (Section 6.2 + Appendix C.1–C.4)
# ============================================================
def eval_ari(cl_true: np.ndarray, clusters_hat: List[Set[int]]) -> float:
    p = cl_true.size
    cl_hat = np.empty(p, dtype=int)
    for ci, C in enumerate(clusters_hat):
        for v in C:
            cl_hat[v] = ci
    return float(adjusted_rand_score(cl_true, cl_hat))

def evaluate_typing_accuracy(
    conf_pairs_hat: Set[Tuple[int,int]],
    type_labels_hat: Dict[Tuple[int,int], str],
    true_inter: np.ndarray
) -> Tuple[float, float]:
    if len(conf_pairs_hat) == 0:
        return 0.0, 0.0
    inter_cnt = 0
    correct = 0
    total = 0
    for (i, j) in conf_pairs_hat:
        pred_inter = (type_labels_hat.get((i, j), "intra_or_mixed") == "inter")
        gt_inter = (true_inter[i, j] == 1)
        inter_cnt += int(pred_inter)
        correct += int(pred_inter == gt_inter)
        total += 1
    return float(inter_cnt / total), float(correct / total)

def run_synthetic_experiments(
    synth_cfg: SynthConfig,
    lcdnn_cfg: LCDNNConfig,
    n_trials: int = 20,
    alpha_pc_fci: float = 0.01,
    out_dir: str = "tables_generated",
    run_grandag: bool = True
) -> Dict[str, pd.DataFrame]:
    ensure_dir(out_dir)

    methods = ["PC", "DirectLiNGAM", "NOTEARS", "GraN-DAG", "FCI", "RFCI", "RCD", "LCDNN"]
    if not run_grandag:
        methods.remove("GraN-DAG")

    # Store per-run metrics
    rows = []

    # For appendix tables
    ari_rows = []
    f1_per_target = {m: [] for m in methods}
    typing_per_target = []

    scales = list(synth_cfg.scales)
    target_names = [f"N{i}" for i in range(1, 6)]

    pbar = tqdm(range(n_trials), desc="Synthetic trials")
    for t in pbar:
        base = generate_base_sem(synth_cfg, seed=synth_cfg.seed0 + 1000 * t)

        # true clusters are shared across targets
        clusters_true = base["clusters"]
        p = synth_cfg.p
        cl_true = np.empty(p, dtype=int)
        for ci, C in enumerate(clusters_true):
            for v in C:
                cl_true[v] = ci

        for k, scale in enumerate(scales):
            sim = simulate_from_sem(base, synth_cfg, scale=scale, seed=synth_cfg.seed0 + 1000 * t + 10 * k)
            X = sim["X"]
            A_true = sim["A_true"]
            true_inter = sim["true_inter"]

            # ---------- Phase I clustering: Standard GIN vs DL-GIN ----------
            clusters_std = agglomerative_gin_clustering(
                X,
                alpha_gin=lcdnn_cfg.alpha_gin,
                rank_method=lcdnn_cfg.stdgin_rank_method,
                rel_tol=lcdnn_cfg.stdgin_rel_tol,
                noise_k=lcdnn_cfg.dlgin_noise_k,  # unused for stdgin, harmless
                hsic_method=lcdnn_cfg.stdgin_hsic_method,
                hsic_n_perm=lcdnn_cfg.stdgin_hsic_n_perm,
                hsic_max_sigma_points=lcdnn_cfg.stdgin_hsic_max_sigma_points,
                seed=int(lcdnn_cfg.seed) + 10000 * t + 100 * k + 1,
            )
            clusters_dl = agglomerative_gin_clustering(
                X,
                alpha_gin=lcdnn_cfg.alpha_gin,
                rank_method=lcdnn_cfg.dlgin_rank_method,
                rel_tol=lcdnn_cfg.dlgin_rel_tol,
                noise_k=lcdnn_cfg.dlgin_noise_k,
                hsic_method=lcdnn_cfg.dlgin_hsic_method,
                hsic_n_perm=lcdnn_cfg.dlgin_hsic_n_perm,
                hsic_max_sigma_points=lcdnn_cfg.dlgin_hsic_max_sigma_points,
                seed=int(lcdnn_cfg.seed) + 10000 * t + 100 * k + 2,
            )
            ari_std = eval_ari(cl_true, clusters_std)
            ari_dl = eval_ari(cl_true, clusters_dl)
            ari_rows.append({"target": target_names[k], "trial": t, "Standard GIN": ari_std, "DL-GIN (ours)": ari_dl})

            # ---------- Baselines & LCDNN ----------
            # PC
            skel_pc, ori_pc = run_pc(X, alpha=alpha_pc_fci)
            A_pc = oracle_align_from_skeleton(skel_pc, ori_pc, A_true)
            rows.append({"target": target_names[k], "trial": t, "method": "PC", **directed_metrics(A_true, A_pc)})
            f1_per_target["PC"].append((target_names[k], directed_metrics(A_true, A_pc)["f1"]))

            # DirectLiNGAM
            A_dl = run_direct_lingam(X)
            rows.append({"target": target_names[k], "trial": t, "method": "DirectLiNGAM", **directed_metrics(A_true, A_dl)})
            f1_per_target["DirectLiNGAM"].append((target_names[k], directed_metrics(A_true, A_dl)["f1"]))

            # NOTEARS
            A_nt = notears_linear(X, lambda1=0.02, lambda2=0.005, max_iter=50, w_threshold=0.2)
            rows.append({"target": target_names[k], "trial": t, "method": "NOTEARS", **directed_metrics(A_true, A_nt)})
            f1_per_target["NOTEARS"].append((target_names[k], directed_metrics(A_true, A_nt)["f1"]))

            # GraN-DAG (neural, NOTEARS-style)
            if run_grandag:
                A_gd = grandag_nonlinear(X, hidden=32, lambda1=0.01, max_outer=8, inner_epochs=800, w_threshold=0.25)
                rows.append({"target": target_names[k], "trial": t, "method": "GraN-DAG", **directed_metrics(A_true, A_gd)})
                f1_per_target["GraN-DAG"].append((target_names[k], directed_metrics(A_true, A_gd)["f1"]))

            # FCI
            skel_fci, ori_fci, g_fci = run_fci(X, alpha=alpha_pc_fci)
            A_fci = oracle_align_from_skeleton(skel_fci, ori_fci, A_true)
            rows.append({"target": target_names[k], "trial": t, "method": "FCI", **directed_metrics(A_true, A_fci)})
            f1_per_target["FCI"].append((target_names[k], directed_metrics(A_true, A_fci)["f1"]))

            # RFCI
            skel_rfci, ori_rfci, g_rfci = run_rfci(X, alpha=alpha_pc_fci)
            A_rfci = oracle_align_from_skeleton(skel_rfci, ori_rfci, A_true)
            rows.append({"target": target_names[k], "trial": t, "method": "RFCI", **directed_metrics(A_true, A_rfci)})
            f1_per_target["RFCI"].append((target_names[k], directed_metrics(A_true, A_rfci)["f1"]))

            # RCD
            A_rcd, conf_rcd = run_rcd(X)
            rows.append({"target": target_names[k], "trial": t, "method": "RCD", **directed_metrics(A_true, A_rcd)})
            f1_per_target["RCD"].append((target_names[k], directed_metrics(A_true, A_rcd)["f1"]))

            # LCDNN (Phases I–III)
            lcdnn_out = run_lcdnn(X, lcdnn_cfg)
            A_lcdnn = lcdnn_out["A_hat"]
            rows.append({"target": target_names[k], "trial": t, "method": "LCDNN", **directed_metrics(A_true, A_lcdnn)})
            f1_per_target["LCDNN"].append((target_names[k], directed_metrics(A_true, A_lcdnn)["f1"]))

            inter_frac, typing_acc = evaluate_typing_accuracy(lcdnn_out["conf_pairs"], lcdnn_out["type_labels"], true_inter)
            typing_per_target.append({
                "target": target_names[k],
                "trial": t,
                "inter_fraction": inter_frac,
                "typing_accuracy": typing_acc,
                "r_hat": float(lcdnn_out["type_info"]["r_hat"]),
            })

    # -----------------------
    # Aggregate main summary
    # -----------------------
    df_runs = pd.DataFrame(rows)
    summary = (
        df_runs.groupby("method")[["precision", "recall", "f1"]]
        .mean()
        .reset_index()
        .sort_values("method")
        .reset_index(drop=True)
    )

    # Reorder methods to match the paper
    method_order = ["PC", "DirectLiNGAM", "NOTEARS", "GraN-DAG", "FCI", "RFCI", "RCD", "LCDNN"]
    if not run_grandag:
        method_order.remove("GraN-DAG")
    summary["method"] = pd.Categorical(summary["method"], categories=method_order, ordered=True)
    summary = summary.sort_values("method").reset_index(drop=True)

    summary_ranked = make_ranked_format_table(summary, ["precision", "recall", "f1"])

    # -----------------------
    # Table: synth_summary_main
    # -----------------------
    tex_main = df_to_uai_table_tex(
        summary_ranked.rename(columns={"method": "Method", "precision": "Prec.", "recall": "Rec.", "f1": "F1"}),
        caption="Synthetic NoNs: directed-edge recovery (macro-averaged over targets and trials). Best is bold; second-best is underlined.",
        label="tab:synth_summary_main"
    )
    with open(os.path.join(out_dir, "synth_summary_main.tex"), "w") as f:
        f.write(tex_main)

    # -----------------------
    # Appendix C.1: clustering ARI (Standard GIN vs DL-GIN)
    # -----------------------
    df_ari = pd.DataFrame(ari_rows)
    ari_tbl = df_ari.groupby("target")[["Standard GIN", "DL-GIN (ours)"]].mean().reset_index()
    ari_tbl.loc[len(ari_tbl)] = ["Avg."] + list(ari_tbl[["Standard GIN", "DL-GIN (ours)"]].mean().values)
    ari_tbl_fmt = ari_tbl.copy()
    for c in ["Standard GIN", "DL-GIN (ours)"]:
        ari_tbl_fmt[c] = ari_tbl[c].apply(lambda x: fmt_num(float(x), 2))
    tex_ari = df_to_uai_table_tex(
        ari_tbl_fmt.rename(columns={"target": "Target"}),
        caption="Synthetic NoNs: clustering quality (ARI) for Standard GIN vs DL-GIN.",
        label="tabC:synth_ari"
    )
    with open(os.path.join(out_dir, "synth_ari.tex"), "w") as f:
        f.write(tex_ari)

    # -----------------------
    # Appendix C.2: full structure recovery table (P/R/F1)
    # -----------------------
    full_tbl = (
        df_runs.groupby("method")[["precision", "recall", "f1"]]
        .mean()
        .reset_index()
    )
    full_tbl["method"] = pd.Categorical(full_tbl["method"], categories=method_order, ordered=True)
    full_tbl = full_tbl.sort_values("method").reset_index(drop=True)
    full_tbl_fmt = make_ranked_format_table(full_tbl, ["precision", "recall", "f1"])
    tex_full = df_to_uai_table_tex(
        full_tbl_fmt.rename(columns={"method": "Method", "precision": "Precision", "recall": "Recall", "f1": "F1"}),
        caption="Synthetic NoNs: directed-edge recovery (macro average).",
        label="tabC:synth_structure_full"
    )
    with open(os.path.join(out_dir, "synth_structure_full.tex"), "w") as f:
        f.write(tex_full)

    # -----------------------
    # Appendix C.3: per-target F1 for selected methods (FCI, RCD, LCDNN)
    # -----------------------
    df_f1 = df_runs[df_runs["method"].isin(["FCI", "RCD", "LCDNN"])].copy()
    f1_stage = df_f1.groupby(["method", "target"])["f1"].mean().reset_index()
    f1_pivot = f1_stage.pivot(index="method", columns="target", values="f1").reset_index()
    # add Avg.
    f1_pivot["Avg."] = f1_pivot[[c for c in f1_pivot.columns if c.startswith("N")]].mean(axis=1)
    # format
    f1_fmt = f1_pivot.copy()
    for c in [c for c in f1_fmt.columns if c != "method"]:
        f1_fmt[c] = f1_fmt[c].apply(lambda x: fmt_num(float(x), 2))
    tex_f1 = df_to_uai_table_tex(
        f1_fmt.rename(columns={"method": "Method"}),
        caption="Synthetic NoNs: directed-edge F1 per target (selected methods).",
        label="tabC:synth_f1_per_target"
    )
    with open(os.path.join(out_dir, "synth_f1_per_target.tex"), "w") as f:
        f.write(tex_f1)

    # -----------------------
    # Appendix C.4: typing table for LCDNN
    # -----------------------
    df_type = pd.DataFrame(typing_per_target)
    type_tbl = df_type.groupby("target")[["inter_fraction", "typing_accuracy", "r_hat"]].mean().reset_index()
    type_tbl.loc[len(type_tbl)] = ["Avg."] + list(type_tbl[["inter_fraction", "typing_accuracy", "r_hat"]].mean().values)
    type_fmt = type_tbl.copy()
    type_fmt["inter_fraction"] = type_tbl["inter_fraction"].apply(lambda x: fmt_num(float(x), 2))
    type_fmt["typing_accuracy"] = type_tbl["typing_accuracy"].apply(lambda x: fmt_num(float(x), 2))
    type_fmt["r_hat"] = type_tbl["r_hat"].apply(lambda x: fmt_num(float(x), 2))
    tex_type = df_to_uai_table_tex(
        type_fmt.rename(columns={"target": "Target", "inter_fraction": "Inter fraction", "typing_accuracy": "Accuracy", "r_hat": r"$\hat r$"}),
        caption="Synthetic NoNs: typing results for LCDNN (inter-fraction and accuracy).",
        label="tabC:synth_typing"
    )
    with open(os.path.join(out_dir, "synth_typing.tex"), "w") as f:
        f.write(tex_type)

    return {
        "synth_summary_main": summary,
        "synth_ari": ari_tbl,
        "synth_structure_full": full_tbl,
        "synth_f1_per_target": f1_pivot,
        "synth_typing": type_tbl,
    }


# ============================================================
# 11) Optional: causalAssembly benchmark (Section 6.3 + Appendix C.5–C.8)
# ============================================================

def _causalassembly_load():
    from causalassembly import ProductionLineGraph
    import requests

    data = ProductionLineGraph.get_data()
    gt = ProductionLineGraph.get_ground_truth()  # ProductionLineGraph object
    return data, gt

def _try_get_cells(gt):
    # robust access
    if hasattr(gt, "cells"):
        return gt.cells
    if hasattr(gt, "stations"):
        return gt.stations
    # fallback: try dict
    if isinstance(gt, dict) and "cells" in gt:
        return gt["cells"]
    raise RuntimeError("Could not access stage/cell structure from causalAssembly ProductionLineGraph.")

def _to_networkx(gt):
    # robust networkx conversion
    if hasattr(gt, "to_networkx"):
        return gt.to_networkx()
    if hasattr(gt, "nx_graph"):
        return gt.nx_graph
    # fallback: if gt already is nx graph
    try:
        import networkx as nx
        if isinstance(gt, nx.DiGraph):
            return gt
    except Exception:
        pass
    raise RuntimeError("Could not convert ground truth to a NetworkX DiGraph.")

def _build_node_to_stage_map(cells) -> Dict[str, str]:
    node2stage = {}
    # cells may be dict[str, nx_graph or custom]
    for stage_name, cell in cells.items():
        # try networkx-like
        if hasattr(cell, "nodes"):
            for v in cell.nodes():
                node2stage[str(v)] = str(stage_name)
        elif isinstance(cell, dict) and "nodes" in cell:
            for v in cell["nodes"]:
                node2stage[str(v)] = str(stage_name)
        else:
            # try attribute
            if hasattr(cell, "visible_nodes") or hasattr(cell, "hidden_nodes"):
                try:
                    for v in cell.visible_nodes():
                        node2stage[str(v)] = str(stage_name)
                    for v in cell.hidden_nodes():
                        node2stage[str(v)] = str(stage_name)
                except Exception:
                    pass
    return node2stage

def _extract_stage_observed_nodes(stage_name: str, cells, data_cols: List[str]) -> List[str]:
    cell = cells[stage_name]
    # get candidate nodes from cell
    nodes = []
    if hasattr(cell, "nodes"):
        nodes = [str(v) for v in cell.nodes()]
    elif hasattr(cell, "visible_nodes"):
        nodes = [str(v) for v in cell.visible_nodes()]
    elif isinstance(cell, dict) and "nodes" in cell:
        nodes = [str(v) for v in cell["nodes"]]
    # intersect with data columns
    obs = [v for v in nodes if v in data_cols]
    return obs

def _ground_truth_within_stage_edges(nxg, obs_nodes: List[str]) -> np.ndarray:
    import networkx as nx
    p = len(obs_nodes)
    idx = {v: i for i, v in enumerate(obs_nodes)}
    A = np.zeros((p, p), dtype=int)
    for u, v in nxg.edges():
        u = str(u); v = str(v)
        if u in idx and v in idx:
            A[idx[u], idx[v]] = 1
    np.fill_diagonal(A, 0)
    return A

def _stage_local_confounded_pairs(nxg, obs_nodes: List[str], hidden_set: Set[str]) -> np.ndarray:
    import networkx as nx
    p = len(obs_nodes)
    idx = {v: i for i, v in enumerate(obs_nodes)}

    # precompute ancestors for each node
    anc = {}
    for v in obs_nodes:
        anc[v] = set(str(a) for a in nx.ancestors(nxg, v)) & hidden_set

    conf = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i + 1, p):
            vi, vj = obs_nodes[i], obs_nodes[j]
            if len(anc[vi].intersection(anc[vj])) > 0:
                conf[i, j] = conf[j, i] = 1
    return conf

def run_causalassembly_experiments(
    lcdnn_cfg: LCDNNConfig,
    alpha_pc_fci: float = 0.01,
    out_dir: str = "tables_generated"
) -> Dict[str, pd.DataFrame]:
    ensure_dir(out_dir)
    import networkx as nx

    data, gt = _causalassembly_load()
    cells = _try_get_cells(gt)
    nxg = _to_networkx(gt)

    # node-to-stage
    node2stage = _build_node_to_stage_map(cells)

    stage_names = list(cells.keys())
    # heuristically keep 5 stages in order if present
    stage_names = sorted(stage_names)[:5]

    # results collectors
    stage_size_rows = []
    structure_rows = []
    f1_stage_rows = []
    span_rows = []
    typing_rows = []

    for stage_name in tqdm(stage_names, desc="causalAssembly stages"):
        obs_nodes = _extract_stage_observed_nodes(stage_name, cells, list(data.columns))
        if len(obs_nodes) < 5:
            continue

        X = standardize(data[obs_nodes].to_numpy(dtype=float))
        p = X.shape[1]
        # Hidden set: all nodes not in this stage's observed nodes
        all_nodes = set(str(v) for v in nxg.nodes())
        obs_set = set(obs_nodes)
        hidden_set = all_nodes - obs_set

        A_true = _ground_truth_within_stage_edges(nxg, obs_nodes)
        conf_true = _stage_local_confounded_pairs(nxg, obs_nodes, hidden_set=hidden_set)

        # Methods: FCI, RFCI, RCD, LCDNN (as in paper tables)
        # FCI
        skel_fci, ori_fci, g_fci = run_fci(X, alpha=alpha_pc_fci)
        A_fci = oracle_align_from_skeleton(skel_fci, ori_fci, A_true)
        met_fci = directed_metrics(A_true, A_fci)
        structure_rows.append({"stage": stage_name, "method": "FCI", **met_fci})
        f1_stage_rows.append({"stage": stage_name, "method": "FCI", "f1": met_fci["f1"]})

        # RFCI
        skel_rfci, ori_rfci, g_rfci = run_rfci(X, alpha=alpha_pc_fci)
        A_rfci = oracle_align_from_skeleton(skel_rfci, ori_rfci, A_true)
        met_rfci = directed_metrics(A_true, A_rfci)
        structure_rows.append({"stage": stage_name, "method": "RFCI", **met_rfci})
        f1_stage_rows.append({"stage": stage_name, "method": "RFCI", "f1": met_rfci["f1"]})

        # RCD
        A_rcd, conf_rcd_pairs = run_rcd(X)
        met_rcd = directed_metrics(A_true, A_rcd)
        structure_rows.append({"stage": stage_name, "method": "RCD", **met_rcd})
        f1_stage_rows.append({"stage": stage_name, "method": "RCD", "f1": met_rcd["f1"]})

        # LCDNN
        lcdnn_out = run_lcdnn(X, lcdnn_cfg)
        A_lcdnn = lcdnn_out["A_hat"]
        met_lcdnn = directed_metrics(A_true, A_lcdnn)
        structure_rows.append({"stage": stage_name, "method": "LCDNN", **met_lcdnn})
        f1_stage_rows.append({"stage": stage_name, "method": "LCDNN", "f1": met_lcdnn["f1"]})

        # Stage sizes + estimated clusters + r_hat
        m_hat = len(lcdnn_out["clusters_hat"])
        r_hat = float(lcdnn_out["type_info"]["r_hat"])
        stage_size_rows.append({"stage": stage_name, "p": p, r"$\hat m$": m_hat, r"$\hat r$": r_hat})

        # Latent detection F1 (confounded pairs): compare unordered pairs
        def conf_pairs_from_matrix(M):
            p = M.shape[0]
            pairs = set()
            for i in range(p):
                for j in range(i + 1, p):
                    if M[i, j] == 1:
                        pairs.add((i, j))
            return pairs

        true_pairs = conf_pairs_from_matrix(conf_true)
        lcdnn_pairs = set(lcdnn_out["conf_pairs"])
        rcd_pairs = conf_rcd_pairs  # already unordered (i,j)

        def pair_f1(pred: Set[Tuple[int,int]], true: Set[Tuple[int,int]]) -> float:
            tp = len(pred & true)
            fp = len(pred - true)
            fn = len(true - pred)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        f1_lat_rcd = pair_f1(rcd_pairs, true_pairs)
        f1_lat_lcdnn = pair_f1(lcdnn_pairs, true_pairs)

        # "rho" span alignment: data-driven proxy
        # True inter-span proxy: SVD of Cov(R_true, X_out), using X_out = all other observed columns
        X_out_cols = [c for c in data.columns if c not in obs_nodes]
        X_out = standardize(data[X_out_cols].to_numpy(dtype=float))
        # residuals after regressing out *true within-stage parents* (estimated by OLS)
        # Build parent sets from A_true
        parents_true = [list(np.where(A_true[:, j] == 1)[0]) for j in range(p)]
        B_true_fit = refit_linear_sem(X, parents_true)
        R_true = compute_residuals_from_B(X, B_true_fit)

        # cross-cov (stage residuals vs out-of-stage observables)
        Sigma_ro = (R_true.T @ X_out) / max(R_true.shape[0] - 1, 1)
        U_true, s_true, _ = np.linalg.svd(Sigma_ro, full_matrices=False)
        r_true = estimate_rank_by_relative_threshold(s_true, rel_tol=0.05)
        U_true = U_true[:, :r_true] if r_true > 0 else np.zeros((p, 0))

        # estimated inter-span from LCDNN typing info (recompute U_hat)
        R_hat = compute_residuals_from_B(X, lcdnn_out["B_hat"])
        U_hat, r_hat2, _ = estimate_inter_span_from_residuals(R_hat, lcdnn_out["clusters_hat"], max_rank=10)

        # subspace alignment rho via principal angles
        def subspace_rho(Ua: np.ndarray, Ub: np.ndarray) -> float:
            if Ua.size == 0 or Ub.size == 0:
                return 0.0
            # orthonormalize
            Qa, _ = np.linalg.qr(Ua)
            Qb, _ = np.linalg.qr(Ub)
            M = Qa.T @ Qb
            s = np.linalg.svd(M, compute_uv=False)
            return float(np.mean(s ** 2))

        rho = subspace_rho(U_true, U_hat)

        span_rows.append({
            "stage": stage_name,
            "RCD F1": f1_lat_rcd,
            "LCDNN F1": f1_lat_lcdnn,
            r"$\hat r$": r_hat,
            r"$\rho$": rho
        })

        if len(true_pairs) > 0:
            # predicted labels
            pred_labels = lcdnn_out["type_labels"]
            correct_inter = 0
            false_intra = 0
            total_true = 0
            for (i, j) in true_pairs:
                total_true += 1
                pred_inter = (pred_labels.get((i, j), "intra_or_mixed") == "inter")
                if pred_inter:
                    correct_inter += 1
                else:
                    false_intra += 1
            inter_acc = correct_inter / total_true
            false_intra_rate = false_intra / total_true
        else:
            inter_acc, false_intra_rate = 0.0, 0.0

        typing_rows.append({
            "stage": stage_name,
            "Inter accuracy": inter_acc,
            "False intra": false_intra_rate
        })

    # -----------------------
    # Build and write tables
    # -----------------------
    df_stage = pd.DataFrame(stage_size_rows)
    df_stage_fmt = df_stage.copy()
    tex_stage = df_to_uai_table_tex(
        df_stage_fmt.rename(columns={"stage": "Stage", "p": r"$p$"}),
        caption="causalAssembly stage-local view: stage sizes and LCDNN estimates.",
        label="tabC:causalassembly_stage_sizes"
    )
    with open(os.path.join(out_dir, "causalassembly_stage_sizes.tex"), "w") as f:
        f.write(tex_stage)

    df_struct = pd.DataFrame(structure_rows)
    struct_tbl = df_struct.groupby("method")[["precision", "recall", "f1"]].mean().reset_index()
    order = ["FCI", "RFCI", "RCD", "LCDNN"]
    struct_tbl["method"] = pd.Categorical(struct_tbl["method"], categories=order, ordered=True)
    struct_tbl = struct_tbl.sort_values("method").reset_index(drop=True)
    struct_fmt = make_ranked_format_table(struct_tbl, ["precision", "recall", "f1"])
    tex_struct = df_to_uai_table_tex(
        struct_fmt.rename(columns={"method": "Method", "precision": "Precision", "recall": "Recall", "f1": "F1"}),
        caption="causalAssembly stage-local: within-stage directed-edge recovery (macro average).",
        label="tabC:causalassembly_structure_macro"
    )
    with open(os.path.join(out_dir, "causalassembly_structure_macro.tex"), "w") as f:
        f.write(tex_struct)

    df_f1 = pd.DataFrame(f1_stage_rows)
    f1_pivot = df_f1.pivot_table(index="method", columns="stage", values="f1", aggfunc="mean").reset_index()
    f1_pivot["Avg."] = f1_pivot[[c for c in f1_pivot.columns if c != "method"]].mean(axis=1)
    f1_fmt = f1_pivot.copy()
    for c in f1_fmt.columns:
        if c != "method":
            f1_fmt[c] = f1_fmt[c].apply(lambda x: fmt_num(float(x), 2))
    tex_f1 = df_to_uai_table_tex(
        f1_fmt.rename(columns={"method": "Method"}),
        caption="causalAssembly stage-local: directed-edge F1 per stage.",
        label="tabC:causalassembly_f1_stagewise"
    )
    with open(os.path.join(out_dir, "causalassembly_f1_stagewise.tex"), "w") as f:
        f.write(tex_f1)

    df_span = pd.DataFrame(span_rows)
    span_tbl = df_span.copy()
    for c in ["RCD F1", "LCDNN F1", r"$\hat r$", r"$\rho$"]:
        span_tbl[c] = span_tbl[c].apply(lambda x: fmt_num(float(x), 2))
    tex_span = df_to_uai_table_tex(
        span_tbl.rename(columns={"stage": "Stage"}),
        caption="causalAssembly stage-local: latent detection and spillover-span alignment.",
        label="tabC:causalassembly_span"
    )
    with open(os.path.join(out_dir, "causalassembly_span.tex"), "w") as f:
        f.write(tex_span)

    df_typ = pd.DataFrame(typing_rows)
    typ_tbl = df_typ.copy()
    for c in ["Inter accuracy", "False intra"]:
        typ_tbl[c] = typ_tbl[c].apply(lambda x: fmt_num(float(x), 2))
    tex_typ = df_to_uai_table_tex(
        typ_tbl.rename(columns={"stage": "Stage"}),
        caption="causalAssembly stage-local: typing performance for LCDNN.",
        label="tabC:causalassembly_typing"
    )
    with open(os.path.join(out_dir, "causalassembly_typing.tex"), "w") as f:
        f.write(tex_typ)

    # main summary table in Section 6.3 (macro average)
    main_tbl = struct_tbl.copy()
    main_tbl_fmt = make_ranked_format_table(main_tbl, ["precision", "recall", "f1"])
    tex_main = df_to_uai_table_tex(
        main_tbl_fmt.rename(columns={"method": "Method", "precision": "Prec.", "recall": "Rec.", "f1": "F1"}),
        caption="causalAssembly stage-local: within-stage directed-edge recovery (macro average).",
        label="tab:causalassembly_summary_main"
    )
    with open(os.path.join(out_dir, "causalassembly_summary_main.tex"), "w") as f:
        f.write(tex_main)

    return {
        "causalassembly_stage_sizes": df_stage,
        "causalassembly_structure_macro": struct_tbl,
        "causalassembly_f1_stagewise": f1_pivot,
        "causalassembly_span": df_span,
        "causalassembly_typing": df_typ,
    }


# ============================================================
# 12) Run everything and write tables
# ============================================================
RUN_SYNTHETIC = False
RUN_CAUSALASSEMBLY = False   # set False if you only want simulations

OUT_DIR = "tables_generated"
ensure_dir(OUT_DIR)

set_all_seeds(0)

synth_cfg = SynthConfig(
    p=30,
    m=6,
    r_int=3,
    n=1000,
    edge_prob=0.6,
    interface_size=1,
    scales=(0.6, 0.9, 1.3, 0.7, 1.0),
    seed0=2026
)

lcdnn_cfg = LCDNNConfig(
    alpha_gin=0.05,
    dlgin_rank_method="noise",
    dlgin_noise_k=1.5,
    dlgin_hsic_method="perm",
    dlgin_hsic_n_perm=200,
    stdgin_rank_method="relative",
    stdgin_rel_tol=0.01,
    stdgin_hsic_method="perm",
    stdgin_hsic_n_perm=200,
    rcd_max_explanatory_num=6,
    rcd_cor_alpha=0.05,
    rcd_ind_alpha=0.05,
    rcd_shapiro_alpha=0.05,
    ca_do_prune=False,
    alpha_type=0.05,
    typing_hsic_method="perm",
    typing_hsic_n_perm=200,
    max_rank=10,
    seed=2026,
)

all_tables = {}

if RUN_SYNTHETIC:
    synth_tables = run_synthetic_experiments(
        synth_cfg=synth_cfg,
        lcdnn_cfg=lcdnn_cfg,
        n_trials=20,
        alpha_pc_fci=0.01,
        out_dir=OUT_DIR,
        run_grandag=True
    )
    all_tables.update(synth_tables)

if RUN_CAUSALASSEMBLY:
    ca_tables = run_causalassembly_experiments(
        lcdnn_cfg=lcdnn_cfg,
        alpha_pc_fci=0.01,
        out_dir=OUT_DIR
    )
    all_tables.update(ca_tables)

print(f"\nDone. LaTeX tables written to: ./{OUT_DIR}/\n")
print("Generated files (key ones):")
for fn in sorted(os.listdir(OUT_DIR)):
    if fn.endswith(".tex"):
        print(" -", fn)
# ============================================================
# 12) Diagnostics + simple hyperparameter tuning helpers
# ============================================================
def _rank_of_crosscov(X: np.ndarray, C: Set[int], tol: float = 1e-6) -> int:
    C = sorted(list(C))
    R = [i for i in range(X.shape[1]) if i not in C]
    if len(C) == 0 or len(R) == 0:
        return 0
    Sigma = np.cov(X, rowvar=False, ddof=1)
    Sigma_cr = Sigma[np.ix_(C, R)]
    s = np.linalg.svd(Sigma_cr, compute_uv=False)
    if s.size == 0:
        return 0
    thresh = max(tol, tol * float(s.max()))
    return int(np.sum(s > thresh))



def _clusters_to_labels(clusters: Iterable[Iterable[int]], p: int) -> np.ndarray:
    lab = np.full(int(p), -1, dtype=int)
    for ci, C in enumerate(clusters):
        for v in C:
            lab[int(v)] = int(ci)
    if np.any(lab < 0):
        missing = np.where(lab < 0)[0]
        raise ValueError(f"_clusters_to_labels: {missing.size} unassigned indices (first few: {missing[:10].tolist()})")
    return lab

def diagnose_lcdnn_one_run(
    synth_cfg: SynthConfig,
    lcdnn_cfg: LCDNNConfig,
    *,
    trial_seed: int = 0,
    target_scale: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    base = generate_base_sem(synth_cfg, seed=synth_cfg.seed0 + 1000 * trial_seed)
    clusters_true = base["clusters"]
    cl_true = _clusters_to_labels(clusters_true, synth_cfg.p)

    # Choose scale (default: strongest spillover)
    if target_scale is None:
        target_scale = float(max(synth_cfg.scales))

    sim = simulate_from_sem(
        base, synth_cfg,
        scale=float(target_scale),
        seed=synth_cfg.seed0 + 1000 * trial_seed + 77
    )
    X = sim["X"]
    A_true = sim["A_true"]

    # -------- Phase I clustering
    clusters_hat = agglomerative_gin_clustering(
        X,
        alpha_gin=lcdnn_cfg.alpha_gin,
        rank_method=lcdnn_cfg.dlgin_rank_method,
        rel_tol=lcdnn_cfg.dlgin_rel_tol,
        noise_k=lcdnn_cfg.dlgin_noise_k,
        hsic_method=lcdnn_cfg.dlgin_hsic_method,
        hsic_n_perm=lcdnn_cfg.dlgin_hsic_n_perm,
        hsic_max_sigma_points=lcdnn_cfg.dlgin_hsic_max_sigma_points,
        seed=lcdnn_cfg.seed,
    )
    ari = eval_ari(cl_true, clusters_hat)

    # -------- Phase II: RCD baseline (no cluster constraints)
    try:
        A_rcd, _ = run_rcd(X)
        met_rcd = directed_metrics(A_true, A_rcd)
    except Exception as e:
        met_rcd = {"precision": np.nan, "recall": np.nan, "f1": np.nan}
        if verbose:
            print("[diagnose] RCD baseline failed:", repr(e))

    # -------- LCDNN end-to-end (estimated clusters)
    out_lcdnn = run_lcdnn(X, lcdnn_cfg)
    met_lcdnn = directed_metrics(A_true, out_lcdnn["A_hat"])

    # -------- Phase II upper bound: CA-RCD with ORACLE clusters
    try:
        B_oracle, _ = ca_rcd(
            X,
            clusters_true,
            rcd_max_explanatory_num=lcdnn_cfg.rcd_max_explanatory_num,
            rcd_cor_alpha=lcdnn_cfg.rcd_cor_alpha,
            rcd_ind_alpha=lcdnn_cfg.rcd_ind_alpha,
            rcd_shapiro_alpha=lcdnn_cfg.rcd_shapiro_alpha,
            rcd_MLHSICR=lcdnn_cfg.rcd_MLHSICR,
            rcd_bw_method=lcdnn_cfg.rcd_bw_method,
            rcd_independence=lcdnn_cfg.rcd_independence,
            w_threshold=lcdnn_cfg.ca_w_threshold,
            do_prune=lcdnn_cfg.ca_do_prune,
            prune_alpha=lcdnn_cfg.ca_prune_alpha,
            prune_hsic_method=lcdnn_cfg.ca_prune_hsic_method,
            prune_hsic_n_perm=lcdnn_cfg.ca_prune_hsic_n_perm,
            prune_hsic_max_sigma_points=lcdnn_cfg.ca_prune_hsic_max_sigma_points,
            prune_max_rounds=lcdnn_cfg.ca_prune_max_rounds,
            seed=lcdnn_cfg.seed,
        )
        A_oracle = (np.abs(B_oracle) > lcdnn_cfg.edge_threshold).astype(int)
        np.fill_diagonal(A_oracle, 0)
        met_oracle = directed_metrics(A_true, A_oracle)
    except Exception as e:
        met_oracle = {"precision": np.nan, "recall": np.nan, "f1": np.nan}
        if verbose:
            print("[diagnose] Oracle CA-RCD failed:", repr(e))

    # -------- Interface-bottleneck check on TRUE clusters
    bottleneck = []
    for C in clusters_true:
        rk = _rank_of_crosscov(X, C, tol=1e-6)
        bottleneck.append({"cluster_size": len(C), "rank_crosscov": rk})

    if verbose:
        sizes_hat = sorted([len(C) for C in clusters_hat])
        sizes_true = sorted([len(C) for C in clusters_true])
        print(f"Target scale = {target_scale:g}")
        print(f"True #clusters: {len(clusters_true)} sizes: {sizes_true}")
        print(f"Estimated #clusters: {len(clusters_hat)} sizes: {sizes_hat}")
        print(f"ARI: {ari:.3f}")
        print("Directed metrics (precision/recall/f1):")
        print("  RCD baseline:", met_rcd)
        print("  LCDNN (est clusters):", met_lcdnn)
        print("  Oracle clusters (upper bound):", met_oracle)
        print("Interface-bottleneck check (need rank < cluster_size):")
        for b in bottleneck:
            ok = b["rank_crosscov"] < b["cluster_size"]
            print(f"  size={b['cluster_size']:2d} rank={b['rank_crosscov']:2d}  {'OK' if ok else 'VIOLATION'}")

    return {
        "ari": ari,
        "met_rcd": met_rcd,
        "met_lcdnn": met_lcdnn,
        "met_oracle": met_oracle,
        "bottleneck": bottleneck,
        "clusters_true": clusters_true,
        "clusters_hat": clusters_hat,
    }


def tune_lcdnn_grid(
    synth_cfg: SynthConfig,
    *,
    n_trials: int = 5,
    target_scale: Optional[float] = None,
    grid: Optional[Dict[str, List[Any]]] = None,
    seed_offset: int = 0,
) -> pd.DataFrame:

    if target_scale is None:
        target_scale = float(max(synth_cfg.scales))

    if grid is None:
        grid = {
            "alpha_gin": [0.05, 0.1, 0.2],
            "dlgin_noise_k": [1.0, 1.5, 2.0],
            "dlgin_hsic_n_perm": [200, 500],
            "rcd_max_explanatory_num": [2, 3, 4],
            "rcd_ind_alpha": [0.01, 0.05],
            "rcd_cor_alpha": [0.01, 0.05],
            "ca_w_threshold": [1e-6, 0.05, 0.1],
            "edge_threshold": [1e-6, 0.05, 0.1],
        }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    rows = []
    for combo in tqdm(combos, desc="Tuning grid"):
        params = dict(zip(keys, combo))
        cfg = LCDNNConfig(**params)

        f1s = []
        aris = []
        for t in range(n_trials):
            base = generate_base_sem(synth_cfg, seed=synth_cfg.seed0 + seed_offset + 1000 * t)
            clusters_true = base["clusters"]
            cl_true = _clusters_to_labels(clusters_true, synth_cfg.p)

            sim = simulate_from_sem(
                base, synth_cfg,
                scale=float(target_scale),
                seed=synth_cfg.seed0 + seed_offset + 1000 * t + 77
            )
            X = sim["X"]
            A_true = sim["A_true"]

            out = run_lcdnn(X, cfg)
            f1s.append(directed_metrics(A_true, out["A_hat"])["f1"])
            aris.append(eval_ari(cl_true, out["clusters_hat"]))

        rows.append({**params, "mean_f1": float(np.mean(f1s)), "mean_ari": float(np.mean(aris))})

    df = pd.DataFrame(rows).sort_values(["mean_f1", "mean_ari"], ascending=False).reset_index(drop=True)
    return df

# ============================================================
# Paper-aligned generator + diagnostics + fast verification run
# ============================================================

from dataclasses import dataclass, asdict
from typing import List, Set, Dict, Tuple, Optional, Iterable
import math, time, json, hashlib
from pathlib import Path
from tqdm.auto import tqdm
from IPython.display import display

# ----------------------------
# 0) Patch: enforce paper's max merge union size s_max=12 in Phase I
#    (This is a *hyperparameter in your paper*, not an algorithm change.)
# ----------------------------
def agglomerative_gin_clustering(
    X: np.ndarray,
    *,
    alpha_gin: float,
    rank_method: str = "noise",
    rel_tol: float = 0.3,
    noise_k: float = 1.5,
    hsic_method: str = "perm",
    hsic_n_perm: int = 200,
    hsic_max_sigma_points: int = 200,
    max_merges: Optional[int] = None,
    max_pair_checks: Optional[int] = None,
    max_union_size: Optional[int] = 12,   # <- paper: s_max = 12
    seed: int = 0,
) -> List[Set[int]]:
    """Greedy agglomerative clustering based on DL-GIN, with optional max union size."""
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    clusters: List[Set[int]] = [{i} for i in range(p)]
    merges = 0

    while True:
        merged = False

        clusters_sorted = sorted(clusters, key=lambda s: (len(s), min(s)))

        pairs = []
        for a in range(len(clusters_sorted)):
            for b in range(a + 1, len(clusters_sorted)):
                Ca = clusters_sorted[a]
                Cb = clusters_sorted[b]
                pairs.append((len(Ca) + len(Cb), min(Ca), min(Cb), Ca, Cb))
        pairs.sort(key=lambda t: (t[0], t[1], t[2]))

        checks = 0
        for _, _, _, Ca, Cb in pairs:
            C = Ca | Cb
            if len(C) == p:
                continue  # never merge into a single cluster (R would be empty)
            if (max_union_size is not None) and (len(C) > int(max_union_size)):
                continue

            pval = dl_gin_pvalue(
                X,
                C,
                alpha_hsic=alpha_gin,
                rank_method=rank_method,
                rel_tol=rel_tol,
                noise_k=noise_k,
                hsic_method=hsic_method,
                hsic_n_perm=hsic_n_perm,
                hsic_max_sigma_points=hsic_max_sigma_points,
                seed=seed + merges * 10000 + checks,
            )
            checks += 1
            if pval > alpha_gin:
                new_clusters = [S for S in clusters if S != Ca and S != Cb]
                new_clusters.append(C)
                clusters = new_clusters
                merged = True
                merges += 1
                break

            if (max_pair_checks is not None) and (checks >= max_pair_checks):
                break

        if not merged:
            break
        if (max_merges is not None) and (merges >= max_merges):
            break

    return [set(sorted(list(c))) for c in clusters]

# ----------------------------
# 1) Paper-aligned synthetic generator (from your TeX)
# ----------------------------
@dataclass(frozen=True)
class PaperSynthConfig:
    T: int = 5
    p: int = 48
    m: int = 8
    k_int: int = 2
    r: int = 2
    n: int = 2000
    edge_prob_cluster: float = 0.25
    scales: Tuple[float, float, float, float, float] = (0.20, 0.50, 1.00, 0.50, 0.20)
    # coefficient & loading ranges (paper)
    w_low: float = 0.3
    w_high: float = 0.7
    lam_low: float = 0.8
    lam_high: float = 1.2
    noise_dist: str = "laplace"   # paper: Laplace(0,1)

def _sign_uniform(rng: np.random.Generator, low: float, high: float, size):
    mag = rng.uniform(low, high, size=size)
    sgn = rng.choice([-1.0, 1.0], size=size)
    return sgn * mag

def _draw_noise(rng: np.random.Generator, shape, dist: str):
    dist = dist.lower()
    if dist == "laplace":
        return rng.laplace(loc=0.0, scale=1.0, size=shape)
    elif dist == "gaussian" or dist == "normal":
        return rng.normal(loc=0.0, scale=1.0, size=shape)
    else:
        raise ValueError(f"Unknown noise_dist: {dist}")

def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / sd

def generate_paper_target(cfg: PaperSynthConfig, *, seed: int, scale: float) -> Dict:
    """Generate one target dataset N_j with spillover scale `scale` (paper Algorithm)."""
    rng = np.random.default_rng(seed)
    p, m, k_int, r, n = cfg.p, cfg.m, cfg.k_int, cfg.r, cfg.n

    # --- partition into equal-size clusters
    assert p % m == 0, "paper uses equal-size clusters"
    cl_size = p // m
    perm = rng.permutation(p)
    clusters = [perm[i*cl_size:(i+1)*cl_size].tolist() for i in range(m)]
    clusters_sets = [set(c) for c in clusters]

    cl_id = np.empty(p, dtype=int)
    for ci, C in enumerate(clusters_sets):
        for v in C:
            cl_id[v] = ci

    # --- interface sets per cluster
    interface_sets: List[Set[int]] = []
    interface_all: Set[int] = set()
    for C in clusters_sets:
        I = set(rng.choice(list(C), size=min(k_int, len(C)), replace=False).tolist())
        interface_sets.append(I)
        interface_all |= I

    # --- local loadings Lambda^{loc}: p x q, q=m (one latent per cluster)
    q = m
    Lambda = np.zeros((p, q), dtype=float)
    for ci, C in enumerate(clusters_sets):
        lam_vec = _sign_uniform(rng, cfg.lam_low, cfg.lam_high, size=len(C))
        for kk, v in enumerate(sorted(list(C))):
            Lambda[v, ci] = lam_vec[kk]

    # --- directed structure across clusters: random topo order + Bernoulli edges
    topo = rng.permutation(m).tolist()
    topo_pos = {c: i for i, c in enumerate(topo)}
    B = np.zeros((p, p), dtype=float)  # our convention: B[u,v] = u -> v coefficient

    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            if topo_pos[a] >= topo_pos[b]:
                continue  # respect topo order
            if rng.random() >= cfg.edge_prob_cluster:
                continue
            # place ONE variable-level edge u in C_a -> v in I_b
            u = int(rng.choice(clusters[a], size=1)[0])
            v = int(rng.choice(list(interface_sets[b]), size=1)[0])
            B[u, v] = float(_sign_uniform(rng, cfg.w_low, cfg.w_high, size=1)[0])

    # --- (optional) rescale weights (paper says rho(B) <= 0.8)
    # For a DAG this is typically already satisfied (eigs are ~0), but we keep the step for faithfulness.
    try:
        eigs = np.linalg.eigvals(B)
        rho = float(np.max(np.abs(eigs)))
    except Exception:
        rho = 0.0
    if rho > 0.8 + 1e-12:
        B = B * (0.8 / rho)

    A_true = (np.abs(B) > 0).astype(int)
    np.fill_diagonal(A_true, 0)

    # --- inter-network loadings Gamma^{int}: p x r, only interface rows are nonzero, then scaled by s_j
    Gamma = np.zeros((p, r), dtype=float)
    for v in interface_all:
        Gamma[v, :] = rng.normal(loc=0.0, scale=1.0, size=r)
    Gamma = Gamma * float(scale)

    # --- draw samples
    L_loc = _draw_noise(rng, (n, q), cfg.noise_dist)
    L_int = _draw_noise(rng, (n, r), cfg.noise_dist)
    E = _draw_noise(rng, (n, p), cfg.noise_dist)

    # model in your paper: X = B^T X + Lambda L_loc + Gamma L_int + e
    # row-wise simulation: X = (L_loc Lambda^T + L_int Gamma^T + E) (I - B)^{-1}
    A = np.linalg.inv(np.eye(p) - B)  # (I - B)^{-1}
    S = L_loc @ Lambda.T + L_int @ Gamma.T + E
    X = S @ A
    X = _standardize(X)

    # --- confounding metadata (optional debug)
    interface_mask = (np.abs(Gamma).sum(axis=1) > 1e-12).astype(int)
    inter_conf = np.outer(interface_mask, interface_mask).astype(int)
    np.fill_diagonal(inter_conf, 0)
    local_conf = (cl_id.reshape(-1, 1) == cl_id.reshape(1, -1)).astype(int)
    np.fill_diagonal(local_conf, 0)
    conf_any = ((inter_conf + local_conf) > 0).astype(int)

    return dict(
        X=X,
        B=B,
        A_true=A_true,
        clusters_true=clusters_sets,
        cl_id_true=cl_id,
        interface_sets=interface_sets,
        interface_mask=interface_mask,
        conf_any=conf_any,
        Gamma=Gamma,
        scale=float(scale),
        seed=int(seed),
        rho=float(rho),
        n_edges=int(A_true.sum()),
    )

def generate_paper_instance(cfg: PaperSynthConfig, *, seed: int) -> List[Dict]:
    """Generate one Monte Carlo instance: targets N1..N5."""
    out = []
    for j in range(cfg.T):
        out.append(generate_paper_target(cfg, seed=seed + 1000*j, scale=cfg.scales[j]))
    return out

# ----------------------------
# 2) Metric sanity checks
# ----------------------------
def _metric_unit_tests():
    rng = np.random.default_rng(0)
    p = 10
    A = (rng.random((p,p)) < 0.1).astype(int)
    np.fill_diagonal(A, 0)

    same = directed_metrics(A, A)
    assert abs(same["precision"] - 1.0) < 1e-9
    assert abs(same["recall"] - 1.0) < 1e-9
    assert abs(same["f1"] - 1.0) < 1e-9

    # Transpose should usually break directed edges (unless symmetric by chance)
    tr = directed_metrics(A, A.T)
    if A.sum() > 0 and (A == A.T).all() is False:
        # not a strict assert (randomly could be symmetric), but warn if suspicious
        if tr["f1"] > 0.5:
            print("[WARN] metric unit test: A and A.T are unusually similar; ignoring.")
    print("Metric unit tests: OK")

_metric_unit_tests()

# ----------------------------
# 3) Debug helpers
# ----------------------------
def overlap_counts(A_true: np.ndarray, A_hat: np.ndarray) -> Dict[str,int]:
    A_true = (A_true > 0).astype(int)
    A_hat = (A_hat > 0).astype(int)
    tp = int(np.sum((A_true==1) & (A_hat==1)))
    fp = int(np.sum((A_true==0) & (A_hat==1)))
    fn = int(np.sum((A_true==1) & (A_hat==0)))
    return {"tp": tp, "fp": fp, "fn": fn, "true": int(A_true.sum()), "pred": int(A_hat.sum())}

def transpose_diagnostic(A_true: np.ndarray, A_hat: np.ndarray) -> Dict[str, Dict[str, float]]:
    m0 = directed_metrics(A_true, A_hat)
    m1 = directed_metrics(A_true, A_hat.T)
    return {"as_is": m0, "transposed": m1}

def frac_true_edges_inter_confounded(target: Dict) -> float:
    A_true = target["A_true"]
    interface = target["interface_mask"].astype(bool)
    idx = np.argwhere(A_true == 1)
    if idx.shape[0] == 0:
        return 0.0
    cnt = 0
    for u,v in idx:
        if interface[u] and interface[v]:
            cnt += 1
    return cnt / idx.shape[0]

# ----------------------------
# 4) Fast verification runner (LCDNN only)
# ----------------------------
CACHE_DIR = "cache_lcdnn_paper_debug"   # set None to disable caching
USE_CACHE = True

MODE = "FAST"     # "FAST" or "PAPER"
TARGETS_TO_RUN = (2,)   # default: only N3 (0-indexed), strongest spillover
N_TRIALS = 3

set_all_seeds(0)

paper_cfg = PaperSynthConfig()

if MODE.upper() == "FAST":
    # Much faster, still paper-like enough to check that scores are not collapsing.
    # (If scores look good, switch MODE="PAPER".)
    paper_cfg = PaperSynthConfig(n=1000)  # smaller n
    hsic_perm = 200
else:
    hsic_perm = 500

# LCDNN hyperparameters (from Table "LCDNN hyperparameters used in all experiments")
# - alpha_merge = 0.10
# - alpha_test  = 0.01 (Phases II-III)
# - kappa = 1.2 for nullspace threshold
# - HSIC permutations B=500 (PAPER) / lower for FAST
lcdnn_cfg = LCDNNConfig(
    # Phase I
    alpha_gin=0.10,
    dlgin_rank_method="noise",
    dlgin_noise_k=1.2,
    dlgin_hsic_method="perm",
    dlgin_hsic_n_perm=hsic_perm,

    # Phase II (RCD tests)
    rcd_cor_alpha=0.01,
    rcd_ind_alpha=0.01,
    rcd_shapiro_alpha=0.01,
    rcd_max_explanatory_num=min(5, int(math.floor(math.log(paper_cfg.n)))),  # s_cond = min{5, floor(log n)}
    ca_w_threshold=0.05,
    ca_do_prune=False,

    # Phase III (typing)
    alpha_type=0.01,
    typing_hsic_method="perm",
    typing_hsic_n_perm=hsic_perm,
    max_rank=10,

    # Thresholding
    edge_threshold=0.05,

    seed=0,
)

def _stable_hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(s).hexdigest()

def _save_clusters(path: Path, clusters: List[Set[int]]) -> None:
    payload = [sorted(list(c)) for c in clusters]
    path.write_text(json.dumps(payload))

def _load_clusters(path: Path) -> List[Set[int]]:
    payload = json.loads(path.read_text())
    return [set(map(int, c)) for c in payload]

def run_trials(cfg: PaperSynthConfig, lcdnn_cfg: LCDNNConfig, n_trials: int, targets_to_run: Iterable[int]) -> pd.DataFrame:
    cache = Path(CACHE_DIR) if (CACHE_DIR is not None) else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "sims").mkdir(exist_ok=True)
        (cache / "outs").mkdir(exist_ok=True)

    rows = []
    for trial in range(n_trials):
        # ---- generate / load sim
        sim_key = _stable_hash({"cfg": asdict(cfg), "trial": trial})
        sim_path = (cache / "sims" / f"{sim_key}.npz") if cache is not None else None
        sim_meta = (cache / "sims" / f"{sim_key}.json") if cache is not None else None

        if USE_CACHE and (sim_path is not None) and sim_path.exists() and sim_meta.exists():
            with np.load(sim_path, allow_pickle=True) as z:
                # stored as object arrays of dicts
                targets = z["targets"].tolist()
            # json meta exists for easier inspection; we don't need to parse it
        else:
            targets = generate_paper_instance(cfg, seed=cfg.T * 10000 + trial * 123)
            if sim_path is not None:
                np.savez_compressed(sim_path, targets=np.array(targets, dtype=object))
                sim_meta.write_text(json.dumps({"cfg": asdict(cfg), "trial": trial, "targets": len(targets)}))

        # ---- run lcdnn per selected target
        for j in targets_to_run:
            tgt = targets[int(j)]
            X = tgt["X"]
            A_true = tgt["A_true"]
            cl_true = tgt["cl_id_true"]

            out_key = _stable_hash({"cfg": asdict(cfg), "lcdnn_cfg": asdict(lcdnn_cfg), "trial": trial, "target": int(j)})
            out_npz = (cache / "outs" / f"{out_key}.npz") if cache is not None else None
            out_clusters = (cache / "outs" / f"{out_key}_clusters.json") if cache is not None else None

            loaded = bool(USE_CACHE and out_npz is not None and out_clusters is not None and out_npz.exists() and out_clusters.exists())

            if loaded:
                with np.load(out_npz) as z:
                    A_hat = z["A_hat"]
                    elapsed = float(z["elapsed_sec"][0])
                clusters_hat = _load_clusters(out_clusters)
            else:
                t0 = time.perf_counter()
                out = run_lcdnn(X, lcdnn_cfg)
                elapsed = time.perf_counter() - t0
                A_hat = out["A_hat"]
                clusters_hat = out["clusters_hat"]

                if out_npz is not None:
                    np.savez_compressed(out_npz, A_hat=A_hat, elapsed_sec=np.array([elapsed], dtype=float))
                    _save_clusters(out_clusters, clusters_hat)

            dm = directed_metrics(A_true, A_hat)
            ari = eval_ari(cl_true, clusters_hat)
            diag = transpose_diagnostic(A_true, A_hat)
            ov = overlap_counts(A_true, A_hat)

            rows.append({
                "trial": int(trial),
                "target": int(j) + 1,  # 1..5
                "scale": float(tgt["scale"]),
                "true_edges": int(tgt["n_edges"]),
                "pred_edges": int(ov["pred"]),
                "tp": int(ov["tp"]),
                "fp": int(ov["fp"]),
                "fn": int(ov["fn"]),
                "precision": float(dm["precision"]),
                "recall": float(dm["recall"]),
                "f1": float(dm["f1"]),
                "ari": float(ari),
                "f1_if_transposed": float(diag["transposed"]["f1"]),
                "elapsed_sec": float(elapsed),
                "inter_confounded_edge_frac": float(frac_true_edges_inter_confounded(tgt)),
                "loaded_from_cache": loaded,
            })

    df = pd.DataFrame(rows)

    # macro averages across selected targets (and trials)
    summary = df[["precision","recall","f1","ari","elapsed_sec"]].mean().to_frame("mean").T
    display(df)
    display(summary)
    return df

df = run_trials(paper_cfg, lcdnn_cfg, n_trials=N_TRIALS, targets_to_run=TARGETS_TO_RUN)

# ----------------------------
# 5) Interpretation hints (printed, not asserted)
# ----------------------------
bad = df[df["precision"] <= 1e-9]
if len(bad) > 0:
    print("\n[DIAG] Some runs have precision ~0.")
    print("Common causes:")
    print("  (1) A_true uses the wrong convention (e.g., sign-only instead of abs, or transposed).")
    print("  (2) lingam adjacency convention mismatch (check f1_if_transposed; if it is much higher, fix transpose).")
    print("  (3) very few true edges in that trial (true_edges small) — single-trial metrics can be 0 by chance.")
    print("\nLook at columns: true_edges, pred_edges, tp/fp/fn, f1_if_transposed, ari.")

hi_transpose = df[df["f1_if_transposed"] > df["f1"] + 0.2]
if len(hi_transpose) > 0:
    print("\n[DIAG] A_hat.T matches A_true much better than A_hat in some runs.")
    print("This strongly suggests a from/to convention mismatch in how B_hat is parsed from lingam.RCD().")
    print("Fix: in parse_lingam_rcd_output(), swap the conversion (use adj[from,to] instead of adj[to,from]) or transpose B_hat before thresholding.")
