# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Chernoff-optimized per-checkpoint ladder bounds.

Hoeffding uses the range-only MGF bound log M(lambda) - lambda*mu <= lambda^2/8.
For Bernoulli losses (classification 0/1 error) we can optimize lambda in the
MGF exactly, recovering the binomial KL-inversion bound:

    n * KL(p_hat + eps_plus || p_hat) = ln(N_j / delta_j)
    n * KL(p_hat - eps_minus || p_hat) = ln(N_j / delta_j)

where N_j = C(T_max - 1, j - 1) is the number of realizable padded transcripts
with exactly j improvements (Theorem 1, stratified form) and delta_j = delta/K_max
under uniform allocation.

The single-sided form is used with Markov / Chernoff; the two-sided bound carries
an additional factor of 2 in the log term.
"""

import math


def _kl_bern(p: float, q: float) -> float:
    """KL(Bern(p) || Bern(q)) with clamping for numerical stability."""
    eps = 1e-15
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def kl_inv_upper(p_hat: float, n: int, log_term: float, tol: float = 1e-10) -> float:
    """Largest eps such that n * KL(p_hat + eps || p_hat) <= log_term.

    Returns eps_plus in [0, 1 - p_hat]. Uses bisection because KL is convex and
    strictly increasing in its first argument for q > p_hat.
    """
    if p_hat >= 1.0:
        return 0.0
    target = log_term / n
    lo, hi = 0.0, 1.0 - p_hat
    # KL(1 || p_hat) is finite as long as p_hat > 0; if target exceeds it, the
    # inequality is vacuous and we cap at 1 - p_hat.
    if _kl_bern(1.0, p_hat) <= target:
        return hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _kl_bern(p_hat + mid, p_hat) > target:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def kl_inv_lower(p_hat: float, n: int, log_term: float, tol: float = 1e-10) -> float:
    """Largest eps such that n * KL(p_hat - eps || p_hat) <= log_term.

    Returns eps_minus in [0, p_hat].
    """
    if p_hat <= 0.0:
        return 0.0
    target = log_term / n
    lo, hi = 0.0, p_hat
    if _kl_bern(0.0, p_hat) <= target:
        return hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _kl_bern(p_hat - mid, p_hat) > target:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def per_checkpoint_ci_kl(
    j: int,
    p_hat: float,
    n_val: int,
    K_max: int = 7,
    T_max: int = 50,
    delta: float = 0.05,
) -> tuple[float, float]:
    """Per-checkpoint two-sided KL CI at improvement index j.

    Returns (eps_minus, eps_plus) such that with probability >= 1 - delta
    (simultaneously over all checkpoints j = 1..K_max under uniform allocation):

        R(h_j) in [p_hat - eps_minus, p_hat + eps_plus].

    Stratified union bound: N_j = C(T_max - 1, j - 1); delta_j = delta / K_max.
    The two-sided log term is ln(2 N_j / delta_j).
    """
    N_j = math.comb(T_max - 1, j - 1)
    delta_j = delta / K_max
    log_term = math.log(2 * N_j / delta_j)
    eps_plus = kl_inv_upper(p_hat, n_val, log_term)
    eps_minus = kl_inv_lower(p_hat, n_val, log_term)
    return eps_minus, eps_plus


def per_checkpoint_ci_hoeffding(
    j: int,
    n_val: int,
    K_max: int = 7,
    T_max: int = 50,
    delta: float = 0.05,
) -> float:
    """Per-checkpoint symmetric Hoeffding CI at improvement index j."""
    N_j = math.comb(T_max - 1, j - 1)
    delta_j = delta / K_max
    return math.sqrt(math.log(2 * N_j / delta_j) / (2 * n_val))


if __name__ == "__main__":
    print("Per-checkpoint CI comparison (K_max=7, T_max=50, delta=0.05)")
    print(f"{'j':>3} {'N_j':>12} {'p_hat':>6} {'n':>6} "
          f"{'Hoeff (pp)':>12} {'KL- (pp)':>10} {'KL+ (pp)':>10} {'ratio':>6}")
    for n_val, p_hat in [(5000, 0.90), (5000, 0.50), (20000, 0.80), (50000, 0.82)]:
        for j in range(1, 8):
            N_j = math.comb(49, j - 1)
            h = per_checkpoint_ci_hoeffding(j, n_val)
            km, kp = per_checkpoint_ci_kl(j, p_hat, n_val)
            kmax = max(km, kp)
            print(f"{j:>3} {N_j:>12,} {p_hat:>6.2f} {n_val:>6} "
                  f"{h*100:>12.3f} {km*100:>10.3f} {kp*100:>10.3f} "
                  f"{h/kmax:>6.2f}")
        print()
