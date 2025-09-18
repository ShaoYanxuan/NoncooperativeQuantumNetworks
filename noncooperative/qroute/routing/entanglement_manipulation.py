from __future__ import annotations
from collections import deque
from math import ceil, sqrt, isclose
from functools import lru_cache, partial
import math
from multiprocessing import Pool
from os import path
from pathlib import Path
from random import random
from typing import Final

from numpy import load, mean, savez_compressed
from scipy.optimize import brentq
from scipy.stats import entropy
from filelock import FileLock

# ──────────────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────────────
BBPSSW_PROB_THRESHOLD: Final[float] = 1e-9  # drop paths with vanishing success
MAX_PURIFICATION_LEVEL: Final[int] = 50  # hard cap on BBPSSW rounds
DILUTION_PRECOMPUTE_LEVEL: Final[float] = 2  # precompute up to 200% dilution

# Create separate cache subdirectories
CACHE_DIR = Path(path.dirname(path.abspath(__file__))) / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# global cache dict
_precomputed_table_cache = {}


# ──────────────────────────────────────────────────────────────────────────────
#  Elementary conversions
# ──────────────────────────────────────────────────────────────────────────────


def fidelity_from_werner_param(p):
    return (3.0 * p + 1.0) / 4.0


def werner_param_from_fidelity(f):
    return (4.0 * f - 1.0) / 3.0


def fidelity_from_eigenvalue(lam):
    return sqrt(lam * (1.0 - lam)) + 0.5


def werner_param_from_eigenvalue(lam):
    return (4.0 * sqrt(lam * (1.0 - lam)) + 1.0) / 3.0


# ──────────────────────────────────────────────────────────────────────────────
#  BBPSSW primitives
# ──────────────────────────────────────────────────────────────────────────────


def _bbpssw_step(p1: float, p2: float) -> float:
    """Werner parameter after one BBPSSW two‑to‑one step."""
    return (p1 + p2 + 4.0 * p1 * p2) / (3.0 + 3.0 * p1 * p2)


# ──────────────────────────────────────────────────────────────────────────────
#  Numerical purification (M ≤ N)
# ──────────────────────────────────────────────────────────────────────────────


def _numerical_purification(p0, N, M, shots) -> float:
    if p0 == 1.0:
        return 1.0
    means = []
    for _ in range(shots):
        # initialise all N pairs with identical fidelity p0
        Q = deque([p0] * N)

        # keep purifying the two worst pairs until only M or M+1 remain
        while len(Q) > M + 1:
            p1 = Q.popleft()
            p2 = Q.popleft()
            p_out = _bbpssw_step(p1, p2)
            if random() < 0.5 * (1 + p1 * p2):
                Q.append(p_out)
            # if it fails we simply discarded two pairs → len(heap) drops by 2
        if len(Q) == 0:
            print(f"N {N}, M {M}")
        means.append(mean(Q))  # average fidelity of the survivors
    return mean(means)  # average over all Monte‑Carlo runs


# ──────────────────────────────────────────────────────────────────────────────
#  Analytical dilution (M > N)
# ──────────────────────────────────────────────────────────────────────────────


def _lambda0(
    x: float, *, xtol: float = 2e-7, rtol: float = 1e-10, maxiter: int = 100
) -> float:
    """Root of H₂(λ)=1/x.  Keeps Werner p accurate to 5 decimals for x≤100."""
    # TODO check
    if isclose(x, 1.0):
        return 0.5
    if x > 3.0e7:  # threshold where f(1e‑9) changes sign
        return 1e-9  # already within 1 ulp of the true root
    f = lambda lam: entropy([lam, 1 - lam], base=2) - 1 / x
    return brentq(f, 1e-9, 0.5 - 1e-9, xtol=xtol, rtol=rtol, maxiter=maxiter)


def _analytical_dilution(p0: float, N: int, M: int) -> float:
    x = M / N
    if p0 == 1.0:  # perfect Bell pairs → Schumacher compression only
        lam = _lambda0(x)
        return werner_param_from_eigenvalue(lam)
    else:
        return p0 / x


# ──────────────────────────────────────────────────────────────────────────────
#  Unified function
# ──────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=None)  # TODO maybe 200_000
def calculate_effective_edge_p(
    p0: float,  # p0 should be rounded to 5 decimal places BEFORE
    number_initial_pairs: int,
    number_requested_pairs: int,
    shots: int,
    precompute_cache: bool,
) -> float:
    p0 = round(p0, 5)
    if number_initial_pairs == 0:
        return 0.0
    elif number_requested_pairs == number_initial_pairs:
        return p0
    elif p0 == 1.0 and number_requested_pairs < number_initial_pairs:
        return p0

    number_requested_pairs = max(1, number_requested_pairs)
    if precompute_cache:
        key = (
            p0,
            number_initial_pairs,
            number_requested_pairs,
        )
        if key in _precomputed_table_cache:
            return _precomputed_table_cache[key]
        return precompute_table(
            p0,
            number_initial_pairs,
            number_requested_pairs,
            shots,
        )
    else:
        _, p_eff = _compute_one(p0, number_initial_pairs, shots, number_requested_pairs)
        return p_eff


def _compute_one(p0, number_initial_pairs, shots, M):
    if M < number_initial_pairs:
        return M, _numerical_purification(p0, number_initial_pairs, M, shots)
    else:
        return M, _analytical_dilution(p0, number_initial_pairs, M)


def precompute_table(
    p0: float,
    number_initial_pairs: int,
    number_requested_pairs: int,
    shots: int,
) -> float:
    # where to store / read the cache
    out_file = (
        CACHE_DIR / f"bbpssw_N={number_initial_pairs}_p0={p0:.5f}_shots={shots}.npz"
    )
    lock_file = str(out_file) + ".lock"

    with FileLock(lock_file):

        # --- loading -------------------------------------------------
        if out_file.exists():
            raw = load(out_file, allow_pickle=True)
            cache = dict(raw["cache"].item())  # Extract the cache dictionary

            # Populate the in-memory cache with all values from disk
            for M, value in cache.items():
                _precomputed_table_cache[(p0, number_initial_pairs, M)] = value

            # Check if requested value exists in loaded cache
            if number_requested_pairs in cache:
                return cache[number_requested_pairs]

            # If not, determine what values need to be calculated
            need = set()
            for M in range(number_requested_pairs + 1, 0, -1):
                if M not in cache:
                    need.add(M)
                else:
                    break
        else:
            # No cache file exists, need to calculate everything
            cache = {}
            upper = max(
                int(DILUTION_PRECOMPUTE_LEVEL * number_initial_pairs),
                number_requested_pairs,
            )
            need = set(range(1, upper + 1))

        # Calculate missing values
        if need:
            # spawn a pool of workers equal to your CPU count
            with Pool() as pool:
                # partial in the constant args
                worker = partial(_compute_one, p0, number_initial_pairs, shots)
                # map over M-values
                for M, val in pool.map(worker, sorted(need)):
                    cache[M] = val
                    _precomputed_table_cache[(p0, number_initial_pairs, M)] = val

            # now save to disk once
            savez_compressed(out_file, cache=cache)

    return _precomputed_table_cache[(p0, number_initial_pairs, number_requested_pairs)]
