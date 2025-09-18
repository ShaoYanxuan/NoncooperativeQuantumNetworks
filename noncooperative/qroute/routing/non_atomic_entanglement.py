from functools import lru_cache
import math

import numpy as np
from qroute.routing.entanglement_manipulation import (
    _bbpssw_step,
    _lambda0,
    werner_param_from_eigenvalue,
    werner_param_from_fidelity,
)
from scipy.integrate import quad

MAX_PURIFICATION_LEVEL = 50
BBPSSW_PROB_THRESHOLD = 1e-9


def _compute_cost_and_p_list(p0: float) -> tuple[list[float], list[float]]:
    # Pre‑compute quality (p_k) and cost (c_k) for k = 0 … MAX_PURIFICATION_LEVEL
    p_levels = [p0]
    costs = [1.0]

    s, p = 1.0, p0
    for i in range(1, MAX_PURIFICATION_LEVEL):
        s *= 0.5 * (1.0 + p * p)
        if s < BBPSSW_PROB_THRESHOLD:
            break
        p = _bbpssw_step(p, p)
        p_levels.append(p)
        costs.append(2**i / s)
    return costs, p_levels


def analytical_dilution(p0: float, ratio: float) -> float:
    """Effective p when M/N = ratio ≥ 1"""
    if p0 == 1.0:
        lam = _lambda0(ratio)
        return werner_param_from_eigenvalue(lam)
    else:
        return p0 / ratio


def analytical_purification(p0: float, ratio: float) -> float:
    """Effective p when M/N = ratio ≤ 1"""
    costs, p_levels = _compute_cost_and_p_list(p0)
    # find i with 1 ≥ ratio*c_i and 1 ≤ ratio*c_{i+1}
    for i in range(len(costs) - 1):
        c_i, c_next = costs[i], costs[i + 1]
        if 1.0 >= ratio * c_i and 1.0 <= ratio * c_next:
            # r = (N - M*c_next)/(c_i - c_next)
            r = (1 - ratio * c_next) / (c_i - c_next)
            return (r * p_levels[i] + (ratio - r) * p_levels[i + 1]) / (ratio)
    raise RuntimeError(f"Purification ratio out of bounds: {ratio}")


def evaluate_convex_cost(
    flow: float, p0: float = werner_param_from_fidelity(0.95), cap: float = 1.0
) -> float:
    """
    Instantaneous cost c_e(flow) = -flow * ln(p_eff(flow))
    where p_eff(flow) is from analytical_{dilution,purification} using flow/cap
    """
    ratio = flow / cap  # TODO
    if ratio >= 1.0:
        val = analytical_dilution(p0, ratio)
    else:
        val = analytical_purification(p0, ratio)
    return -flow * math.log(val)


def integrate_convex_cost_numerical(x, P0=werner_param_from_fidelity(0.95)):
    def cost(t):
        if t >= 1:
            val = analytical_dilution(P0, t)
        else:
            val = analytical_purification(P0, t)
        return -math.log(val)

    result, _ = quad(cost, 1e-6, x, epsabs=1e-6, epsrel=1e-6)
    return result


def h_purification(x, p0):
    """
    Exact ∫_0^x -ln(p(t)) dt for 0 < x ≤ 1,
    where p(t) comes from _analytical_purification().
    """

    # --- break-points and p-levels -------------------------------
    costs, p_levels = _compute_cost_and_p_list(p0)
    x_break = [1 / c for c in costs]  # descending
    p_levels = p_levels[::-1]  # match ascending x
    x_break = x_break[::-1]  # ascending  (x0 smallest)

    # prepend constant segment [0, x_break[0]]
    p_const = p_levels[0]
    area = 0.0
    if x <= x_break[0]:
        return -math.log(p_const) * x  # still in flat first bit
    area += -math.log(p_const) * x_break[0]
    prev = x_break[0]

    # --- piece-wise A/x + B segments -----------------------------
    for i in range(len(p_levels) - 1):  # now in ascending order
        x_lo, x_hi = x_break[i], x_break[i + 1]  # x_lo < x_hi ≤ 1
        p_lo, p_hi = p_levels[i], p_levels[i + 1]
        A = (p_lo - p_hi) / (1 / x_lo - 1 / x_hi)  # = (p_i - p_{i+1}) / (c_i - c_{i+1})
        B = p_lo - A / x_lo

        # integral on [prev, upper] where upper=min(x,x_hi)
        upper = min(x, x_hi)
        area += _seg_int_rational(A, B, prev, upper)
        if x <= x_hi:
            break
        prev = x_hi

    return area


def _seg_int_rational(A, B, lo, hi):
    """∫_{lo}^{hi} [ln t - ln(A+Bt)] dt  (A>0, B>0)."""

    def F(t):
        return t * math.log(t) - t - (A + B * t) / B * (math.log(A + B * t) - 1.0)

    return F(hi) - F(lo)


# NEW ---------------------------
# analytic ∫_{1}^{x} -ln(p(t)) dt  for dilution -----------------------
def _h_dilution(x: float, p0: float) -> float:
    """Exact area from t=1 to t=x (x≥1) for the dilution regime."""
    if p0 < 1.0:
        lnP = math.log(p0)
        return -(lnP) * (x - 1) + (x * math.log(x) - x + 1)
    # degenerate Schumacher-compression branch, x ↦ λ(x)
    # Happens only when p0 == 1.
    # Still cheap — very small interval — but keep numeric for clarity.
    f = lambda t: -math.log(analytical_dilution(p0, t))
    res, _ = quad(f, 1.0, x, epsabs=1e-9, epsrel=1e-9)
    return res


# ----------------------------------------------------------------------


def integrate_convex_cost_analytical(
    x: float, P0: float = werner_param_from_fidelity(0.95), cap: float = 1.0
) -> float:
    x = x / cap  # TODO: adjust for cap
    """
    Analytic everywhere:
      - exact h_purification for 0 < x ≤ 1
      - exact closed form for dilution part when x > 1
    """
    if x <= 1.0:
        return h_purification(x, P0)
    # split at x=1
    return h_purification(1.0, P0) + _h_dilution(x, P0)
