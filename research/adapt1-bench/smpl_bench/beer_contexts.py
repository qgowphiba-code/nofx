"""Batch contexts for BeerFMT refinement: SMPL's own initial-condition variation (X_L, X_D within 1.8-2.2 g/L, sugar
within 117-143 g/L, uniform), measured at reset. Fixed draws so every arm and comparator sees the same batches.
TRAIN: 64 contexts for refinement updates. HELD: 32 unseen contexts for frozen evaluation. Disjoint by construction.
vec(c) scales each measured variable to [0, 1] over its sampling range.
"""
import numpy as np
import beer_plant as B

LO = np.array([1.8, 1.8, B.INIT_SUGAR * 0.9])
HI = np.array([2.2, 2.2, B.INIT_SUGAR * 1.1])


def _draw(seed, n):
    rng = np.random.default_rng(seed)
    return [B.sample_context(rng) for _ in range(n)]


TRAIN = _draw(20261107, 64)
HELD = _draw(7112026, 32)


def vec(x0):
    """x0 = full 8-vector initial observation -> 3 measured context features in [0, 1]."""
    m = np.array([x0[1], x0[2], x0[3]])
    return [float(v) for v in (m - LO) / (HI - LO)]


def quality_of(u_or_temps, x0, is_u=True):
    temps = B.profile_from_u(u_or_temps) if is_u else u_or_temps
    return B.quality(B.run(temps, x0)['return'])
