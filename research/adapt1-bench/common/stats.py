"""Exact two-sided Mann-Whitney / permutation test on the difference in means (stdlib)."""
import itertools, statistics as st, random

def perm_test(a, b, n_perm=20000, seed=0):
    obs = st.mean(a) - st.mean(b); pool = list(a) + list(b); na = len(a)
    r = random.Random(seed); hits = 0
    for _ in range(n_perm):
        r.shuffle(pool)
        if abs(st.mean(pool[:na]) - st.mean(pool[na:])) >= abs(obs) - 1e-12: hits += 1
    return obs, (hits + 1) / (n_perm + 1)
