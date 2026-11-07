"""Exact design oracle: Boltzmann probability of the target conformation for a designed HP sequence."""
import json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
import lattice2d as L
C = np.load(HERE / 'C.npy')
BANK = json.load(open(HERE / 'bank.json'))['targets']

def reward(seq, target):
    s = np.array([1 if ch == 'H' else 0 for ch in seq])
    hh = np.array([s[i] * s[j] for (i, j) in L.PAIRS])
    E = -(C @ hh)
    w = np.exp(-(E - E.min()) / L.TEMP)
    return float(w[BANK[target]['conf_index']] / w.sum())
