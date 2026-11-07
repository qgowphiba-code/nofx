"""Olympus measured datasets (Hase et al. 2021): crossed_barrel (Gongora et al. 2020, 3D-printed crossed-barrel toughness, 600
designs), photo_pce10 (organic photovoltaic PCE10 power conversion efficiency, 1040). Pool-based: choose among measured rows."""
import csv
from pathlib import Path
RAW = Path(__file__).resolve().parents[1] / 'data/raw/olympus'
def load(name):
    rows = [[float(x) for x in r] for r in csv.reader(open(RAW / f'{name}.csv')) if r]
    X = [r[:-1] for r in rows]; y = [r[-1] for r in rows]
    lo = [min(c) for c in zip(*X)]; hi = [max(c) for c in zip(*X)]
    Xs = [[(v - a) / (b - a) if b > a else 0.0 for v, a, b in zip(x, lo, hi)] for x in X]
    return Xs, y
