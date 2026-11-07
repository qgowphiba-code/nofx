"""Contextual refinement tasks built on the published models (batch-to-batch variation is our addition).

pr2      Park-Ramirez; context: feed concentration sf [12, 28], initial glucose x4(0) [3, 7]          (= Exp A'')
pr10     pr5 + initial volume [0.8, 1.2], secretion-rate factor [0.7, 1.3], substrate-yield constant (published 7.3) [5.5, 9], 2 more distractors
pr5      Park-Ramirez; context: sf, x4(0), initial cells x3(0) [0.6, 1.4], growth-rate factor on the 21.87
         constant [0.75, 1.25] (assay), and one irrelevant measured variable (distractor)
catmix2  catalyst mixing; context: rate ratio (the published 10) [4, 25] and second-reaction factor [0.5, 2] (assays)
batch3   A->B->C batch reactor; context: catalyst activity factors on k1 [0.5, 2.0] and k2 [0.5, 2.0] (assays),
         initial concentration x1(0) [0.7, 1.3]; J = x2(tf)
pr2drift pr2, but after the drift switch the feed pump delivers 0.7 x the commanded rate (not measured)
All controls: 20 equal stages, u in [-1, 1] mapped onto the published bounds.
"""
import math, random, json, glob
from pathlib import Path
import problems
H = 20
ROOT = Path(__file__).resolve().parents[1]

def pr_J(u, sf=20.0, x40=5.0, x30=1.0, gf=1.0, pump=1.0, v0=1.0, sec=1.0, deg=7.3):
    def f(x, c):
        x1, x2, x3, x4, x5 = x
        g3 = gf * 21.87 * x4 / ((x4 + 0.4) * (x4 + 62.5)); g2 = sec * x4 * math.exp(-5 * x4) / (0.1 + x4); g1 = 4.75 * g3 / (0.12 + g3)
        return [g1 * (x2 - x1) - c * x1 / x5, g2 * x3 - c * x2 / x5, g3 * x3 - c * x3 / x5, -deg * g3 * x3 + c * (sf - x4) / x5, c]
    x = [0.0, 0.0, x30, x40, v0]; h = 15.0 / H / 60
    for k in range(H):
        x = problems._rk4(f, x, pump * 2.0 * (max(-1.0, min(1.0, u[k])) + 1) / 2, h, 60)
    return x[0] * x[4]

def batch_J(u, a=1.0, b=1.0, x10=1.0):
    def f(x, T):
        k1 = a * 4000.0 * math.exp(-2500.0 / T); k2 = b * 6.2e5 * math.exp(-5000.0 / T)
        return [-k1 * x[0] ** 2, k1 * x[0] ** 2 - k2 * x[1]]
    x = [x10, 0.0]; h = 1.0 / H / 40
    for k in range(H):
        x = problems._rk4(f, x, 298.0 + 100.0 * (max(-1.0, min(1.0, u[k])) + 1) / 2, h, 40)
    return x[1]

def catmix_J(u, kr=10.0, kd=1.0, x20=0.0):
    def f(x, c):
        return [c * (kr * x[1] - x[0]), c * (x[0] - kr * x[1]) - (1 - c) * kd * x[1]]
    x = [1.0, x20]; h = 1.0 / H / 40
    for k in range(H):
        x = problems._rk4(f, x, (max(-1.0, min(1.0, u[k])) + 1) / 2, h, 40)
    return 1 - x[0] - x[1]

def _base(pb):
    best = max((json.load(open(p)) for p in glob.glob(str(ROOT / f'runs/dyn/machina-{pb}-*/summary.json'))), key=lambda d: d['best_J'])
    return best['best_u'], best['best_J'], best['domain_id']

class Task:
    def __init__(self, name):
        self.name = name; self.adim = 1
        lu = lambda r, lo, hi: math.exp(r.uniform(math.log(lo), math.log(hi)))
        if name in ('pr2', 'pr2drift'):
            self.sample = lambda r: {'sf': r.uniform(12, 28), 'x40': r.uniform(3, 7)}
            self.vec = lambda c: [(c['sf'] - 20) / 8, (c['x40'] - 5) / 2]
            self.J = lambda u, c, drift=False: pr_J(u, c['sf'], c['x40'], pump=0.7 if drift else 1.0)
            self.norm, self.pb = 32.68, 'parkram'
        elif name == 'pr5':
            self.sample = lambda r: {'sf': r.uniform(12, 28), 'x40': r.uniform(3, 7), 'x30': r.uniform(0.6, 1.4), 'gf': r.uniform(0.75, 1.25), 'noise': r.uniform(-1, 1)}
            self.vec = lambda c: [(c['sf'] - 20) / 8, (c['x40'] - 5) / 2, (c['x30'] - 1) / 0.4, (c['gf'] - 1) / 0.25, c['noise']]
            self.J = lambda u, c, drift=False: pr_J(u, c['sf'], c['x40'], c['x30'], c['gf'])
            self.norm, self.pb = 32.68, 'parkram'
        elif name == 'pr10':
            self.sample = lambda r: {'sf': r.uniform(12, 28), 'x40': r.uniform(3, 7), 'x30': r.uniform(0.6, 1.4), 'gf': r.uniform(0.75, 1.25),
                                     'v0': r.uniform(0.8, 1.2), 'sec': r.uniform(0.7, 1.3), 'deg': r.uniform(5.5, 9.0),
                                     'n1': r.uniform(-1, 1), 'n2': r.uniform(-1, 1), 'n3': r.uniform(-1, 1)}
            self.vec = lambda c: [(c['sf'] - 20) / 8, (c['x40'] - 5) / 2, (c['x30'] - 1) / 0.4, (c['gf'] - 1) / 0.25, (c['v0'] - 1) / 0.2,
                                  (c['sec'] - 1) / 0.3, (c['deg'] - 7.25) / 1.75, c['n1'], c['n2'], c['n3']]
            self.J = lambda u, c, drift=False: pr_J(u, c['sf'], c['x40'], c['x30'], c['gf'], v0=c['v0'], sec=c['sec'], deg=c['deg'])
            self.norm, self.pb = 32.68, 'parkram'
        elif name == 'batch3':
            self.sample = lambda r: {'a': lu(r, 0.5, 2.0), 'b': lu(r, 0.5, 2.0), 'x10': r.uniform(0.7, 1.3)}
            self.vec = lambda c: [math.log(c['a']) / math.log(2), math.log(c['b']) / math.log(2), (c['x10'] - 1) / 0.3]
            self.J = lambda u, c, drift=False: batch_J(u, c['a'], c['b'], c['x10'])
            self.norm, self.pb = 0.6108 * 1.3, 'batch'
        elif name == 'catmix2':
            self.sample = lambda r: {'kr': lu(r, 4.0, 25.0), 'kd': lu(r, 0.5, 2.0)}
            self.vec = lambda c: [math.log(c['kr'] / 10.0) / math.log(2.5), math.log(c['kd']) / math.log(2)]
            self.J = lambda u, c, drift=False: catmix_J(u, c['kr'], c['kd'])
            self.norm, self.pb = 0.2, 'catmix'
        elif name == 'pen2':
            import penicillin as PEN
            self.sample = lambda r: {'sf': r.uniform(350, 650), 'x10': r.uniform(1.0, 2.0)}
            self.vec = lambda c: [(c['sf'] - 500) / 150, (c['x10'] - 1.5) / 0.5]
            self.J = lambda u, c, drift=False: PEN.J(u, c['sf'], c['x10'])
            self.norm, self.pb = 88.0, 'penicillin'
        elif name in ('insulin', 'arm'):
            import sys as _s; _s.path.insert(0, str(ROOT / 'varied'))
            M = __import__('insulin_task' if name == 'insulin' else 'arm_task')
            self.sample, self.vec, self.norm, self.pb = M.sample_train, M.vec, 1.0, name
            self.J = lambda u, c, drift=False: M.J(u, c)
            self.adim = 2 if name == 'arm' else 1
            mk2 = lambda n, s0: [M.sample_train(random.Random(s0 * 1000 + i)) for i in range(n)]
            self.SUPPORT, self.TRAIN, self.HELD = mk2(64, 141), mk2(256, 252), M.HELD
            self.base, self.baseJ, self.base_src = _base(self.pb)
            return
        seeds = {'pr2': (101, 202, 303), 'pr2drift': (101, 202, 303), 'pr5': (111, 222, 333), 'batch3': (121, 232, 343), 'catmix2': (131, 242, 353), 'pen2': (151, 262, 373), 'pr10': (161, 272, 383)}[name]
        mk = lambda n, s: [self.sample(random.Random(s * 1000 + i)) for i in range(n)]
        if name in ('pr2', 'pr2drift'):           # identical draws to Exp A''
            import pr_ctx as P
            self.SUPPORT, self.TRAIN, self.HELD = P.SUPPORT, P.TRAIN, P.HELD
        else:
            self.SUPPORT, self.TRAIN, self.HELD = mk(64, seeds[0]), mk(256, seeds[1]), mk(32, seeds[2])
        self.base, self.baseJ, self.base_src = _base(self.pb)
