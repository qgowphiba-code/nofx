"""B1 (+ B2 final evaluation): replicated, length-matched lineages (plan section 6).

Per lineage l: 48 training chains sequences(80200+l, [24,32,40,48,56,60]*8) excluding Eval-L, the Liu panel
and the other lineages. Arms A0 (lam 0) and BL (lam*, d*), both with r* and f* from the G1 manifest
(manifests/track_b.json). One domain per (arm, lineage).
  g0: 144 random folds (3 per chain, Random(80300+l)). A0 fitted first; BL's g0 targets use u at depth d*
      from the g0 A0 model (reconsideration pass).
  g1, g2: 144 own-policy folds collected by the arm's previous candidate at depth 2, eps 0.15
      (Random(80400+10l+g) per fold stream); the search runs at every step before the eps coin, so the
      depth-2 value is recorded for every state. Fits are cold (or f*) on all rows so far, install false.
Evaluation: g0, g1 -> Eval-Liu depth 3. g2 -> Eval-L (40) and Eval-Liu depth 3; B2: leaf blend
lam_leaf in {0.25, 0.5, 0.75} on Eval-Liu (development), the chosen lam_leaf on Eval-L (test); depths 1, 2
on Eval-L for C1. Guards: every generation's candidate must change predictions on a fixed probe set.
Futility gate G2 after lineage 1's g1: BL trailing A0 by > 8 on Eval-Liu drops BL from lineages 2-3.
Usage: b1_lineages.py <lineage> [--arms A0,BL]
"""
import argparse
import json
import random
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpx import settings as S, suites as SU, targets as T  # noqa: E402
from hpx.client import GuardError  # noqa: E402
from hpx.evaluate import long_eval, model_evaluator  # noqa: E402
from hpx.hpenv import BASIC, FEATURES, Fold  # noqa: E402
from hpx.planner import Blend, run_folds  # noqa: E402
from hpx.runlog import ROOT, Run, connect, sha256_json  # noqa: E402
from hpx.vendor import lattice_env, liu_panel  # noqa: E402

TRACK_B = Path(__import__('os').environ.get('HPX2_TRACK_B', ROOT / 'manifests' / 'track_b.json'))
LENGTHS = [24, 32, 40, 48, 56, 60] * 8
LEAF_LAMBDAS = (0.25, 0.5, 0.75)


def lineage_chains(lineage):
    excl = {lattice_env.sequence_key(s) for s in SU.load('eval_l')['sequences'] + [s for _, s in liu_panel()]}
    for other in (1, 2, 3):
        if other < lineage:
            excl |= {lattice_env.sequence_key(s) for s in lineage_chains(other)}
    return lattice_env.sequences(80200 + lineage, LENGTHS, excluded=excl)


def probe_inputs(conditions):
    suite = SU.load('eval_s')
    seqs = {c['ci']: c['sequence'] for c in suite['chains']}
    return [Fold(seqs[d['ci']], 3, d['chain']).observation(conditions) for d in suite['decisions'][::14][:64]]


class Arm:
    def __init__(self, name, lam, depth, cfg, lineage, client, run):
        self.name, self.lam, self.depth, self.cfg, self.lineage = name, lam, depth, cfg, lineage
        self.c, self.run = client, run
        self.conditions = FEATURES if cfg['conditions'] == 'features' else BASIC
        self.domain = None
        self.episodes, self.row_ids, self.models = [], [], []
        self.out = {'arm': name, 'lam': lam, 'd': depth, 'generations': []}
        self.probe = probe_inputs(self.conditions)
        self.last_probe = None

    def log(self, *m):
        self.run.log(f'[l{self.lineage} {self.name}]', *m)

    def setup(self):
        self.domain = self.c.create_domain('b1', self.name.lower(), self.lineage)
        self.c.configure(self.domain, S.value_config(self.cfg['representation']))
        self.out['domain'] = self.domain

    def evaluator(self, sha):
        return model_evaluator(self.c, self.domain, 'candidate', sha, conditions=self.conditions, name=f'{self.name}:{sha[:8]}')

    def rows(self, eps, u=None, assessor=None):
        if self.lam == 0:
            return T.terminal_rows(eps, arm=f'B1{self.name}', conditions=self.conditions)
        return T.blended_rows(eps, self.lam, self.depth, u, assessor, arm=f'B1{self.name}', conditions=self.conditions)

    def fit(self, g):
        val = T.validation_episodes(self.episodes, fit_index=300 + 10 * self.lineage + g)
        opts = S.fit_options(self.cfg['base'], validation_episode_ids=val, **self.cfg['fit_overrides'])
        fit = self.c.train(self.domain, f'b1-l{self.lineage}-{self.name}-g{g}', opts, install=False,
                           observation_ids=self.row_ids, name=f'l{self.lineage}-{self.name}-g{g}')
        sha = fit['sha256']
        pv = self.c.predict(self.domain, self.probe, 'candidate', sha)
        if self.last_probe is not None and max(abs(a - b) for a, b in zip(pv, self.last_probe)) == 0:
            raise GuardError(f'{self.name} g{g}: predictions identical to the previous generation (identical generation)')
        self.last_probe = pv
        self.models.append(sha)
        rep = fit['state']['training']['report']
        self.log(f'g{g} fitted {sha[:12]} on {len(self.row_ids)} rows in {fit["elapsed"]}s, selected_epoch {rep.get("selected_epoch")}')
        return sha, rep, val

    def collect(self, g, sha):
        """144 own-policy folds at depth 2, eps 0.15, by the previous candidate; u recorded at every step."""
        chains = self.cfg['chains']
        master = random.Random(80400 + 10 * self.lineage + g)
        rngs = [random.Random(master.random()) for _ in range(len(chains) * 3)]
        seqs = [chains[i // 3] for i in range(len(chains) * 3)]
        ev = self.evaluator(sha)
        res = run_folds(seqs, 2, ev, rngs=rngs, eps=0.15, record=True)
        eps, u2 = [], {}
        for i, r in enumerate(res):
            chs = [tuple(map(tuple, s['chain'])) for s in r['trajectory']] + [tuple(map(tuple, r['chain']))]
            e = T.Episode(f'b1l{self.lineage}{self.name}-g{g}-c{i // 3}-f{i % 3}', seqs[i], chs, generation=g, collector=sha)
            eps.append(e)
            for k, s in enumerate(r['trajectory']):
                u2[(e.id, k)] = s['u']
        contacts = [e.contacts for e in eps]
        self.log(f'g{g} collected 144 folds: mean contacts {sum(contacts) / len(contacts):.2f}; predict {ev.stats}')
        return eps, u2, ev

    def add(self, eps, u=None, assessor=None):
        rows = self.rows(eps, u, assessor)
        self.c.observe(self.domain, rows)
        self.episodes.extend(eps)
        self.row_ids.extend(r['record_id'] for r in rows)
        return rows


def eval_liu(arm, sha, run_seed=0):
    ev = arm.evaluator(sha)
    r = long_eval(ev, [s for _, s in liu_panel()], 3, seed=run_seed)
    return r, ev


def run_arm_generations(arm, g0_eps, g0_u, a0_g0_sha, results, lock, gate):
    """g0 rows/fit (A0 already fitted for arm A0), then g1, g2; final evaluation."""
    try:
        if arm.name != 'A0':
            arm.add(g0_eps, g0_u, a0_g0_sha)
            sha, rep, val = arm.fit(0)
            lv, _ = eval_liu(arm, sha)
            arm.out['generations'].append({'g': 0, 'sha256': sha, 'n_rows': len(arm.row_ids), 'liu_depth3': lv,
                                           'selected_epoch': rep.get('selected_epoch'), 'assessor': a0_g0_sha})
            arm.log(f'g0 Eval-Liu depth 3: {lv["total"]} {lv["per_chain"]}')
        sha = arm.out['generations'][-1]['sha256']
        for g in (1, 2):
            eps, u2, cev = arm.collect(g, sha)
            if arm.lam:
                u = u2 if arm.depth == 2 else T.assess(eps, arm.depth, cev)
                arm.add(eps, u, sha)
            else:
                arm.add(eps)
            sha, rep, val = arm.fit(g)
            gen = {'g': g, 'sha256': sha, 'n_rows': len(arm.row_ids), 'selected_epoch': rep.get('selected_epoch'),
                   'collector_mean_contacts': sum(e.contacts for e in eps) / len(eps), 'collector': cev.name}
            if g == 1:
                lv, _ = eval_liu(arm, sha)
                gen['liu_depth3'] = lv
                arm.log(f'g1 Eval-Liu depth 3: {lv["total"]} {lv["per_chain"]}')
                arm.out['generations'].append(gen)
                gate.wait()   # G2 checkpoint: both arms reach g1; lineage 1 records the decision for lineages 2-3
            else:
                arm.out['generations'].append(gen)
        final_eval(arm, sha)
    except BaseException as e:  # noqa: BLE001 - recorded and re-raised by the main thread
        arm.out['error'] = repr(e)
        gate.abort()
        raise
    finally:
        with lock:
            results['arms'][arm.name] = arm.out
            arm.run.write_results(results, f'results_l{arm.lineage}.json')


def final_eval(arm, sha):
    """g2: Eval-L and Eval-Liu at depth 3; B2 leaf blend (dev Liu -> test Eval-L); depths 1, 2 on Eval-L (C1)."""
    evl = SU.load('eval_l')['sequences']
    liu = [s for _, s in liu_panel()]
    ev = arm.evaluator(sha)
    fin = {'sha256': sha, 'eval_l_depth3': long_eval(ev, evl, 3), 'liu_depth3': long_eval(ev, liu, 3)}
    arm.log(f'g2 Eval-L depth 3 total {fin["eval_l_depth3"]["total"]}, Eval-Liu {fin["liu_depth3"]["total"]}')
    blend_dev = {}
    for lam in LEAF_LAMBDAS:
        blend_dev[str(lam)] = long_eval(Blend(ev, lam), liu, 3)
        arm.log(f'B2 leaf blend {lam} on Eval-Liu: {blend_dev[str(lam)]["total"]}')
    best = max(LEAF_LAMBDAS, key=lambda x: blend_dev[str(x)]['total'])
    fin['blend_dev_liu'] = blend_dev
    fin['lam_leaf'] = best
    fin['blend_test_eval_l'] = long_eval(Blend(ev, best), evl, 3)
    arm.log(f'B2 chosen lam_leaf {best}: Eval-L {fin["blend_test_eval_l"]["total"]}')
    for d in (1, 2):
        fin[f'eval_l_depth{d}'] = long_eval(ev, evl, d)
    fin['predict_stats'] = dict(ev.stats)
    arm.out['final'] = fin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lineage', type=int)
    ap.add_argument('--arms', default='A0,BL')
    a = ap.parse_args()
    cfg = json.loads(TRACK_B.read_text())
    run = Run('B1', name=f'l{a.lineage}-' + __import__('hpx.runlog', fromlist=['stamp']).stamp())
    c = connect(run)
    v = c.check_build('start')
    chains = lineage_chains(a.lineage)
    base = S.OBJECT_BASE if cfg['representation'] == 'object_graph' else S.PROJECTED_BASE
    arm_cfg = {'conditions': cfg['conditions'], 'representation': cfg['representation'], 'base': base,
               'fit_overrides': cfg.get('fit_overrides', {}), 'chains': chains}
    names = a.arms.split(',')
    arms = [Arm('A0', 0.0, 0, arm_cfg, a.lineage, c, run)]
    if 'BL' in names:
        arms.append(Arm('BL', cfg['lam_star'], cfg['d_star'], arm_cfg, a.lineage, c, run))
    run.note(host=S.HOST, build=v, track_b=cfg, chains_sha256=sha256_json(chains), lineage=a.lineage,
             seeds={'chains': 80200 + a.lineage, 'g0_folds': 80300 + a.lineage, 'collectors': [80400 + 10 * a.lineage + g for g in (1, 2)]})
    results = {'lineage': a.lineage, 'arms': {}, 'heuristic_card_liu': SU.load('baselines_eval_liu')['heuristic']['3']['mean_total']}
    lock = threading.Lock()
    try:
        for arm in arms:
            arm.setup()
        g0_eps = T.random_episodes(chains, 3, random.Random(80300 + a.lineage), f'b1l{a.lineage}-g0')
        a0 = arms[0]
        a0.add(g0_eps)
        sha0, rep0, _ = a0.fit(0)
        lv0, ev0 = eval_liu(a0, sha0)
        a0.out['generations'].append({'g': 0, 'sha256': sha0, 'n_rows': len(a0.row_ids), 'liu_depth3': lv0,
                                      'selected_epoch': rep0.get('selected_epoch')})
        a0.log(f'g0 Eval-Liu depth 3: {lv0["total"]} {lv0["per_chain"]}')
        g0_u = None
        if len(arms) > 1:
            g0_u = T.assess(g0_eps, arms[1].depth, a0.evaluator(sha0))
        gate = threading.Barrier(len(arms), action=lambda: g2_gate(results, arms, a.lineage, run))
        threads = [threading.Thread(target=run_arm_generations, args=(arm, g0_eps, g0_u, sha0, results, lock, gate))
                   for arm in arms]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        errors = {arm.name: arm.out.get('error') for arm in arms if arm.out.get('error')}
        if errors:
            raise RuntimeError(f'arm errors: {errors}')
    finally:
        for arm in arms:
            if arm.domain:
                try:
                    c.delete_domain(arm.domain)
                except Exception as e:  # noqa: BLE001
                    run.log('DELETE FAILED', arm.domain, repr(e))
        results['version_end'] = c.check_build('end')
        results['requests'] = c.counts
        run.write_results(results, f'results_l{a.lineage}.json')
        run.finish(requests=c.counts['requests'])


def g2_gate(results, arms, lineage, run):
    if lineage != 1 or len(arms) < 2:
        return
    a0 = arms[0].out['generations'][-1]['liu_depth3']['total']
    bl = arms[1].out['generations'][-1]['liu_depth3']['total']
    results['G2'] = {'a0_liu_g1': a0, 'bl_liu_g1': bl, 'drop_bl_from_lineages_2_3': bl < a0 - 8}
    run.log('G2 futility gate:', results['G2'])


if __name__ == '__main__':
    main()
