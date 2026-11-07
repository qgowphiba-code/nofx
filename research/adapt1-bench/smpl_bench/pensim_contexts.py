"""Batch contexts for PenSim refinement: PenSimPy's own per-batch initial-state draw, selected by random_seed_ref.
TRAIN: 12 seed_refs, HELD: 16 seed_refs (disjoint, fixed). vec(seed_ref) = the 8 measured initial conditions scaled to about
[0, 1] (pensim_plant.context_of), cached after one reset per seed_ref (no simulation needed).
GATE: 16 further seed_refs used only by the C2 refinement gate."""
import json, os
import numpy as np
import pensim_plant as P

TRAIN = list(range(2000, 2012))
HELD = list(range(3000, 3016))
GATE = list(range(4000, 4016))
_CACHE = {}
_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'pensim', 'contexts.json')


def vec(seed_ref):
    if seed_ref not in _CACHE:
        if os.path.exists(_CACHE_FILE) and not _CACHE:
            try:
                _CACHE.update({int(k): v for k, v in json.load(open(_CACHE_FILE)).items()})
            except Exception:
                pass
        if seed_ref not in _CACHE:
            e = P.make_env(); e.reset(random_seed_ref=seed_ref)
            raw, scaled = P.context_of(e)
            _CACHE[seed_ref] = {'raw': raw.tolist(), 'vec': scaled}
            os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
            json.dump({str(k): v for k, v in _CACHE.items()}, open(_CACHE_FILE, 'w'), indent=0)
    return _CACHE[seed_ref]['vec']
