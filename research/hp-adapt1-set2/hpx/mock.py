"""Local mock of the Adapt-1 routes the experiments use (plan 2.4 "dry runs").

Response shapes mirror what T0.2 recorded on Fast-UAT build af6670b: domain create -> 201, value/state with
`model.model_sha256` / `candidate.model_sha256` / `training{status: trained, request_id, installed, report}`,
predict -> {evaluator{model_sha256, ...}, predictions[]}, planning -> {evaluator, plan{action, value, ...}}.
Refusals mirror recorded ones: 409 on a changed record_id or a wrong expected sha, 413 above 1 MiB, 422 on
rows without exactly one supervision source or continuation rows without a fitted reference.

The value function of a mock "model" is the contact heuristic recomputed from the observation plus a
deterministic sha-keyed perturbation, so every fit has a distinct, reproducible identity and output.
`faults` injects statuses per route (applied after or before the write) to exercise retry/reconcile paths.
"""
import hashlib
import json
import time

ONE_MIB = 1 << 20


def obs_heuristic(obs, dim=3):
    types = [o['type'][0] for o in obs.get('objects') or []]
    n = len(types)
    if not n:
        return .5
    ub = max(1, sum(2 * dim - int(i > 0) - int(i < n - 1) for i, t in enumerate(types) if t == 'H') // 2)
    c = sum(r.get('type') == 'hh_contact' for r in obs.get('relations') or [])
    return .5 + .5 * c / ub


def _h(*parts):
    return hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()


class MockModel:
    def __init__(self, sha, amplitude=0.02, value_fn=obs_heuristic, version=1, dataset_sha=None, model_type='object_value'):
        self.sha, self.amplitude, self.value_fn = sha, amplitude, value_fn
        self.version, self.dataset_sha, self.model_type = version, dataset_sha, model_type

    def value(self, obs):
        key = json.dumps({k: obs.get(k) for k in ('objects', 'relations', 'conditions')}, sort_keys=True, separators=(',', ':'))
        noise = int(_h(self.sha, key)[:8], 16) / 0xffffffff * 2 - 1
        return min(1., max(0., self.value_fn(obs) + self.amplitude * noise))

    def summary(self):
        return {'model_sha256': self.sha, 'model_type': self.model_type, 'version': self.version,
                'dataset_sha256': self.dataset_sha, 'validation_skill': 0.5}


class MockDomain:
    def __init__(self, domain_id):
        self.id = domain_id
        self.config = None
        self.rows = {}           # record_id -> canonical json
        self.order = []
        self.model = None
        self.candidate = None
        self.training = None
        self.job = None          # (polls_left, finalize)
        self.version = 0

    def dataset_sha(self, ids=None):
        ids = self.order if ids is None else ids
        return _h(*[self.rows[i] for i in ids]) if ids else _h()

    def state(self):
        return {'domain_id': self.id, 'status': 'configured' if self.config else 'unconfigured', 'config': self.config,
                'sample_count': len(self.order), 'dataset_version': self.version, 'dataset_sha256': self.dataset_sha(),
                'model': self.model.summary() if self.model else None,
                'candidate': self.candidate.summary() if self.candidate else None,
                'training': self.training, 'frozen': False}


class MockTransport:
    def __init__(self, commit='0' * 40, fit_polls=1, amplitude=0.02, value_fn=obs_heuristic, faults=None):
        self.commit, self.fit_polls, self.amplitude, self.value_fn = commit, fit_polls, amplitude, value_fn
        self.domains = {}
        self.faults = faults or {}     # route -> list of (status, 'before'|'after')
        self.calls = []

    # ------------------------------------------------------------------ plumbing
    def request(self, method, path, body_bytes=None):
        body = json.loads(body_bytes) if body_bytes else None
        route = self._route(method, path)
        self.calls.append(route)
        if body_bytes is not None and len(body_bytes) > ONE_MIB:
            return 413, {'detail': 'trajectory payload exceeds 1 MiB'}, {}
        fault = self.faults.get(route, [])
        if fault and fault[0][1] == 'before':
            status, _ = fault.pop(0)
            return status, {'detail': 'injected'}, {}
        status, data = self._handle(method, path, body)
        if fault and fault[0][1] == 'after':
            status, _ = fault.pop(0)
            return status, {'detail': 'injected after apply'}, {}
        return status, data, {}

    @staticmethod
    def _route(method, path):
        parts = path.strip('/').split('/')
        if len(parts) >= 4 and parts[2] == 'domains':
            return method + ' ' + '/'.join(['domains', '{id}'] + parts[4:])
        return method + ' ' + '/'.join(parts)

    def _handle(self, method, path, body):
        parts = path.strip('/').split('/')
        if path == '/version':
            return 200, {'commit_sha': self.commit, 'schema_version': '5', 'version': 'mock'}
        if parts[:3] == ['api', 'v1', 'domains'] and len(parts) == 3:
            if method == 'GET':
                return 200, {'domains': [{'domain_id': d} for d in self.domains]}
            did = body['domain_id']
            if did in self.domains:
                return 409, {'detail': 'domain exists'}
            self.domains[did] = MockDomain(did)
            return 201, {'domain_id': did, 'created_at': time.time(), 'description': body.get('description')}
        did = parts[3]
        dom = self.domains.get(did)
        if len(parts) == 4:
            if dom is None:
                return 404, {'detail': 'Domain not found'}
            if method == 'DELETE':
                del self.domains[did]
                return 200, {'deleted': True, 'domain_id': did, 'status': 'success'}
            return 200, {'domain_id': did}
        if dom is None:
            return 404, {'detail': 'Domain not found'}
        if parts[4] == 'planning':
            return self._plan(dom, body)
        op = parts[5]
        return getattr(self, '_v_' + op)(dom, body)

    # ------------------------------------------------------------------ value routes
    def _v_configure(self, dom, body):
        cfg = dict(body['config'])
        cfg.setdefault('representation', 'projected')
        cfg.setdefault('max_samples', 20000)
        cfg.setdefault('promotion', {'mode': 'prediction', 'minimum_cases': 16, 'required_improvement': 0,
                                     'maximum_prediction_regression': 0.05})
        dom.config = cfg
        return 200, dom.state()

    def _v_state(self, dom, body):
        if dom.job is not None:
            left, fin = dom.job
            if left <= 0:
                fin()
                dom.job = None
            else:
                dom.job = (left - 1, fin)
        return 200, dom.state()

    def _v_observe(self, dom, body):
        obs = body.get('observations') or []
        if len(obs) > 50:
            return 422, {'detail': [{'loc': ['body', 'observations'], 'msg': 'at most 50', 'type': 'too_long'}]}
        for r in obs:
            if sum(r.get(k) is not None for k in ('target', 'outcome', 'continuation')) != 1:
                return 422, {'detail': 'provide exactly one of target or an observed outcome or a continuation'}
        acc = dup = cont = prov = 0
        for r in obs:
            canon = json.dumps(r, sort_keys=True, separators=(',', ':'))
            old = dom.rows.get(r['record_id'])
            if old is not None and old != canon:
                return 409, {'detail': 'record_id already exists with different contents'}
        for r in obs:
            canon = json.dumps(r, sort_keys=True, separators=(',', ':'))
            if r['record_id'] in dom.rows:
                dup += 1
                continue
            dom.rows[r['record_id']] = canon
            dom.order.append(r['record_id'])
            dom.version += 1
            acc += 1
            if r.get('continuation') is not None:
                cont += 1
            else:
                prov += 1
        return 200, {**dom.state(), 'accepted': acc, 'duplicates': dup,
                     'target_sources': {'model_continuation': cont, 'provided_or_observed': prov}}

    def _v_train(self, dom, body):
        ids = body.get('observation_ids') or list(dom.order)
        missing = [i for i in ids if i not in dom.rows]
        if missing:
            return 422, {'detail': f'unknown observation ids: {missing[:3]}'}
        rows = [json.loads(dom.rows[i]) for i in ids]
        opts = body.get('options') or {}
        if any(r.get('continuation') for r in rows) and dom.model is None:
            return 422, {'detail': 'nonterminal continuation targets require a fitted reference model'}
        warm = opts.get('warm_start', True)
        src = body.get('warm_start_source', 'installed')
        init = (dom.model if src == 'installed' else dom.candidate) if warm else None
        exp = body.get('expected_initial_model_sha256')
        if exp and (init is None or init.sha != exp):
            return 409, {'detail': 'initial model changed; expected_initial_model_sha256 does not match'}
        install = body.get('install', True)
        dsha = dom.dataset_sha(ids)
        sha = _h('fit', dsha, json.dumps(opts, sort_keys=True), init.sha if init else None, body['request_id'])
        dom.version += 1
        model = MockModel(sha, self.amplitude, self.value_fn, dom.version, dsha, opts.get('model_type', 'mlp_v2'))
        rid = body['request_id']
        dom.training = {'status': 'running', 'request_id': rid, 'install': install}

        def finalize():
            if install:
                dom.model = model
            else:
                dom.candidate = model
            dom.training = {'status': 'trained' if install else 'candidate', 'request_id': rid, 'install': install,
                            'installed': bool(install),
                            'options': opts, 'sample_count': len(ids), 'initial_model_sha256': init.sha if init else None,
                            'report': {'selected_epoch': 1, 'regularization_target': 'incumbent' if init else 'zero',
                                       'warm_start': {'applied': init is not None,
                                                      **({} if init else {'reason': 'no_initial_model'})},
                                       'fitted_samples': len(ids), 'validation_skill': 0.5,
                                       'target_refresh': {'continuation_records': sum(bool(r.get('continuation')) for r in rows),
                                                          'reference_model_sha256': dom.model.sha if dom.model else None}}}
        dom.job = (self.fit_polls, finalize)
        return 202, dom.state()

    def _evaluator(self, dom, source, expected):
        model = dom.model if source in ('installed', 'value_model') else dom.candidate
        if model is None:
            return None, (409, {'detail': 'owned Domain has no candidate value model' if source in ('candidate', 'value_candidate')
                                else 'owned Domain has no installed value model'})
        if expected and expected != model.sha:
            return None, (409, {'detail': 'value model changed; expected_model_sha256 does not match'})
        return model, None

    def _v_predict(self, dom, body):
        inputs = body.get('inputs') or []
        if not 1 <= len(inputs) <= 256:
            return 422, {'detail': [{'loc': ['body', 'inputs'], 'msg': '1 to 256 inputs', 'type': 'length'}]}
        model, err = self._evaluator(dom, body.get('model_source', 'installed'), body.get('expected_model_sha256'))
        if err:
            return err
        return 200, {'domain_id': dom.id, 'evaluator': {'source': 'value_model', 'model_type': model.model_type,
                                                         'model_version': model.version, 'model_sha256': model.sha,
                                                         'learning_updates': False, 'frozen': False,
                                                         'dataset_sha256': model.dataset_sha},
                     'predictions': [model.value(x) for x in inputs]}

    # ------------------------------------------------------------------ native planning
    def _plan(self, dom, body):
        src = body.get('evaluator_source', 'domain_model')
        model, err = self._evaluator(dom, src, body.get('expected_model_sha256'))
        if err:
            return err
        nodes = {n['id']: n for n in body.get('nodes') or []}
        root = nodes.get(body['root']['id'], body['root'])
        depth = (body.get('options') or {}).get('max_depth', 3)
        count = {'evaluations': 0, 'transitions': 0}

        def value(node, d):
            if node.get('terminal_value') is not None:
                return node['terminal_value']
            kids = node.get('children') or []
            if d == 0 or not kids:
                count['evaluations'] += 1
                return model.value(node)
            vals = []
            for e in kids:
                count['transitions'] += 1
                vals.append(value(nodes[e['target']], d - 1))
            return max(vals) if node.get('maximizing', True) else min(vals)
        scored = []
        for e in root.get('children') or []:
            count['transitions'] += 1
            scored.append((e['action'], value(nodes[e['target']], depth - 1)))
        best = (max if root.get('maximizing', True) else min)(v for _, v in scored)
        action = next(a for a, v in scored if v == best)
        return 200, {'evaluator': {'source': 'value_model' if src == 'value_model' else src, 'model_type': model.model_type,
                                   'model_version': model.version, 'model_sha256': model.sha, 'learning_updates': False,
                                   'frozen': False, 'dataset_sha256': model.dataset_sha},
                     'plan': {'action': action, 'value': best, 'completed_depth': depth, 'evaluations': count['evaluations'],
                              'transitions': count['transitions'], 'budget_exhausted': False, 'deadline_reached': False,
                              'extensions': 0, 'completed_extensions': 0, 'cache_hits': 0, 'branch_width': None},
                     'status': 'completed'}


def mock_client(run_dir, tmp, **kw):
    """A Client wired to a MockTransport with its own fit lock and live registry under `tmp`."""
    from pathlib import Path
    from .client import Client
    tmp = Path(tmp)
    t = MockTransport(**kw)
    c = Client(run_dir, host='mock', transport=t, sleep=lambda s: None,
               live_registry=tmp / 'live.json', fit_lock=tmp / 'fit.lock')
    return c, t


__all__ = ['MockTransport', 'MockModel', 'obs_heuristic', 'mock_client']
