"""C-run: safe Adapt-1 client (plan section 2.4).

What it guarantees for every experiment script:
- a write-ahead journal: an intent line before each request and a receipt line after it;
- operation classes with the documented retry rules (docs: operational-behavior-and-retries):
    READ      value/state, value/predict, GET routes, planning -> bounded retry with backoff;
    OBSERVE   value/observe -> resend the identical body; check accepted + duplicates == rows;
    WRITE     domain create/delete, configure, train, promote -> on an ambiguous failure,
              reconcile against state before any resend;
- a STOP file checked before every request (run dir or package root);
- at most 2 requests in flight per process (plan 2.4; 6 streams overran af6670b);
- an owner-wide fit lock (flock) held from train submission until the fit finishes;
- a build check: GET /version at start and end, abort if it changes;
- payload guards: compact JSON, bodies <= 900 KiB, observe <= 50 rows, predict <= 256 inputs;
- identity checks: predict/planning bind expected_model_sha256; returned identity is checked;
- domain hygiene: only ids with the hpx2- prefix are ever addressed; at most 4 live.

No credential is handled here: the environment's egress proxy injects the bearer token.
"""
import fcntl
import hashlib
import json
import os
import random
import re
import secrets
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

HOSTS = {
    'fast-uat': 'https://rei-neuroadapt-fast-uat.reilabs.org',
    'production': 'https://rei-neuroadapt-api.reilabs.org',
}
PREFIX = 'hpx2-'
MAX_BODY = 900 * 1024
MAX_OBSERVE = 50
MAX_PREDICT = 256
MAX_LIVE_DOMAINS = 4
TRANSIENT = {429, 502, 503, 504}
USER_AGENT = 'hpx2-client/0.1 (research; HP x Adapt-1 set 2)'
PKG_ROOT = Path(__file__).resolve().parent.parent
FIT_LOCK = Path(os.environ.get('HPX2_FIT_LOCK', Path.home() / '.hpx2-fit.lock'))
LIVE_REGISTRY = PKG_ROOT / 'runs' / 'live_domains.json'
# observed on af6670b: install=true ends 'trained', install=false ends 'candidate'; anything else is a failure
FINISHED_OK = {True: 'trained', False: 'candidate'}
RUNNING = {'running', 'queued', 'pending', 'started', 'accepted', 'training', ''}


def compact(obj):
    return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)


def sha256_text(text):
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def utc_stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


REDACT_KEYS = {'session_identity'}
_REDACT_RE = re.compile(r"""(session_id['"]?\s*[:=]\s*['"])[^'"]+""")


def redact_text(text):
    return _REDACT_RE.sub(r'\1<redacted>', text)


def redact(obj):
    """Drop the account's effective session identity from anything written to disk."""
    if isinstance(obj, dict):
        return {str(k): ('<redacted>' if isinstance(k, str) and (k in REDACT_KEYS or k.endswith('session_id')) else redact(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


class ApiError(Exception):
    def __init__(self, status, body, path=''):
        super().__init__(f'HTTP {status} on {path}: {str(body)[:500]}')
        self.status, self.body, self.path = status, body, path


class Ambiguous(ApiError):
    """A write whose effect is unknown (timeout, connection loss, 502/503/504)."""


class Stopped(RuntimeError):
    pass


class BuildChanged(RuntimeError):
    pass


class GuardError(RuntimeError):
    """A plan-section-2.4 guard failed (identity, sha change, admission count, payload)."""


class HttpTransport:
    def __init__(self, base, timeout=180):
        self.base, self.timeout = base, timeout

    def request(self, method, path, body_bytes=None):
        headers = {'User-Agent': USER_AGENT, 'Accept': 'application/json'}
        if body_bytes is not None:
            headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(self.base + path, data=body_bytes, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw, status, hdrs = r.read(), r.status, dict(r.headers)
        except urllib.error.HTTPError as e:
            raw, status, hdrs = e.read(), e.code, dict(e.headers or {})
        try:
            data = json.loads(raw) if raw else None
        except ValueError:
            data = raw.decode(errors='replace')
        return status, data, hdrs


class Client:
    def __init__(self, run_dir, host='fast-uat', transport=None, max_in_flight=2, timeout=180,
                 max_attempts=6, sleep=time.sleep, live_registry=LIVE_REGISTRY, fit_lock=FIT_LOCK):
        self.host = host
        self.base = HOSTS.get(host, host)
        self.transport = transport or HttpTransport(self.base, timeout)
        self.run_dir = Path(run_dir)
        (self.run_dir / 'responses').mkdir(parents=True, exist_ok=True)
        self.journal_path = self.run_dir / 'journal.jsonl'
        self._jlock = threading.Lock()
        self._sem = threading.BoundedSemaphore(max_in_flight)
        self.max_in_flight = max_in_flight
        self._seq = 0
        self.max_attempts = max_attempts
        self.sleep = sleep
        self.live_registry = Path(live_registry)
        self.fit_lock_path = Path(fit_lock)
        self.build = None
        self.counts = {'requests': 0, 'retries': 0, 'by_route': {}}

    # ---------------------------------------------------------------- journal and guards
    def _journal(self, rec):
        rec = {'t': round(time.time(), 3), **rec}
        with self._jlock:
            with open(self.journal_path, 'a') as f:
                f.write(compact(rec) + '\n')

    def _next_id(self):
        with self._jlock:
            self._seq += 1
            return self._seq

    def check_stop(self):
        for p in (self.run_dir / 'STOP', PKG_ROOT / 'STOP'):
            if p.exists():
                raise Stopped(f'STOP file present: {p}')

    def save(self, name, obj):
        path = self.run_dir / 'responses' / f'{name}.json'
        path.write_text(json.dumps(redact(obj), indent=1, sort_keys=True))
        return path

    @staticmethod
    def _check_domain(domain_id):
        if not isinstance(domain_id, str) or not domain_id.startswith(PREFIX):
            raise GuardError(f'refusing to address a domain without the {PREFIX} prefix: {domain_id!r}')

    @staticmethod
    def _route(path):
        parts = path.split('?')[0].strip('/').split('/')
        if len(parts) >= 4 and parts[2] == 'domains':
            return '/'.join(['domains', '{id}'] + parts[4:])
        return '/'.join(parts)

    # ---------------------------------------------------------------- one HTTP exchange
    def _exchange(self, method, path, body, op, size_guard=True):
        """One attempt. Returns (status, data, headers) or raises Ambiguous on a transport failure."""
        self.check_stop()
        raw = None if body is None else compact(body).encode()
        if size_guard and raw is not None and len(raw) > MAX_BODY:
            raise GuardError(f'body {len(raw)} B exceeds {MAX_BODY} B on {path}')
        rid = self._next_id()
        self._journal({'kind': 'intent', 'id': rid, 'op': op, 'method': method, 'path': path,
                       'bytes': 0 if raw is None else len(raw), 'body_sha256': None if raw is None else sha256_text(raw)})
        t0 = time.monotonic()
        try:
            with self._sem:
                status, data, hdrs = self.transport.request(method, path, raw)
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
            self._journal({'kind': 'receipt', 'id': rid, 'status': None, 'error': repr(e)[:300],
                           'elapsed': round(time.monotonic() - t0, 3)})
            raise Ambiguous(None, repr(e), path)
        route = self._route(path)
        with self._jlock:
            self.counts['requests'] += 1
            self.counts['by_route'][route] = self.counts['by_route'].get(route, 0) + 1
        summary = None
        if status >= 300:
            summary = str(data)[:600]
        self._journal({'kind': 'receipt', 'id': rid, 'status': status, 'elapsed': round(time.monotonic() - t0, 3),
                       **({'error_body': summary} if summary else {})})
        return status, data, hdrs

    def _backoff(self, attempt, hdrs=None):
        ra = (hdrs or {}).get('Retry-After') or (hdrs or {}).get('retry-after')
        try:
            wait = float(ra) if ra is not None else None
        except ValueError:
            wait = None
        if wait is None:
            wait = min(8.0, 0.5 * 2 ** attempt) * (0.75 + 0.5 * random.random())
        with self._jlock:
            self.counts['retries'] += 1
        self.sleep(wait)

    def read(self, method, path, body=None, ok=(200,), op='READ'):
        """READ class: retry transient failures and transport errors with bounded backoff."""
        last = None
        for attempt in range(self.max_attempts):
            try:
                status, data, hdrs = self._exchange(method, path, body, op)
            except Ambiguous as e:
                last = e
                self._backoff(attempt)
                continue
            if status in TRANSIENT or (isinstance(data, dict) and 'engine_execution_overrun' in str(data)[:2000]):
                last = ApiError(status, data, path)
                self._backoff(attempt, hdrs)
                continue
            if status not in ok:
                raise ApiError(status, data, path)
            return data
        raise last

    def write(self, method, path, body, reconcile, ok=(200,), op='WRITE'):
        """WRITE class: one send; after an ambiguous failure, reconcile() decides whether it applied.

        reconcile() returns (applied: bool, result). 429 is a definite non-application and is retried."""
        last = None
        for attempt in range(self.max_attempts):
            try:
                status, data, hdrs = self._exchange(method, path, body, op)
            except Ambiguous as e:
                last = e
                self._backoff(attempt)
                applied, result = reconcile()
                self._journal({'kind': 'reconcile', 'path': path, 'applied': applied})
                if applied:
                    return result
                continue
            if status == 429:
                last = ApiError(status, data, path)
                self._backoff(attempt, hdrs)
                continue
            if status in TRANSIENT:
                last = Ambiguous(status, data, path)
                self._backoff(attempt, hdrs)
                applied, result = reconcile()
                self._journal({'kind': 'reconcile', 'path': path, 'applied': applied})
                if applied:
                    return result
                continue
            if status not in ok:
                raise ApiError(status, data, path)
            return data
        raise last

    # ---------------------------------------------------------------- build check
    def version(self):
        return self.read('GET', '/version')

    def check_build(self, stage='start'):
        v = self.version()
        commit = v.get('commit_sha')
        self._journal({'kind': 'build', 'stage': stage, 'version': v})
        if self.build is None:
            self.build = commit
        elif commit != self.build:
            raise BuildChanged(f'build changed mid-run: {self.build} -> {commit}')
        return v

    # ---------------------------------------------------------------- domains and hygiene
    def _registry(self):
        try:
            return json.loads(self.live_registry.read_text())
        except (FileNotFoundError, ValueError):
            return []

    @contextmanager
    def _registry_lock(self):
        self.live_registry.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.live_registry) + '.lock', 'w') as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    @staticmethod
    def domain_name(exp, arm, lineage):
        name = f'{PREFIX}{exp}-{arm}-{lineage}-{utc_stamp()}-{secrets.token_hex(3)}'.lower()
        assert len(name) <= 128
        return name

    def get_domain(self, domain_id):
        """Returns the domain record, or None on 404."""
        self._check_domain(domain_id)
        for attempt in range(self.max_attempts):
            try:
                status, data, hdrs = self._exchange('GET', f'/api/v1/domains/{domain_id}', None, 'READ')
            except Ambiguous:
                self._backoff(attempt)
                continue
            if status == 404:
                return None
            if status in TRANSIENT:
                self._backoff(attempt, hdrs)
                continue
            if status != 200:
                raise ApiError(status, data, f'/api/v1/domains/{domain_id}')
            return data
        raise Ambiguous(None, 'get_domain retries exhausted', domain_id)

    def create_domain(self, exp, arm, lineage, description=''):
        domain_id = self.domain_name(exp, arm, lineage)
        with self._registry_lock():
            live = self._registry()
            if len(live) >= MAX_LIVE_DOMAINS:
                raise GuardError(f'{len(live)} live hpx2 domains already (limit {MAX_LIVE_DOMAINS}): {live}')
            live.append(domain_id)
            self.live_registry.write_text(json.dumps(live, indent=1))
        body = {'domain_id': domain_id, 'description': description or f'HP x Adapt-1 set 2: {exp}/{arm}/{lineage}',
                'schema': {}, 'hypotheses': [], 'query_templates': {}}
        resp = self.write('POST', '/api/v1/domains', body, ok=(200, 201),
                          reconcile=lambda: (self.get_domain(domain_id) is not None, {'domain_id': domain_id}))
        self._journal({'kind': 'domain_created', 'domain_id': domain_id})
        self.save(f'domain-create-{domain_id}', resp)
        return domain_id

    def delete_domain(self, domain_id):
        self._check_domain(domain_id)
        path = f'/api/v1/domains/{domain_id}'
        resp = self.write('DELETE', path, None, ok=(200, 204, 404),
                          reconcile=lambda: (self.get_domain(domain_id) is None, {'reconciled': True}))
        if self.get_domain(domain_id) is not None:
            raise GuardError(f'domain {domain_id} still present after delete')
        with self._registry_lock():
            live = [d for d in self._registry() if d != domain_id]
            self.live_registry.write_text(json.dumps(live, indent=1))
        self._journal({'kind': 'domain_deleted', 'domain_id': domain_id, 'verified_404': True})
        return resp

    def _v(self, domain_id, op):
        self._check_domain(domain_id)
        return f'/api/v1/domains/{domain_id}/value/{op}'

    # ---------------------------------------------------------------- value routes
    def state(self, domain_id):
        return self.read('POST', self._v(domain_id, 'state'), {'session_id': 'ignored'})

    def configure(self, domain_id, config):
        body = {'session_id': 'ignored', 'config': config}

        def reconcile():
            st = self.state(domain_id)
            cfg = (st or {}).get('config') or {}
            return (all(cfg.get(k) == v for k, v in config.items() if k != 'promotion'), st)
        resp = self.write('POST', self._v(domain_id, 'configure'), body, reconcile=reconcile)
        self.save(f'configure-{domain_id}', resp)
        return resp

    def observe(self, domain_id, rows):
        """Upload rows in chunks of <= 50 (and <= 900 KiB). Each chunk is an exact-replay-safe OBSERVE."""
        total = {'submitted': 0, 'accepted': 0, 'duplicates': 0, 'target_sources': {}}
        for chunk in self._chunks(rows, MAX_OBSERVE, lambda r: len(compact(r)), overhead=64):
            body = {'session_id': 'ignored', 'observations': chunk}
            data = self.read('POST', self._v(domain_id, 'observe'), body, op='OBSERVE')
            acc, dup = int(data.get('accepted', 0)), int(data.get('duplicates', 0))
            if acc + dup != len(chunk):
                self.save(f'observe-mismatch-{domain_id}-{self._seq}', data)
                raise GuardError(f'observe admission mismatch: accepted {acc} + duplicates {dup} != {len(chunk)}')
            total['submitted'] += len(chunk)
            total['accepted'] += acc
            total['duplicates'] += dup
            for k, v in (data.get('target_sources') or {}).items():
                if isinstance(v, (int, float)):
                    total['target_sources'][k] = total['target_sources'].get(k, 0) + v
            total['last'] = data
        return total

    @staticmethod
    def _chunks(items, max_count, size_of, overhead=256):
        chunk, size = [], overhead
        for it in items:
            s = size_of(it) + 1
            if s + overhead > MAX_BODY:
                raise GuardError(f'single item of {s} B cannot fit a {MAX_BODY} B body')
            if chunk and (len(chunk) >= max_count or size + s > MAX_BODY):
                yield chunk
                chunk, size = [], overhead
            chunk.append(it)
            size += s
        if chunk:
            yield chunk

    @staticmethod
    def model_sha(state, source):
        """sha256 of the installed model ('installed') or the candidate ('candidate') from value/state."""
        node = (state or {}).get('model' if source == 'installed' else 'candidate') or {}
        if isinstance(node, dict):
            for k in ('sha256', 'model_sha256', 'parameters_sha256'):
                if isinstance(node.get(k), str) and len(node[k]) == 64:
                    return node[k]
        return None

    @contextmanager
    def fit_lock(self):
        self.fit_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.fit_lock_path, 'w') as lf:
            t0 = time.monotonic()
            fcntl.flock(lf, fcntl.LOCK_EX)
            self._journal({'kind': 'fit_lock', 'waited': round(time.monotonic() - t0, 1)})
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def train(self, domain_id, request_id, options, install, warm_start_source=None,
              expected_initial_model_sha256=None, observation_ids=None, poll_seconds=5.0,
              max_wait=7200, require_new_candidate=True, name=None):
        """Submit a fit, wait for it, run the post-fit checks, save the full report. Returns final state."""
        body = {'session_id': 'ignored', 'request_id': request_id, 'install': bool(install), 'options': options}
        if warm_start_source is not None:
            body['warm_start_source'] = warm_start_source
        if expected_initial_model_sha256 is not None:
            body['expected_initial_model_sha256'] = expected_initial_model_sha256
        if observation_ids is not None:
            if not 4 <= len(observation_ids) <= 20000:
                raise GuardError(f'observation_ids has {len(observation_ids)} ids (allowed 4..20000)')
            body['observation_ids'] = list(observation_ids)
        with self.fit_lock():
            before = self.state(domain_id)
            prev = {'installed': self.model_sha(before, 'installed'), 'candidate': self.model_sha(before, 'candidate')}

            def reconcile():
                st = self.state(domain_id)
                return ((st.get('training') or {}).get('request_id') == request_id, st)
            t0 = time.monotonic()
            ack = self.write('POST', self._v(domain_id, 'train'), body, reconcile=reconcile, ok=(200, 202))
            self._journal({'kind': 'train_submitted', 'domain_id': domain_id, 'request_id': request_id})
            while True:
                self.sleep(poll_seconds)
                st = self.state(domain_id)
                tr = st.get('training') or {}
                status = str(tr.get('status', '')).lower()
                if tr.get('request_id') == request_id and status not in RUNNING:
                    break
                if time.monotonic() - t0 > max_wait:
                    raise GuardError(f'fit {request_id} not finished after {max_wait}s (status {status})')
        elapsed = round(time.monotonic() - t0, 1)
        self.save(f'train-{name or request_id}', {'request': {k: v for k, v in body.items() if k != 'observation_ids'},
                                                  'n_observation_ids': len(observation_ids or []),
                                                  'ack': ack, 'final_state': st, 'before': before, 'elapsed': elapsed})
        if status != FINISHED_OK[bool(install)]:
            raise GuardError(f'fit {request_id} ended with status {status}: {str(tr)[:800]}')
        if bool(tr.get('installed')) != bool(install):
            raise GuardError(f'fit {request_id}: install={install} requested but training.installed={tr.get("installed")}')
        after = {'installed': self.model_sha(st, 'installed'), 'candidate': self.model_sha(st, 'candidate')}
        new = after['installed'] if install else after['candidate']
        if new is None:
            raise GuardError(f'fit {request_id}: no {"installed" if install else "candidate"} sha in state')
        slot = 'installed' if install else 'candidate'
        if require_new_candidate and new == prev[slot]:
            raise GuardError(f'fit {request_id}: {slot} sha {new[:12]} unchanged from the previous {slot}')
        self._journal({'kind': 'fit_done', 'request_id': request_id, 'elapsed': elapsed, 'sha': new, 'prev': prev})
        return {'state': st, 'sha256': new, 'previous': prev, 'elapsed': elapsed}

    def predict(self, domain_id, inputs, model_source, expected_model_sha256, workers=None):
        """Predict for any number of inputs: chunks of <= 256 inputs and <= 900 KiB, <= 2 in flight."""
        if model_source not in ('installed', 'candidate'):
            raise ValueError(model_source)
        if not expected_model_sha256:
            raise GuardError('predict requires expected_model_sha256 (plan 2.4 identity check)')
        sizes = [len(compact(x)) for x in inputs]
        idx = list(range(len(inputs)))
        chunks = list(self._chunks(idx, MAX_PREDICT, lambda i: sizes[i], overhead=200))

        def run(chunk):
            body = {'session_id': 'ignored', 'model_source': model_source,
                    'expected_model_sha256': expected_model_sha256, 'inputs': [inputs[i] for i in chunk]}
            data = self.read('POST', self._v(domain_id, 'predict'), body)
            preds = data.get('predictions') if isinstance(data, dict) else None
            if not isinstance(preds, list) or len(preds) != len(chunk):
                raise GuardError(f'predict returned {type(preds).__name__} of {len(preds or [])} for {len(chunk)} inputs')
            got = self.response_model_sha(data)
            if got is not None and got != expected_model_sha256:
                raise GuardError(f'predict served model {got[:12]}, expected {expected_model_sha256[:12]}')
            return [float(p['value']) if isinstance(p, dict) else float(p) for p in preds]
        out = [None] * len(inputs)
        if len(chunks) == 1 or (workers or self.max_in_flight) == 1:
            results = [run(c) for c in chunks]
        else:
            with ThreadPoolExecutor(max_workers=workers or self.max_in_flight) as ex:
                results = list(ex.map(run, chunks))
        for chunk, vals in zip(chunks, results):
            for i, v in zip(chunk, vals):
                out[i] = v
        return out

    @staticmethod
    def response_model_sha(data):
        if not isinstance(data, dict):
            return None
        ev = data.get('evaluator')
        if isinstance(ev, dict) and isinstance(ev.get('model_sha256'), str):
            return ev['model_sha256']
        for k in ('model_sha256', 'evaluated_model_sha256'):
            if isinstance(data.get(k), str) and len(data[k]) == 64:
                return data[k]
        model = data.get('model')
        if isinstance(model, dict):
            for k in ('sha256', 'model_sha256'):
                if isinstance(model.get(k), str) and len(model[k]) == 64:
                    return model[k]
        return None

    def plan(self, domain_id, body):
        """Native planning (READ class: it does not change value state). Caller builds the graph body."""
        self._check_domain(domain_id)
        b = {'session_id': 'ignored', **body}
        return self.read('POST', f'/api/v1/domains/{domain_id}/planning/adversarial', b)

    def raw(self, method, path, body=None, size_guard=True):
        """Unretried single exchange, for limit probes (T0.2). Returns (status, data)."""
        status, data, _ = self._exchange(method, path, body, 'PROBE', size_guard=size_guard)
        return status, data
