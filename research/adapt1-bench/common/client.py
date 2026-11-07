"""Stdlib Adapt-1 client for Fast-UAT. Credentials are injected by the session proxy; none are held here.

Every request is journaled (method, path, body, status, seconds, response) to a JSONL file.
Reads retry on transient errors. Mutations retry only on 429/503 (not applied); any other
ambiguous failure raises Ambiguous so the runner can reconcile instead of replaying.
"""
import json, time, urllib.request, urllib.error, uuid, lzma
from pathlib import Path

BASE = 'https://rei-neuroadapt-fast-uat.reilabs.org/api/v1'
UA = 'adapt1-sci-runner/0.1'


class Ambiguous(RuntimeError):
    pass


class ApiError(RuntimeError):
    def __init__(self, code, body, path):
        super().__init__(f'{code} {path}: {body[:600]}')
        self.code, self.body, self.path = code, body, path


class Client:
    def __init__(self, journal_path=None, base=BASE, slim=None):
        self.base, self.calls, self.seconds, self.slim = base, 0, 0.0, slim
        if journal_path is None:
            self.journal = None
        elif str(journal_path).endswith('.xz'):
            self.journal = lzma.open(journal_path, 'at')
        else:                                      # plain JSONL, flushed per line: survives a container restart
            self.journal = open(journal_path, 'a', buffering=1)

    def _log(self, rec):
        if self.journal:
            self.journal.write(json.dumps(rec, separators=(',', ':')) + '\n')

    def close(self):
        if self.journal:
            self.journal.close()

    def call(self, method, path, body=None, mutation=None):
        mutation = (method != 'GET') if mutation is None else mutation
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        if data is not None and len(data) > 1024 * 1024:
            raise ValueError('body exceeds 1 MiB')
        last = None
        for attempt in range(12):       # 429/503 are "not applied": retry with backoff up to ~6 min (production returned 503 bursts under load)
            req = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={'Content-Type': 'application/json', 'User-Agent': UA})
            t0 = time.time()
            self.calls += 1
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    txt = r.read().decode() or '{}'
                    dt = time.time() - t0
                    self.seconds += dt
                    resp = json.loads(txt)
                    self._log({'t': t0, 'm': method, 'p': path, 'b': body, 's': r.status, 'dt': round(dt, 3), 'r': self.slim(path, resp) if self.slim else resp})
                    return resp
            except urllib.error.HTTPError as e:
                txt = e.read().decode(errors='replace')
                dt = time.time() - t0
                self._log({'t': t0, 'm': method, 'p': path, 'b': body, 's': e.code, 'dt': round(dt, 3), 'r': txt[:4000]})
                if e.code in (429, 503) or (e.code in (502, 504) and not mutation):
                    # 503 engine_queue_timeout carries Retry-After (5 s on Fast-UAT, 2026-11-07): never retry sooner than asked
                    try:
                        ra = float(e.headers.get('Retry-After') or 0)
                    except (TypeError, ValueError):
                        ra = 0.0
                    time.sleep(max(min(ra, 120), min(60, 2 ** attempt))); last = e; continue
                if e.code in (502, 504) and mutation:
                    raise Ambiguous(f'{method} {path}: HTTP {e.code} (outcome unknown)')
                raise ApiError(e.code, txt, path)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                self._log({'t': t0, 'm': method, 'p': path, 'b': body, 's': None, 'err': repr(e)})
                if mutation:
                    raise Ambiguous(f'{method} {path}: {e!r}')
                time.sleep(min(30, 2 ** attempt)); last = e
        raise RuntimeError(f'{method} {path}: retries exhausted ({last!r})')

    # convenience
    def post(self, path, body=None, mutation=True):
        return self.call('POST', path, {} if body is None else body, mutation=mutation)

    def traj(self, domain, op, body=None):
        return self.post(f'/domains/{domain}/trajectory/{op}', body,
                         mutation=op not in ('state', 'records', 'evaluate'))

    def create_domain(self, decl, attempts=8):
        """POST /domains, reconciling an ambiguous outcome (504, connection reset) by reading the Domain back: if it exists,
        the earlier attempt created it and its stored record is returned (it carries the same `learning` block). Fast-UAT
        returned 504 on two Domain creations on 2026-11-07 19:26 UTC while the Domains were in fact created."""
        did = decl['domain_id']
        for attempt in range(attempts):
            try:
                return self.post('/domains', decl)
            except (Ambiguous, ApiError) as e:
                if isinstance(e, ApiError) and e.code < 500 and not (e.code == 409 and attempt > 0):
                    raise
            time.sleep(min(60, 5 * (attempt + 1)))
            try:
                got = self.call('GET', f'/domains/{did}', None, mutation=False)
                if got:
                    return got
            except ApiError as g:
                if g.code != 404:
                    continue
        raise SystemExit(f'could not create Domain {did} after {attempts} attempts')

    def feedback(self, domain, body, stats, attempts=6):
        """POST one feedback record; if the outcome is ambiguous (connection reset, timeout, 502/504), reconcile before
        resending. Production probes on 2026-11-07 (scratchpad probes/): a resend with the same decision_id is accepted and
        applied a second time (feedback_policy.sample_count 4 -> 5), and GET /domains/{id} event_count rises by exactly one
        per feedback and not per query. So the client keeps the Domain's expected event count, re-reads it after an
        ambiguous POST, and resends only if the record did not land. Returns the service response, or
        {'reconciled': 'applied'} when the record is known to have landed but its response was lost."""
        if not hasattr(self, '_ev'):
            self._ev = {}
        if domain not in self._ev:
            self._ev[domain] = self.call('GET', f'/domains/{domain}', None, mutation=False).get('event_count')
        for attempt in range(attempts):
            try:
                r = self.post(f'/domains/{domain}/feedback', body)
                if self._ev[domain] is not None:
                    self._ev[domain] += 1
                return r
            except (Ambiguous, ApiError) as e:
                if isinstance(e, ApiError) and e.code < 500:
                    raise
                stats['feedback_ambiguous'] = stats.get('feedback_ambiguous', 0) + 1
            time.sleep(min(60, 5 * (attempt + 1)))
            n = self.call('GET', f'/domains/{domain}', None, mutation=False).get('event_count')   # reads retry on their own
            exp = self._ev[domain]
            if exp is not None and n == exp + 1:
                self._ev[domain] = n; stats['feedback_landed_unacked'] = stats.get('feedback_landed_unacked', 0) + 1
                return {'reconciled': 'applied'}
            if exp is not None and n != exp:
                stats['feedback_count_mismatch'] = stats.get('feedback_count_mismatch', 0) + 1
                raise SystemExit(f'feedback reconciliation: event_count {n}, expected {exp} or {exp + 1}; writer paused')
            stats['feedback_resent'] = stats.get('feedback_resent', 0) + 1
        raise SystemExit(f'feedback to {domain} failed {attempts} times')

    def delete_domain(self, domain, attempts=5):
        """Delete and confirm by readback. A 5xx on the DELETE is ambiguous: re-read, and if the Domain is still there retry
        (production returned a 502 on a delete during the XLand smoke test)."""
        r = None
        for attempt in range(attempts):
            try:
                r = self.call('DELETE', f'/domains/{domain}', None, mutation=True)
            except (Ambiguous, ApiError) as e:
                r = {'error': repr(e)[:200]}
                if isinstance(e, ApiError) and e.code == 404:
                    return {'deleted': r, 'readback': 404}
            try:
                self.call('GET', f'/domains/{domain}', None, mutation=False)
            except ApiError as e:
                if e.code == 404:
                    return {'deleted': r, 'readback': 404}
            except Exception:
                pass
            time.sleep(min(30, 2 ** attempt))
        return {'deleted': r, 'readback': 'still_present'}


def new_id(prefix):
    return f"{prefix}-{time.strftime('%Y%m%d%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}"
