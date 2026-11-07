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
        for attempt in range(6):
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
                    time.sleep(min(30, 2 ** attempt)); last = e; continue
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

    def delete_domain(self, domain):
        r = self.call('DELETE', f'/domains/{domain}', None, mutation=True)
        try:
            self.call('GET', f'/domains/{domain}', None, mutation=False)
            return {'deleted': r, 'readback': 'still_present'}
        except ApiError as e:
            return {'deleted': r, 'readback': e.code}


def new_id(prefix):
    return f"{prefix}-{time.strftime('%Y%m%d%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}"
