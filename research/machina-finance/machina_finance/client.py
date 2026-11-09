"""Adapt-1 trajectory API client (standard library only).

Auth: the documented header is `Authorization: Bearer <ADAPT1_API_KEY>`. The key is read from the
environment and never logged; header name and scheme can be overridden with ADAPT1_AUTH_HEADER and
ADAPT1_AUTH_SCHEME. If no key is set, no auth header is sent (for environments where a network proxy
injects it). Base URL: ADAPT1_BASE_URL, default the production API.

Every request and response body is appended to a JSONL journal (headers are never journaled).
Retries: HTTP 429 is always retried with backoff (rejected before execution); reads also retry on
network errors and 502/503/504; a mutation that fails ambiguously raises AmbiguousWrite instead of
being repeated. HTTP 402 raises CreditsOut. An optional call cap raises BudgetExceeded.
"""
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "https://rei-neuroadapt-api.reilabs.org/api/v1"
UA = "machina-finance/0.1"
READ_OPS = {"state", "records", "evaluate"}


class APIError(Exception):
    def __init__(self, status, body, path):
        super().__init__(f"HTTP {status} on {path}: {str(body)[:400]}")
        self.status, self.body, self.path = status, body, path


class CreditsOut(APIError):
    """HTTP 402: the key has no query credits left (or the plan is inactive)."""


class AmbiguousWrite(Exception):
    """A mutation may or may not have been applied (network error or 5xx); do not blindly repeat it."""


class BudgetExceeded(Exception):
    pass


def urllib_transport(method, url, headers, data, timeout):
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


class Client:
    def __init__(self, journal_path=None, base=None, timeout=180, max_calls=None, cleanup_reserve=20,
                 ledger_path=None, transport=None, sleep=time.sleep):
        self.base = (base or os.environ.get("ADAPT1_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.journal_path = journal_path
        self.ledger_path = ledger_path
        self.timeout = timeout
        self.max_calls = max_calls
        self.cleanup_reserve = cleanup_reserve
        self.transport = transport or urllib_transport
        self.sleep = sleep
        self.calls = 0
        self.n429 = 0
        self._lock = threading.Lock()
        for p in (journal_path, ledger_path):
            if p:
                os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # -- plumbing ----------------------------------------------------------------
    def _headers(self, has_body):
        h = {"User-Agent": UA, "Accept": "application/json"}
        if has_body:
            h["Content-Type"] = "application/json"
        key = os.environ.get("ADAPT1_API_KEY")
        if key:
            scheme = os.environ.get("ADAPT1_AUTH_SCHEME", "Bearer")
            h[os.environ.get("ADAPT1_AUTH_HEADER", "Authorization")] = f"{scheme} {key}".strip()
        return h

    def _journal(self, rec):
        if not self.journal_path:
            return
        rec["ts"] = time.time()
        with self._lock, open(self.journal_path, "a") as f:
            f.write(json.dumps(rec, allow_nan=False) + "\n")

    def _ledger(self, domain, event):
        if self.ledger_path:
            with self._lock, open(self.ledger_path, "a") as f:
                f.write(json.dumps({"domain_id": domain, "event": event, "ts": time.time()}) + "\n")

    def _take_call(self, cleanup):
        with self._lock:
            if self.max_calls is not None:
                limit = self.max_calls if cleanup else self.max_calls - self.cleanup_reserve
                if self.calls >= limit:
                    raise BudgetExceeded(f"call cap reached ({self.calls} of {self.max_calls})")
            self.calls += 1

    def _raw(self, method, path, body, cleanup):
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        if data is not None and len(data) > 1024 * 1024:
            raise ValueError("request body exceeds 1 MiB")
        self._take_call(cleanup)
        status, txt = self.transport(method, self.base + path, self._headers(data is not None), data, self.timeout)
        try:
            out = json.loads(txt) if txt else None
        except json.JSONDecodeError:
            out = {"_raw": txt[:2000]}
        return status, out

    def request(self, method, path, body=None, read=False, max_tries=8, cleanup=False):
        delay, n429, attempt = 1.0, 0, -1
        while True:
            attempt += 1
            if attempt - n429 >= max_tries or n429 >= 60:
                raise APIError(-1, "retries exhausted", path)
            try:
                status, out = self._raw(method, path, body, cleanup)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                self._journal({"method": method, "path": path, "body": body, "error": repr(e), "attempt": attempt})
                if read:
                    self.sleep(delay + random.random())
                    delay = min(delay * 2, 30)
                    continue
                raise AmbiguousWrite(f"{method} {path}: {e!r}")
            self._journal({"method": method, "path": path, "body": body, "status": status, "resp": out,
                           "attempt": attempt})
            if status in (200, 201):
                return out
            if status == 429:
                n429 += 1
                with self._lock:
                    self.n429 += 1
                self.sleep(min(2.0 * n429, 20.0) * (0.5 + random.random()))
                continue
            if status in (502, 503, 504):
                if read:
                    self.sleep(delay + random.random())
                    delay = min(delay * 2, 30)
                    continue
                raise AmbiguousWrite(f"{method} {path}: HTTP {status} {str(out)[:300]}")
            if status == 402:
                raise CreditsOut(status, out, path)
            raise APIError(status, out, path)

    # -- API -----------------------------------------------------------------------
    def traj(self, domain, op, body=None):
        return self.request("POST", f"/domains/{domain}/trajectory/{op}", {} if body is None else body,
                            read=op in READ_OPS)

    def list_domains(self):
        return self.request("GET", "/domains", read=True, cleanup=True)

    def credits_ok(self):
        """Cheap probe: state of a Domain that does not exist. 404 -> credits OK, 402 -> no credits."""
        try:
            self.traj("finm-probe-none", "state")
        except CreditsOut:
            return False
        except APIError as e:
            if e.status == 404:
                return True
            raise
        return True

    def create_stage(self, domain, config, description="", wait_for_slot=1200):
        """Create a Domain and configure its trajectory learner (owners may hold at most 32 learners)."""
        self._ledger(domain, "create_intent")
        body = {"domain_id": domain}
        if description:
            body["description"] = description
        self.request("POST", "/domains", body)
        st = self.traj(domain, "state")
        if st.get("status") != "not_configured":
            raise RuntimeError(f"{domain}: expected a fresh trajectory learner, got {st}")
        t0 = time.time()
        while True:
            try:
                receipt = self.traj(domain, "configure", {"config": config})
                break
            except APIError as e:
                if e.status == 409 and "limit" in str(e.body) and time.time() - t0 < wait_for_slot:
                    self.sleep(20 + random.random() * 5)
                    continue
                raise
        st2 = self.traj(domain, "state")
        if st2.get("status") == "not_configured":
            raise RuntimeError(f"{domain}: configuration not applied")
        return receipt, st2

    def delete_domain(self, domain, missing_ok=True):
        try:
            out = self.request("DELETE", f"/domains/{domain}", cleanup=True)
        except APIError as e:
            if not (missing_ok and e.status == 404):
                raise
            out = None
        self._ledger(domain, "deleted")
        return out
