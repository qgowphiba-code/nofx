"""Run directories, logs and manifests (plan section 13: manifest.json, results.json, log.txt)."""
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'runs'


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def git_head():
    try:
        return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


class Run:
    def __init__(self, exp, name=None):
        self.exp = exp
        self.dir = RUNS / exp / (name or stamp())
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.manifest = {'experiment': exp, 'started_utc': stamp(), 'git_head': git_head(),
                         'python': sys.version.split()[0], 'platform': platform.platform()}
        self.log(f'run {exp} -> {self.dir}')

    def log(self, *msg):
        from .client import redact_text
        line = redact_text(f'[{time.time() - self.t0:8.1f}s] ' + ' '.join(str(m) for m in msg))
        print(line, flush=True)
        with open(self.dir / 'log.txt', 'a') as f:
            f.write(line + '\n')

    def note(self, **kw):
        self.manifest.update(kw)
        self.write_manifest()

    def write_manifest(self):
        (self.dir / 'manifest.json').write_text(json.dumps(self.manifest, indent=1, sort_keys=True, default=str))

    def write_results(self, results, name='results.json'):
        from .client import redact
        (self.dir / name).write_text(json.dumps(redact(results), indent=1, sort_keys=True, default=str))
        return self.dir / name

    def finish(self, **kw):
        self.manifest.update(kw)
        self.manifest['finished_utc'] = stamp()
        self.manifest['seconds'] = round(time.time() - self.t0, 1)
        self.write_manifest()
