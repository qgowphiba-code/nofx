"""Run every command and Python snippet in HOWTO.md, in order, in a fresh copy of the package.

usage: python scripts/check_howto.py [--keep-tmp]

- bash blocks run in one shell session (so `cd` and `source .venv/bin/activate` carry over), with `set -e`;
- python blocks whose first line is `# <name>.py` are saved under that name; other python blocks are run as
  scripts from the package folder;
- blocks preceded by `<!-- check: live -->` run against the in-memory fake API (MACHINA_FINANCE_FAKE=1, with
  state kept across commands); no request reaches the real API;
- blocks preceded by `<!-- check: skip ... -->` (interactive key entry) are not run.
Needs network access to PyPI for the install steps.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FENCE = re.compile(r"```(bash|python)\n(.*?)```", re.S)
MARK = re.compile(r"<!--\s*check:\s*(live|skip)[^>]*-->\s*$")


def blocks(md):
    out = []
    for m in FENCE.finditer(md):
        before = md[:m.start()].rstrip().splitlines()
        mark = MARK.search(before[-1]) if before else None
        out.append((m.group(1), m.group(2), mark.group(1) if mark else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-tmp", action="store_true")
    a = ap.parse_args()
    md = open(os.path.join(ROOT, "HOWTO.md")).read()
    tmp = tempfile.mkdtemp(prefix="howto-check-")
    pkg = os.path.join(tmp, "research", "machina-finance")
    shutil.copytree(ROOT, pkg, ignore=shutil.ignore_patterns("runs", ".venv", "__pycache__", ".pytest_cache", "*.egg-info",
                                                            "build", "*.pkl"))
    state = os.path.join(tmp, "fake_state.pkl")
    lines = ["set -e -o pipefail", f"cd {tmp}", "unset ADAPT1_API_KEY MACHINA_FINANCE_FAKE MACHINA_FINANCE_FAKE_STATE"]
    n_run = n_skip = 0
    for i, (lang, body, mark) in enumerate(blocks(md), 1):
        if mark == "skip":
            n_skip += 1
            lines.append(f"echo '### block {i}: skipped (interactive)'")
            continue
        if lang == "python":
            first = body.splitlines()[0].strip()
            named = re.fullmatch(r"#\s*([\w.-]+\.py)", first)
            fname = named.group(1) if named else f".howto_block_{i}.py"
            with open(os.path.join(pkg, fname), "w") as f:
                f.write(body)
            if named:
                lines.append(f"echo '### block {i}: saved {fname}'")
                continue
            cmd = f"python {fname}"
        else:
            cmd = body.strip()
        n_run += 1
        shown = cmd.splitlines()[0][:90].replace("'", "")
        lines.append(f"echo '### block {i} ({lang}{', fake API' if mark == 'live' else ''}): {shown}'")
        if mark == "live":
            lines.append(f"export MACHINA_FINANCE_FAKE=1 MACHINA_FINANCE_FAKE_STATE={state}")
        lines.append(cmd)
        if mark == "live":
            lines.append("unset MACHINA_FINANCE_FAKE MACHINA_FINANCE_FAKE_STATE")
    script = os.path.join(tmp, "howto.sh")
    with open(script, "w") as f:
        f.write("\n".join(lines) + "\n")
    env = {k: v for k, v in os.environ.items() if k not in ("ADAPT1_API_KEY", "VIRTUAL_ENV", "PYTHONPATH")}
    r = subprocess.run(["bash", script], env=env, text=True, capture_output=True)
    log = os.path.join(tmp, "howto.log")
    with open(log, "w") as f:
        f.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    print(r.stdout[-6000:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
    print(f"{'PASS' if r.returncode == 0 else 'FAIL'}: {n_run} blocks run, {n_skip} skipped; log {log}")
    if not a.keep_tmp and r.returncode == 0:
        shutil.rmtree(tmp)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
