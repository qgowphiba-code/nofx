"""Import the unmodified set-1 modules (plan/analysis/vendor) and the plan's offline helpers."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / 'plan' / 'analysis' / 'vendor'
ANALYSIS = ROOT / 'plan' / 'analysis'
LIU_PANEL = ANALYSIS / 'data' / 'liu_panel.json'
for p in (str(VENDOR), str(ANALYSIS)):
    if p not in sys.path:
        sys.path.append(p)

import lattice_env  # noqa: E402
import hp  # noqa: E402


def liu_panel():
    import json
    return [(c['case_id'].replace('liu2025_', ''), c['sequence']) for c in json.loads(LIU_PANEL.read_text())['cases']]


def grading():
    """Set 1's Stage A oracle grading set (builds in ~25 s)."""
    import grading as g
    return g
