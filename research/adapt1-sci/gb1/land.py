"""GB1 four-site combinatorial fitness landscape (Wu et al., eLife 2016; positions V39, D40, G41, V54), FLIP copy.
149,361 measured variants of 160,000; an unmeasured variant counts as a failed experiment (fitness 0). Wild type VDGV = 1.0."""
import csv
from pathlib import Path
AA = 'ACDEFGHIKLMNPQRSTVWY'
F = {r['Variants']: float(r['Fitness']) for r in csv.DictReader(open(Path(__file__).resolve().parents[1] / 'data/raw/gb1/unz/four_mutations_full_data.csv'))}
WT = 'VDGV'
FMAX = max(F.values())
RANKED = sorted(F.values(), reverse=True)
TOP1 = RANKED[int(0.001 * len(RANKED))]          # top 0.1 % threshold
def fit(v): return F.get(v, 0.0)
# amino-acid descriptors (Kyte-Doolittle hydropathy, residue volume A^3 (Zamyatnin), charge at pH 7) for policy features
KD = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
VOL = dict(zip(AA, [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0, 227.8, 193.6]))
CHG = dict(zip(AA, [0, 0, -1, -1, 0, 0, 0.1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
