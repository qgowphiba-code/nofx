"""Published HTE reaction datasets (from the EDBO repository, Shields et al., Nature 2021).

aryl_amination: Buchwald-Hartwig C-N coupling, Ahneman et al., Science 2018 (15 aryl halides x 4 ligands x 3 bases x 22 additives)
suzuki:         Suzuki-Miyaura coupling, Perera et al., Science 2018 (4 electrophiles x 3 nucleophiles x 11 ligands x 7 bases x 4 solvents)
An episode = one substrate (aryl halide / electrophile-nucleophile pair); actions = that substrate's reaction conditions;
the outcome = the recorded yield (%/100). Context = the substrate's identity as one-hot leaving group and core/partner.
"""
import csv, hashlib
from pathlib import Path
RAW = Path(__file__).resolve().parents[1] / 'data' / 'raw'
SPEC = {
    'aryl_amination': {'file': 'edbo_aryl_amination_experiment_index.csv', 'sha256': '88e47ff4ebcd1dac',
                       'substrate': ['Aryl_halide_SMILES'], 'conds': ['Ligand_SMILES', 'Base_SMILES', 'Additive_SMILES']},
    'suzuki': {'file': 'edbo_suzuki_experiment_index.csv', 'sha256': 'cc1abd59e4f5f8f8',
               'substrate': ['Electrophile_SMILES', 'Nucleophile_SMILES'], 'conds': ['Ligand_SMILES', 'Base_SMILES', 'Solvent_SMILES']},
}

def halide_of(smi):
    for h, tag in (('I', 'I'), ('Br', 'Br'), ('Cl', 'Cl'), ('S(', 'OTf')):
        if h in smi: return tag
    return 'other'

def load(name):
    sp = SPEC[name]; path = RAW / sp['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest().startswith(sp['sha256'])
    rows = list(csv.DictReader(open(path)))
    levels = {c: sorted({r[c] for r in rows}) for c in sp['conds']}
    short = {c: {v: f"{c.split('_')[0][0].lower()}{i}" for i, v in enumerate(levels[c])} for c in sp['conds']}
    subs = sorted({tuple(r[s] for s in sp['substrate']) for r in rows})
    # context: one-hot leaving group + one-hot core (amination) / one-hot nucleophile (suzuki)
    if name == 'aryl_amination':
        lg = sorted({halide_of(s[0]) for s in subs}); cores = sorted({s[0].replace('Br', 'X').replace('Cl', 'X').replace('I', 'X') for s in subs})
        ctx = {s: [1.0 if halide_of(s[0]) == g else 0.0 for g in lg] + [1.0 if s[0].replace('Br', 'X').replace('Cl', 'X').replace('I', 'X') == c else 0.0 for c in cores] for s in subs}
        ctx_names = [f'lg_{g}' for g in lg] + [f'core_{i}' for i in range(len(cores))]
    else:
        lg = sorted({halide_of(s[0]) for s in subs}); nus = sorted({s[1] for s in subs})
        ctx = {s: [1.0 if halide_of(s[0]) == g else 0.0 for g in lg] + [1.0 if s[1] == n else 0.0 for n in nus] for s in subs}
        ctx_names = [f'lg_{g}' for g in lg] + [f'nuc_{i}' for i in range(len(nus))]
    conds = sorted({tuple(short[c][r[c]] for c in sp['conds']) for r in rows})
    table = {}
    for r in rows:
        table[(tuple(r[s] for s in sp['substrate']), tuple(short[c][r[c]] for c in sp['conds']))] = float(r['yield']) / 100.0
    return {'name': name, 'substrates': subs, 'conditions': conds, 'yield': table, 'context': ctx, 'context_names': ctx_names,
            'cond_fields': [c.split('_')[0].lower() for c in sp['conds']], 'levels': {c.split('_')[0].lower(): len(levels[c]) for c in sp['conds']}}

def policy_id(cond):
    return '_'.join(cond)
