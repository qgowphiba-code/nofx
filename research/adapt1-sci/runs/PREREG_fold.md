# Exp C (light) pre-registration: 2D HP inverse folding x Adapt-1 sequential route

Question: with a reward that arrives only after the whole sequence is designed, does the sequential route
(eligibility traces + sequential Q) learn a design policy that transfers to unseen targets?
Bank: the 25 designable 12-residue conformations (exact enumeration of 15,037 conformations); train on targets
0-19 (80 episodes, fixed shuffled cycle), hold out 20-24 (10 frozen exploit episodes, no feedback).
Context: position, the residue's contact count in the target, chain-end flag, previous choice. Reward p(target), T = 0.3.
Comparators: random sequences (floor); tabular Monte-Carlo control with the same context and episodes (a general
learner); burial heuristic (H iff the residue has a contact) is a diagnostic and close to a solver (91 % of the
ceiling); exact ceiling over all 4096 sequences.
