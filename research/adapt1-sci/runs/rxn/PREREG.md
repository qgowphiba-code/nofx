# Exp B pre-registration: experiment-level reaction optimization x Adapt-1 contextual policy route

Question: when each query is one experiment (choose conditions, get the measured yield back immediately), does
Adapt-1 find high-yield conditions in few experiments, and does retained state from earlier reactions help on
new ones (the "useful behavior on unseen cases" claim)?

Task: 48 conditions (4 catalysts x 4 temperatures x 3 residence times). 40 reactions (ids 0-39), drawn from four
recurring kinetic families with jittered parameters; 20 experiments per reaction; untested conditions only.
Context = three measured substrate descriptors (two noisy family signals, one pure noise). Kinetics hidden.

Arms: Adapt-1 `ctx` (descriptors) and `noctx` (constant context), Thompson exploration, one Domain per arm across
all 40 reactions (retained state is the point), online throughout.
Comparators, same information and budget, same measurement for the same experiment:
- random untested conditions (floor)
- GP-BO per reaction (expected improvement, no memory of earlier reactions)
- GP-BO with transfer (one GP over descriptors + conditions using all earlier reactions' data)
- ceiling: the grid optimum under the true kinetics
Metric: best true yield found after n experiments / grid optimum, n in {1, 3, 5, 10, 20}; reported for reactions
0-9 (little retained state) and 20-39 (after transfer has had data). Paired by reaction.
