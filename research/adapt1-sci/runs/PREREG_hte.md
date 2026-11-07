# Exp B' pre-registration: published HTE datasets x Adapt-1 contextual policy route

Data (from the EDBO repository, Shields et al., Nature 2021; recorded yields, no simulator):
- Buchwald-Hartwig amination, Ahneman et al., Science 2018: 15 aryl halides x 264 conditions (4 ligands x 3 bases x 22 additives)
- Suzuki-Miyaura coupling, Perera et al., Science 2018: 12 substrate pairs x 308 conditions (11 ligands x 7 bases x 4 solvents)
Protocol: an episode = one substrate; 30 experiments per substrate, no repeats; outcome = recorded yield.
Substrates are visited in 3 fixed random orders (seeds 0, 1, 2); Adapt-1 keeps one Domain per (dataset, order).
Context = substrate identity (one-hot leaving group + core / nucleophile); condition features = ligand/base/additive or solvent labels.
Comparators, same information and budget: random; GP-EI per substrate on one-hot conditions (the EDBO paper's one-hot
baseline form); GP-EI with transfer (substrate one-hot + condition one-hot, all earlier substrates in the order).
Ceiling: the best recorded yield for the substrate. Metric: best yield found after n experiments / ceiling,
n in {1, 5, 10, 20, 30}; split by position in the order (first third vs last two thirds) to see transfer.
