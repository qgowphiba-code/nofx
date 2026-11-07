# HP × Adapt-1, set 2: lessons so far

*Status as of 2026-11-07. Done: Track 0, Track A, A3b, D1, and B1 lineage 1. Running: E1g. Queued: X1, B1 lineages 2–3, D2.*
*The numbers here come from [`RESULTS.md`](RESULTS.md). Raw outputs are under `runs/`.*

**Setup in one paragraph.** The task is to fold an HP chain on a 3D grid; the score is the number of H–H contacts. A depth-3 search picks each move, and every method uses the same search. Only the scoring of unfinished folds differs:
- **Baseline:** the *contact rule*. A fixed script that counts the contacts made so far.
- **Learned model:** an Adapt-1 value model, trained on (position, target) rows, that predicts how a fold will end.

The comparisons are head-to-head at equal search effort. They make no claim about reaching the optimum.

---

## 1. What we learned about the science

### A learned evaluator can beat the hand-made rule at equal search effort
*One lineage so far; the confirmatory test needs lineages 2–3.*

Eval-L is 40 test chains, never used for training. Totals are contacts summed over all 40.

| Search depth | Contact rule | Learned A0 (after 2 own-play rounds) |
|---|---|---|
| 1 | 578 | **688** |
| 2 | 879 | **902** |
| 3 | 975 ± 13 | **1042** |
| 4 | 1029 | — |
| 5 | 1048 | — |

- The learned model wins at every depth with the same search work. At depth 3 it matches the rule searching 4–5 moves ahead, which costs about 5–25× the evaluations.
- **Caveats:**
  - The model's inputs include the contact count itself (one of its 14 features). Read it as "the rule plus learned corrections".
  - Training cost is not counted.
  - It is still far from good folds. On the Liu panel it scores 145, against 155 for the best of 128 randomized attempts and 171 for the best-known folds. On Eval-L it scores 1042, against 1125 for the best of 128 attempts.

### Own-play data is the lever, but it took two rounds
A0 on the Liu panel went **130 → 126 → 145** across its three training rounds: random folds, then two rounds of its own play. The first own-play round made it slightly worse; the second made it clearly better than the rule. Set 1 never trained on chains matched to the test lengths for two rounds, which is the likely reason it never got there.

### Plain outcome labels beat the "smarter" lookahead labels once the model learns from its own play
- A0 labels each position with *how its fold actually ended*.
- BL, the substrate's documented method, labels positions with *the model's own two-move lookahead estimate*.

| Where | A0 | BL |
|---|---|---|
| Short chains, fixed random data (A3, best-move rate) | 0.772 | **0.798** |
| Short chains, after 1 own-play round (A3b) | **0.825** | 0.813 |
| Long chains, after 2 own-play rounds (B1 l1, Eval-L) | **1042** | 980 |

- Lookahead labels help on a fixed batch of random data. Learning from its own estimates then appears to compound the model's errors, while real outcomes keep it honest.
- A second round of lookahead labels (BL-iter) sharpened one-move judgement (best-move rate 0.809, correlation with the true optimum 0.60). Its rankings got *worse* under deeper search, in every draw.

### Comparing sibling moves helps most on short chains (exploratory)
BLG adds rows for every alternative move, trained to rank them against each other. It was the best arm at every depth on short chains: 0.811, 0.901 and 0.952, the last above the rule's 0.938. Its edge over BL at depth 1 (+0.013) fell below the pre-registered 0.03 bar, so Track B does not use it.

### Representation matters, and the raw graph is not enough
- **The raw object graph** imitates the contact rule's *next move* 99.5% of the time. Under depth-3 search it falls below the rule's band: 124 on the Liu panel, and 417 against 429–458 on 20 long chains.
- **Adding 14 summary features as inputs** puts it in band (125 and 440). Small leaf-value errors that are harmless one move ahead add up in deeper search.

### Smaller lessons
- **Fit settings barely matter at small scale.** Warm or cold, 120 or 240 epochs: all land within 0.015. Set 1's "fit stuck at epoch 0" problem did not reproduce.
- **Blending the rule into a good learned model adds nothing** (1039 against 1042). The learned values mostly break the rule's many exact ties.
- **The robust recipe (ROB) is inconsistent against plain labels:** +0.036, −0.017 and +0.022 across the three draws.

### Neighbouring problems
- **2D folding:** dead ends matter more there. On short chains the learned edge over the 2D rule is larger at depth 1 (BLG 0.796 against 0.639, +0.157; in 3D +0.116) and near zero at depth 3.
- **Four-in-a-row (5×4):**
  - Random-game outcomes are nearly unlearnable. Even the exact Bayes predictor explains only 11% of the variance, and the server rejected the model.
  - With heuristic-guided games the models train, but stay well below the hand-made line rule: depth-1 best-move rate about 0.54–0.62 against 0.795. Those numbers come from 2 of 3 draws, with no own-play rounds.

### Noise is large; trust multi-draw and 40-chain numbers
- A single 7-chain panel score swung by 10 or more points between runs. Gate G2 flipped between protocol versions, from BL trailing by 15 to BL leading by 6.
- Changing only the validation split moved a short-chain rate by 0.03.
- So single-run differences under about 0.03, or about 5 panel points, are noise.

---

## 2. What we learned about the substrate (Adapt-1 on Fast-UAT, build `af6670b`)

| Lesson | Consequence |
|---|---|
| The environment's key is rejected by production (401); Fast-UAT works and runs set 1's build | All runs use Fast-UAT |
| Native planning gives **exactly** the same result as client-side search over `predict` (390/390 cases, \|Δ\| = 0) | Evaluation uses the cheaper batched client-side search |
| A repeat fit on the same data gets a **new sha but identical predictions** | A "new sha" check can't detect a stalled generation; compare predictions instead |
| Fits are deterministic in function, across domains too (ROB scored 0.818 three times) | A lost domain can be rebuilt exactly |
| Candidate fits end as `candidate` and installed fits as `trained`; domain create returns 201 | Handled in the client |
| A fit that uses `continuation` rows **cannot be installed** (422) | Continuation fits stay candidates |
| `adaptation_prediction` selection works only for dense models (422 for object graphs) | A2 arm refused and recorded |
| The `projected` representation refuses long-chain object graphs (vocabulary limit) | A1's R2 arm was impossible |
| The server **rejects models with no validation skill** (`candidate_failed_validation`) | Caught a bug of mine (game positions lost their winner) and an unlearnable data design |
| Training jobs fail at roughly 300 s under load, and time is **not linear in rows × epochs** | B1 caps rows per fit at 12k (dropping g0 at g2) and uses fewer epochs |
| **One value fit at a time per owner**, shared with other workloads on the same account | Fits wait on 409 and retry failures; throughput halves under contention |
| Payload limit is 1 MiB (413 at about 1,029 KiB); 2 requests in flight is safe | Client caps bodies at 900 KiB |
| Cloudflare blocks the default Python user agent | Client sends its own user agent |

---

## 3. Method lessons

- **Dry runs against a mock catch our bugs, not the platform's rules.** Every real refusal above was new. Each was added to the mock afterwards, so later dry runs cover it.
- **Expect interruptions.** A container restart, server-side failures and a shared fit slot all happened. Every run deletes its domains at the end, a registry tracks live ones, orphans were cleaned up, and fits retry.
- **Pre-register decision rules and follow the plan's text.** In A3 my code read "no BL arm beats A0" more narrowly than the plan does. I fixed the code to match the plan and reported both readings.
- **Re-run gates on the final protocol.** Gate G2 fired under a superseded protocol and did not fire under the final one. Only the final-protocol run counts.
- **Exact references wherever possible.** The short-chain oracle and the game solver gave unambiguous grading. Without them, the long chains depend on the rule's spread and on published best-known values.

---

## 4. Still open

- **Confirmatory tests R1, R2, R3** need B1 lineages 2–3, now queued: per-chain differences, bootstrap CI, Wilcoxon, Holm correction.
  - R1 (BL vs A0) currently leans *against* BL.
  - R2 (learned vs rule) and R3 (blend vs rule) lean *for*, on one lineage.
- **Search extensions (X1, queued):** native `max_extensions` 2/4/8 on short chains, graded exactly. Also learned **depth-4** search on the panel in lineages 2–3.
- **D2:** a 2D long-chain lineage, approved by gate GD and queued.
- **E1g:** draw 3 and gate GE. A pass looks unlikely.
- **A stronger compute-matched baseline:** best-of-k randomized search at the same number of evaluations as the learned search. It beats the learned model at large k; the matched comparison is planned for C1.
