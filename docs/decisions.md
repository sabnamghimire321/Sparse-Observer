# Decision log

This is where I kept track of what I decided, what went wrong and what the results were. The
entries are grouped by topic instead of by date, and each one keeps the date it was first written.
The work itself happened between 2026-10-04 and 2026-10-12, in a few intense days, and this file
does not pretend otherwise.

## How to read this file

- Dates are the ones from my working log. Several entries share a date because I wrote them in the
  same session.
- "Pre-specified" means a criterion was written down before the run it applies to. My log used to
  say "pre-registered", but none of these were registered anywhere with an independent timestamp
  (see section 5), so I use "pre-specified" throughout.
- "Confirmatory" means a pre-specified hypothesis or criterion that was evaluated as written.
  "Exploratory" means everything else, including descriptive comparisons.
- A few entries were reworded when I tidied the project up. The numbers were not changed.

## Summary

- Task: predict whether an AI system answers a benchmark item correctly when 0 to 31 of its
  outcomes on that benchmark are revealed. Scored by Brier ALC (lower is better).
- Data: five benchmarks of the Measurement Data Bank training pool with 0/1 responses; MMDocRAG is
  excluded. There are 221 eligible subject-benchmark pairs. Analyses that compare subjects within
  a benchmark use 220 of them (without swe_rebench, which has one subject).
- Main result: learning one success rate per pair from the revealed labels lowers ALC by 0.0407
  against a constant 0.5 (interval -0.0508 to -0.0299). Four estimators of that rate are within
  0.007 of each other. Subject ability (H1) is not supported. Item text does not transfer across
  benchmarks. Item-level evidence from other pairs only pays off from about ten source pairs. An
  empirical-Bayes interval met the coverage criterion I set (H5).
- Not tested: acquisition policies (H2 to H4).

## 1. Scope and task

- The task, as described in the starting-kit README and in `docs/protocol_notes.md`: for a subject
  and a benchmark item, predict P(correct) at label budgets 0, 1, 3, 7, 15 and 31 per
  subject-benchmark pair, with nested label sets. Scored by Brier ALC with weights 0.1, 0.2, 0.2,
  0.2, 0.2, 0.1, averaged equally over pairs.
- (2026-10-06) Risk for any follow-up: `access_date` and `harness_version` are almost entirely
  empty, so this data cannot support a study of change over time. Another dataset would be needed.
- (2026-10-08) I froze the experiments. After that only additions that the report needed were
  allowed.

## 2. Data and eligibility

- (2026-10-04) MMDocRAG is excluded from the main analysis. Its responses are judge scores in steps
  of 0.04 and there is no obvious rule for making them 0/1. A threshold analysis would be a
  separate sensitivity study (`configs/data.yaml` shows how the rule is set, and `labels.py` prints
  a threshold sensitivity).
- (2026-10-04) Responses that are exactly 0 or 1 are kept. MathArena drops 83 of 92,944.
- (2026-10-06) The task description gives every pair its own 50/50 split, so an item that is
  evaluated for one subject can be an acquisition label for another subject on the same benchmark.
- (2026-10-06) Pairs are counted as (benchmark, subject_id), because subject_id repeats across
  benchmarks. There are 221 pairs, 60 of them small (at most 150 distinct items).
- (2026-10-09) Pairs per benchmark: 81 matharena, 82 multi_swebench, 26 real_webagents, 31
  researchcodebench, 1 swe_rebench. Analyses that need at least two subjects per benchmark
  (comparisons, headroom, break-even, downside, heterogeneity) use the 220 pairs without
  swe_rebench.
- (2026-10-08) The local data copy had 48 files and 67,172,513 bytes and was intact. 99.9% of
  evaluation items have a response from another subject on the same benchmark.

## 3. Simulation design

- (2026-10-04) Split: a pair's items are shuffled with a seed derived from the benchmark and the
  subject. With an odd count the extra item first went to the evaluation half; this was corrected
  later (section 7.2).
- (2026-10-04) Reveal: if an acquired item has several recorded trials, one is revealed at random.
  Every trial of an evaluation item is a separate target.
- (2026-10-04) Deviation from the plan, made before any H1 result: intervals come from a bootstrap
  over model names instead of over pairs, because several subjects are the same model.
- (2026-10-09) Seed 0 was used during development (including debugging the Newton step) and is
  also one of the five reported seeds (0 to 4).
- (2026-10-12) The interval-coverage experiment uses seeds 1000 to 1199 (200 draws per pair),
  separate from the seeds of the other experiments.
- (2026-10-06) `labels_per_covered` is zero-filled for pairs with no covered item, so it is biased
  low and I do not use it.
- (2026-10-08) One runner (`results/main_v1`) now holds all seven models under the final protocol,
  and one `stats.py` makes every comparison from it.

## 4. Models

- (2026-10-06) Empirical-Bayes prior (exploratory): a smoothed distribution of pair success rates
  from the training benchmarks that have at least 5 eligible pairs, equal weight per benchmark,
  smoothing bandwidth 0.05, floor 0.1. Likelihood tempering 1.0 is the main version and 0.5 is
  exploratory.
- (2026-10-06) Shared benchmark-level evidence: other pairs' labels on the same benchmark enter the
  pair's rate with weight 0.5 and one source pair (main version). Weight 0.25 or 1.0 and three
  source pairs are exploratory. Shared evidence cannot help at budget 0, and it only applies to
  benchmarks that have a second pair in the draw.
- (2026-10-06) Item-level model: a per-pair offset plus an item effect estimated from labels that
  other pairs acquired on the same item. Tested with 0, 1, 3, 10 and 30 source pairs.
- (plan, 2026-10-04) H1 models: `offset_logit_s0` (no subject information), `offset_logit_s1`
  (full-scale ability, main) and `offset_logit_s0.5` (exploratory). Prior sd 1.0, shrinkage 5.0,
  ability clipped at rates 0.02 to 0.98.
- Items have no id in the standalone predictor's input, so `models/runtime.py` matches items by a
  hash of their text.

## 5. Hypotheses and outcomes

The plan (`docs/analysis_plan.md`) was written on 2026-10-04 before the first model comparison. In
the version history the file was empty on 2026-10-04 and first held the text on 2026-10-06, after
the first H1 and shared-evidence runs. The addenda (2026-10-09 and 2026-10-11) were not committed
before the coverage experiment ran. My log says the criteria for the shared-evidence,
text-transfer and break-even checks were fixed before those runs, but the file history cannot
confirm that and no external timestamp exists.

| Item | Specified | Outcome |
|---|---|---|
| H1: subject ability from other benchmarks lowers ALC | plan, 2026-10-04 | Not supported: -0.0028 [-0.0091, +0.0038], multi_swebench and real_webagents, 108 pairs, 35 clusters |
| H2: uncertainty-driven acquisition beats random | plan | Not tested |
| H3: streaming gap grows as pool-to-budget ratio shrinks | plan | Not tested |
| H4: surprise-adaptive threshold helps | plan | Not tested |
| Gate for H2 to H4: oracle selection must beat random by more than 0.01 | plan | Never computed |
| Shared benchmark evidence is kept if ALC is at least 0.003 lower in at least 3 of 4 benchmarks | log, 2026-10-06 | Not met: 0 of 4 |
| Sentence-embedding difficulty transfer: skill above 0.05 in at least 3 of 4 benchmarks | log, 2026-10-06 | Not met: 0 of 4 |
| Item-level pooling beats its control by at least 0.003 in at least 3 of 4 benchmarks | log, 2026-10-06 | Met at 10 source pairs, not at 3 |
| H5: empirical-Bayes interval coverage in 0.85 to 0.95 at budgets 7, 15, 31 in each of 4 benchmarks | addendum, 2026-10-09 | Supported (12 of 12 estimates inside); two cells are marginal |

Details of the outcomes:

- (2026-10-04) H1 is kept as written. Ability is known for at least half of the subjects only in
  multi_swebench (0.73) and real_webagents (0.55); researchcodebench is just under (0.48). Any
  other reading is exploratory.
- (2026-10-08) H1 under the final protocol: -0.0028 [-0.0091, +0.0038], the same as before the
  protocol fix to the reported precision.
- (2026-10-06) Shared benchmark evidence (weight 0.5, one source pair), ALC difference against the
  control: -0.0001 (matharena), -0.0010 (multi_swebench), +0.0073 (real_webagents), -0.0018
  (researchcodebench). None of the four reaches -0.003. Shared labels help at 1 to 3 labels and
  hurt from 15 labels on (Brier 0.1897 against 0.1840 at 31). Dropped. The weight-0.25,
  three-source variant (-0.0016 overall) is exploratory.
- (2026-10-06) Sentence-embedding difficulty model (MiniLM, ridge, equal weight per benchmark):
  cross-benchmark skill above 0.05 in 0 of 4 benchmarks. Cross-benchmark Spearman rho: matharena
  -0.195, multi_swebench 0.231, real_webagents -0.091, researchcodebench -0.298. Item text is
  dropped as a starting point for a new benchmark. The criterion was fixed before the run, with
  training items weighted so that benchmarks count equally (a change from my first TF-IDF look,
  applied to both so they stay comparable).
- (2026-10-06) Break-even: at least 0.003 ALC better than the 0-source control in at least 3 of 4
  benchmarks. Met at 10 source pairs (matharena -0.0035, multi_swebench -0.0015, real_webagents
  -0.0076, researchcodebench -0.0122) and not at 3 (1 of 4). So the break-even lies between 3 and
  10 on a coarse grid. Overall ALC: 0 sources 0.2099, 1 source 0.2093, 3 sources 0.2084, 10 sources
  0.2054, 30 sources 0.1992.
- Pair-level intervals for the break-even experiment are in `tables/breakeven_v1.csv`: all pairs,
  ALC against the control, 1 source -0.0005 [-0.0007, -0.0004], 3 sources -0.0015 [-0.0017,
  -0.0012], 10 sources -0.0045 [-0.0052, -0.0039], 30 sources -0.0107 [-0.0122, -0.0095].
- (2026-10-12) H5: all 12 empirical-Bayes coverage estimates (four benchmarks times budgets 7, 15,
  31) lie in 0.85 to 0.95 (range 0.860 to 0.927), so the rule is met. Two cells are marginal:
  matharena at 7 labels, 0.860 [0.834, 0.883], and multi_swebench at 7 labels, 0.877 [0.849,
  0.900]. swe_rebench has one pair and is not part of H5. Coverage is over random label draws in
  the simulation, and four benchmarks are the unit of inference.

## 6. Exploratory experiments

- (2026-10-06) The empirical-Bayes prior does not beat the shrunk mean: ALC 0.211 against 0.210
  (all pairs) and 0.207 against 0.206 (small pairs). The tempered variant gives 0.216. Base-rate
  models are stuck near 0.21 locally.
- (2026-10-08) `results/main_v1` reproduces the earlier final-protocol rows to three digits (ALC
  all pairs: constant 0.250, pair_mean_k2 0.210, offset_logit_s0 0.212, empirical_bayes_t1 0.211,
  t0.5 0.216; new: offset_logit_s0.5 0.210, s1 0.210).
- (2026-10-08) Adapting to the labels against the constant: -0.0407 [-0.0508, -0.0299]. This is
  descriptive, since the plan does not state it as a hypothesis. Half-scale ability: -0.0021
  [-0.0041, -0.0001], listed in the plan as an exploratory variant. The other comparisons are in
  `tables/comparisons_main_v1.csv`.
- (2026-10-06) Share of evaluation items covered by other pairs' acquired labels (1 source, 31
  labels): matharena 5.8%, multi_swebench 1.8%, real_webagents 18.9%, researchcodebench 14.7%; at
  7 labels at most 4.4%. With 3 sources: 14.2%, 5.3%, 43.8%, 38.0% at 31 labels.
- (2026-10-06) A rough bound on the gain from shared evidence: coverage times the item-oracle gain
  (0.083), about 0.002 to 0.005 ALC with one source. My first write-up of this compared it with the
  0.01 gate for acquisition policies and concluded that H2 to H4 were not worth pursuing. That
  mixed two different quantities (see 7.5). A model that has one number per pair gains nothing
  from selection, because every label tells it the same thing.
- (2026-10-06) The break-even gain follows coverage (share of evaluation items with a shared label
  at 31 labels: 6 to 19% with 1 source, 41 to 99% with 30). Caveat: sources are sampled uniformly
  from eligible pairs, so models that appear under several scaffolds in multi_swebench could
  inflate its gain.
- (2026-10-07) Item-level effect by pair success rate (10 sources, ALC difference, negative is
  better): rate below 0.1: -0.0001; 0.1 to 0.3: -0.0041; 0.3 to 0.7: -0.0071; 0.7 and above:
  -0.0024. Pairs near the floor gain nothing at any number of sources.
- (2026-10-08) Downside, item-level minus control, per pair and seed: 1 source mean -0.0005 (95th
  percentile +0.0009, worst +0.0032); 3 sources -0.0015 (worst +0.0077); 10 sources -0.0045
  (worst +0.0078). In random draws of nine pairs the item model looks worse than its control in
  10.6% (1 source), 5.3% (3 sources) and 0.6% (10 sources) of draws.
- (2026-10-08) Headroom under the final protocol (five seeds, 220 pairs): constant oracle 0.176,
  item oracle 0.094, `offset_logit_s0` Brier 0.261 at 0 labels and 0.184 at 31. The item oracle is
  an upper bound that uses other subjects' responses on the target benchmark, so it is not an
  achievable method.
- (2026-10-12) Interval results that I did not specify in advance: empirical-Bayes coverage at 1 to
  3 labels is low in matharena (0.81) and multi_swebench (0.85 to 0.88). Wilson under-covers at 31
  labels (0.789 matharena, 0.877 multi_swebench, 0.811 real_webagents, 0.841 researchcodebench).
  The pool-aware interval adds 0.027 to 0.077 coverage there. A deliberately wrong prior lowers
  coverage to 0.55 to 0.86 at 1 label. Full-budget interval widths are 0.23 to 0.29.
- (2026-10-07) I only checked the literature at abstract level for some papers. Truong et al.
  (ICML 2025) report that a content-based difficulty predictor generalizes across datasets when it
  has dataset context. My cross-benchmark text transfer failed in a different setting (agentic
  benchmarks, 31 to 104 subjects, no dataset context). `literature/notes.md` records how closely I
  read each paper.
- Some numbers in the report exist only as console output of my runs: the TF-IDF and
  sentence-embedding difficulty tables, the `item_features` check, the text-length correlations and
  the heterogeneity table. They are regenerated by `python scripts/run_all.py` (item_features and
  heterogeneity) and `python scripts/run_all.py --with-embeddings` (text transfer), with logs in
  `logs/run_all`. I have not re-verified those particular numbers since.

## 7. Bugs, corrections and superseded results

### 7.1 Newton curvature bug

- (2026-10-04) In `OffsetLogit`, the Newton curvature ignored the number of labels, which made the
  fit unstable from about 7 labels on. I found it by checking against the sample mean with a flat
  prior (0.2 for 3 ones in 15 labels). Results from before the fix were thrown away.

### 7.2 Odd-item split

- (2026-10-06) The task description gives the extra item of an odd count to acquisition, but my
  simulation had given it to evaluation. Fixed in `simulator/protocol.py`. Results from before this
  entry used the old rule. With the fix, the local results are unchanged to three digits.

### 7.3 Pair counting

- (2026-10-06) Pairs were counted by subject_id, which repeats across benchmarks. They are now
  counted over (benchmark, subject_id).

### 7.4 Results that could not be regenerated

- (2026-10-08) The H1 comparison came from an earlier version of the runner and an older results
  folder (from before the odd-item fix), and the repository could no longer regenerate it.
  `headroom.py` also read that stale folder. The repair: one runner (`results/main_v1`) with all
  seven models under the final protocol, and one `stats.py` that makes every comparison from it.
  H1 was recomputed under the final protocol. The older folders were dropped from the project and
  none of their numbers are used in the report.

### 7.5 Inaccuracies I found when tidying up (2026-10-14)

- **Acquisition gate.** Two earlier entries said that the acquisition gate was not met and that all
  pre-specified routes had failed, including the empirical-Bayes prior. In fact no acquisition
  policy was ever implemented and the oracle-selection gate was never computed. The first quantity
  I had compared was the gain from shared evidence, which is a different thing. So H2 to H4 are
  untested, not failed, and the empirical-Bayes prior was exploratory.
- **Counts.** "43 tests" and "18 stages" were true on 2026-10-08. The suite has 55 tests since
  2026-10-12 (metrics 8, protocol 8, models 27, intervals 8, coverage 4).
- **Terminology.** Some criteria were called "pre-registered". See section 5 for what I can and
  cannot claim.
- **Dependencies.** `requirements.txt` did not list numpy or matplotlib and mixed required and
  optional packages, and `pyproject.toml` listed only four of the seven required ones. Both are
  fixed, with the optional embedding packages in `requirements-embeddings.txt`.

### 7.6 Planned but not run

- Controls beyond the wrong-prior test (shuffled labels, a numerical check of the Brier identity
  for the interval study).
- Acquisition policies, the gate for them, and a threshold analysis for MMDocRAG.

## 8. Reproducibility and remaining limits

- `scripts/run_all.py` regenerates every result from the downloaded data. Downloading and the
  optional embedding experiment are separate steps.
- Environment of the reported results: Python 3.12.0; numpy 2.5.3, pandas 3.0.6, pyarrow 25.0.1,
  scipy 1.18.1, scikit-learn 1.9.1, huggingface_hub 1.33.0 (2.0.0 was installed when I downloaded
  the data), matplotlib 3.11.2, and sentence-transformers 6.1.0 for the optional experiment.
- (2026-10-12) A full run of `run_all.py` (19 stages, 21.4 minutes) reproduced the result files of
  the earlier run byte for byte on my Windows machine.
- The result files were written on Windows with Windows line endings and later converted to Unix
  ones for this repository, so file hashes will differ from mine. Compare the tables, not the
  hashes.
- The 55 tests pass, and they do not need the gated data. I did not rerun the full pipeline on the
  final cleaned copy, because that needs the data.

Remaining limits:

- The simulation follows the task description but is not the organizers' code, so local ALC is not
  comparable with leaderboard scores.
- Four benchmarks have several subjects. The intervals reflect pairs and model names, not
  benchmarks. swe_rebench is one pair and MMDocRAG is excluded.
- Seed 0 was used in development and is also a reported seed.
- H2 to H4 are untested and the oracle bounds are not methods.
- The text-transfer test used one TF-IDF and one embedding model with ridge regression.
