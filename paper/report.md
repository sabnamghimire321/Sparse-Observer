# Predicting AI benchmark outcomes from a few labels: what a handful of results can and cannot tell you

**Author:** Sabnam Ghimire

This is a technical report on a personal research project. The code, the decision log and the saved
results are in the same repository. The setup follows the public description of the NeurIPS 2026
Predictive AI Evaluation Competition, but this work is an independent study and none of the numbers
are leaderboard scores.

## Abstract

I study the following task: predict whether an AI system answers a benchmark item correctly when 0
to 31 of its outcomes on that benchmark are revealed. Using the public training pool of the
Measurement Data Bank, I built a local simulation of the task and evaluated simple statistical
predictors by leaving one benchmark out at a time, with intervals that account for repeated model
names. Learning one success rate per subject-benchmark pair from the revealed labels lowers Brier
ALC by 0.041 (95% interval 0.030 to 0.051) against a constant 0.5, and four estimators of that rate
are within 0.007 of each other. Subject information from other benchmarks, item text, and
benchmark-level shared evidence did not meet the success criteria I set in advance. Item-level
information has a large oracle headroom (Brier 0.094 against 0.176) that I could not reach; it only
becomes useful when about ten other subjects have labels on the same benchmark. Finally, I test
whether intervals for a pair's accuracy are calibrated at small label budgets. An empirical-Bayes
interval met the coverage criterion I set at 7, 15 and 31 labels, while the standard Wilson
interval under-covers at large budgets. A version that accounts for the target being a finite
sample under-covers less. I report the failures as results and list the limits of a study with
four benchmarks.

## 1. Introduction

The task is scored by the area under the Brier learning curve (Brier ALC) as labels are revealed.
Most of the weight sits at small budgets, so the question is a practical one: how much does each
extra label improve a prediction, and how far can an estimate built from so few be trusted?

This is an empirical study, not a new method. I contribute four things: (i) a reproducible local
simulation of the task with tests and analysis criteria fixed in advance, (ii) a description of
where the reachable gain is in the low-label regime and where the apparent headroom is out of
reach, (iii) a calibration study of sparse accuracy estimates, including a correction for the
finite evaluation pool, and (iv) a break-even analysis of how much shared item evidence is needed
before item-level modelling pays off.

## 2. Related work

Adaptive and sparse evaluation of language models is well studied, and I do not claim the
underlying ideas are new. I read the first three papers below in full and the last two only at
abstract level.

- Active Evaluation Acquisition [AEA] picks prompts one after another from a pool to estimate a
  model's benchmark score. It reports that uncertainty-driven policies explored poorly while random
  selection stayed competitive.
- "100 instances is all you need" [100I] predicts item-level success of a new model from a small
  reference set. It finds that random references match more elaborate selectors and that
  out-of-distribution prediction is weak.
- ATLAS [ATLAS] uses computerized adaptive testing with a calibrated item bank, and reports that 30
  to 89 items matched whole-bank ability.
- Fluid Benchmarking [FLUID] and Truong et al. [TRUONG] (content-based item difficulty prediction
  that generalizes across datasets when it has dataset context) are described here from their
  abstracts only.

Stream-based selective sampling is also a classical setting in active learning. My setting differs
in using item-level binary outcomes for agentic and code benchmarks, a budget of at most 31 labels,
and scoring by Brier ALC. I did not evaluate any acquisition policy.

## 3. Task, data and protocol

**Data.** The six benchmarks of the Measurement Data Bank training pool (MathArena, MMDocRAG,
Multi-SWE-bench, REAL, ResearchCodeBench, SWE-rebench). I only use the item, subject and response
tables (48 files, 64 MB). MMDocRAG responses are judge scores in steps of 0.04 and there is no
obvious rule for turning them into 0/1, so I leave it out. For the other benchmarks I keep the
responses that are exactly 0 or 1 (83 of 92,944 MathArena responses are dropped).

**Pairs.** A pair is a subject on a benchmark with at least 80 distinct items. That gives 221 pairs
(MathArena 81, Multi-SWE-bench 82, REAL 26, ResearchCodeBench 31, SWE-rebench 1). Comparisons that
need several subjects per benchmark use the 220 pairs without SWE-rebench.

**Protocol (simulation).** The items of a pair are split 50/50 into an acquisition pool and an
evaluation pool (an odd item goes to acquisition). Labels are revealed from the acquisition pool in
a fixed random order, one randomly chosen recorded trial per item, at the nested budgets 0, 1, 3, 7,
15 and 31. Every trial of an evaluation item is a target. Brier ALC is 0.1 B0 + 0.2 B1 + 0.2 B3 +
0.2 B7 + 0.2 B15 + 0.1 B31, averaged over pairs. Each predictor is evaluated by leaving one
benchmark out. The main experiments use five seeds (0 to 4) and the coverage study uses 200 draws
with seeds 1000 to 1199.

**Uncertainty.** Intervals are bootstrapped over model names (120 clusters for the 220 pairs),
because several subjects are the same model under different scaffolds. This does not capture
variation between benchmarks, and there are only four benchmarks with several subjects.

## 4. Predictors

Every predictor gives one probability per pair from the revealed labels (except the item-level
model of Section 6.5).

- **Constant 0.5.**
- **Shrunk mean (`pair_mean_k2`).** The posterior mean with a prior of strength 2, centred on the
  mean success rate of the other benchmarks. This is the main comparator.
- **MAP offset.** A logit offset fitted by MAP with a normal prior (sd 1).
- **Subject ability.** The MAP offset plus a fixed ability: the shrunk log-odds gap of the same
  model name on other benchmarks.
- **Empirical-Bayes prior.** A smoothed distribution of pair rates from the training benchmarks,
  with tempering 1 (main) and 0.5.
- **Item-level pooled model.** The MAP offset plus an item effect estimated from labels that other
  pairs on the same benchmark acquired on the same item (items are matched by text). With no pooled
  labels it equals the MAP offset, which the tests check. A standard-library version is in
  `src/sparse_observer/models/runtime.py`.

## 5. What was specified in advance

| Item | Status | Outcome |
|---|---|---|
| H1: subject ability lowers ALC | plan written 2026-10-04 | not supported |
| H2 to H4: acquisition policies | plan written 2026-10-04 | **not tested** |
| Shared benchmark evidence, text transfer, break-even criteria | fixed in the decision log before each run | shared evidence: 0 of 4; text: 0 of 4; break-even: met at 10 sources |
| H5: coverage of an empirical-Bayes interval | addendum written before any interval was computed | supported (with caveats) |
| Empirical-Bayes prior, MAP comparisons, heterogeneity, downside | exploratory | descriptive |

**When the plan was written.** The text of the plan was written before my first model comparison. In
the repository's earlier history the file was empty in the 2026-10-04 commit and first held the text
in a commit of 2026-10-06, after the first H1 and shared-evidence runs had been seen, and nothing
independent timestamps the paste. So I call it an analysis plan written before the first comparison,
not a verifiably registered one.

Two implementation errors turned up during development: a curvature term in the MAP fit that made
estimates unstable above 7 labels, and the placement of the odd item in the split. Both were fixed
and the results from before the fixes were thrown away. Seed 0 was used during development and is
also a reported seed. I made one deviation from the plan, before any H1 result: intervals are
bootstrapped over model names and not over pairs.

## 6. Results

### 6.1 Adapting to the revealed labels is the one clear effect

Table 1 (221 pairs, mean over five seeds; Brier, lower is better).

| Predictor | ALC all pairs | ALC small pairs (60) | B0 | B1 | B3 | B7 | B15 | B31 |
|---|---|---|---|---|---|---|---|---|
| constant 0.5 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| shrunk mean | 0.2096 | 0.2060 | 0.2605 | 0.2300 | 0.2096 | 0.1968 | 0.1891 | 0.1844 |
| MAP offset | 0.2120 | 0.2079 | 0.2605 | 0.2354 | 0.2138 | 0.1987 | 0.1895 | 0.1843 |
| MAP offset + ability (0.5 / 1) | 0.2099 / 0.2100 | 0.2058 / 0.2061 | 0.2538 / 0.2536 | 0.2311 / 0.2312 | 0.2118 / 0.2120 | 0.1981 / 0.1983 | 0.1893 / 0.1894 | 0.1843 |
| empirical-Bayes prior (1 / 0.5) | 0.2108 / 0.2160 | 0.2071 / 0.2118 | 0.2578 | 0.2332 / 0.2410 | 0.2123 / 0.2218 | 0.1978 / 0.2041 | 0.1894 / 0.1919 | 0.1843 / 0.1849 |

The shrunk mean lowers ALC by 0.0407 against the constant (95% interval 0.0299 to 0.0508, 220
pairs, 120 clusters). The difference is negative in all four multi-subject benchmarks (MathArena
0.0124, Multi-SWE-bench 0.0722, REAL 0.0497, ResearchCodeBench 0.0236) and in every seed. A likely
reason is that many pairs have success rates far from 0.5, which makes a constant 0.5 a poor guess
for them, but I did not isolate this. At zero labels the prior costs something: 0.261 against 0.250
for the constant (Figure 1).

### 6.2 What did not help

- **Subject ability (H1).** The full-scale ability changes ALC by -0.0028 (interval -0.0091 to
  +0.0038) in Multi-SWE-bench and REAL, 108 pairs and 35 clusters. Not supported. A half-scale
  variant gives -0.0021 (interval -0.0041 to -0.0001) over all pairs. I defined it after seeing the
  first run, so it is exploratory.
- **Other estimators**, compared with the shrunk mean (positive is worse): empirical-Bayes prior
  +0.0012 (0.0001 to 0.0026), MAP offset +0.0024 (0.0011 to 0.0038), tempered likelihood +0.0053
  (0.0040 to 0.0068). The signs differ by benchmark: stronger priors help benchmarks whose rate is
  near the training average and hurt atypical ones. None beats the shrunk mean, and the differences
  are small.

### 6.3 Headroom

On the evaluation items, the best constant per pair gives a Brier score of 0.176. Adding each
item's difficulty, estimated from the other subjects' responses, together with a per-pair
intercept gives 0.094 (per benchmark, constant versus item: MathArena 0.200/0.100,
Multi-SWE-bench 0.144/0.083, REAL 0.170/0.103, ResearchCodeBench 0.204/0.101; 99.9% of evaluation
items have another subject's response). The shrunk mean reaches 0.184 at 31 labels, so estimating a
single rate costs 0.008 on top of the best constant. The item figure is an oracle: it uses other
subjects' responses on the target benchmark and the pair's own evaluation outcomes for the
intercept, so it is an upper bound and not something that can be built.

### 6.4 Item text does not transfer across benchmarks

A ridge model on item text, trained on the other benchmarks, predicts item difficulty with positive
skill only for Multi-SWE-bench with TF-IDF (0.126). With MiniLM sentence embeddings the
cross-benchmark skill is negative everywhere (MathArena -0.741, Multi-SWE-bench -0.041, REAL
-0.528, ResearchCodeBench -0.245), and my criterion (skill above 0.05 in at least 3 of 4
benchmarks) was met in 0 of 4. Within a benchmark, difficulty can be learned (Spearman 0.475 to
0.531, skill 0.24 to 0.46), with the caveat that random folds may leave near-duplicate items on both
sides. This differs from reported cross-dataset generalization [TRUONG]. My setting differs in the
benchmarks, the number of subjects and the lack of dataset context.

### 6.5 Shared evidence: break-even density

Labels acquired by other pairs on the same benchmark reach only a few evaluation items. With one
source pair and 31 labels the share is 5.8% (MathArena), 1.8% (Multi-SWE-bench), 18.9% (REAL) and
14.7% (ResearchCodeBench). With ten sources it is 36.6%, 16.9%, 80.1% and 79.6%.

- Benchmark-level pooling (other pairs' labels weighted 0.5 in the pair's rate) was dropped by my
  criterion (ALC gain of at least 0.003 in at least 3 of 4 benchmarks; observed -0.0001, -0.0010,
  +0.0073, -0.0018). It helps at 1 to 3 labels and is worse from 7 labels on (Brier at 7, 15 and 31
  labels: 0.1981, 0.1926, 0.1897 against 0.1969, 0.1889, 0.1840 for the control).
- The item-level model (Section 4) improves ALC over its no-pooling control by 0.0005 (interval
  0.0004 to 0.0007) with one source pair, 0.0015 with three, 0.0045 (0.0039 to 0.0052) with ten and
  0.0107 (0.0095 to 0.0122) with thirty (Figure 3). The criterion is met at ten sources (MathArena
  0.0035, REAL 0.0076, ResearchCodeBench 0.0122; Multi-SWE-bench 0.0015, which needs thirty). So the
  break-even lies between three and ten on this grid.
- The effect is concentrated in pairs with intermediate success rates. At ten sources it is -0.0001
  for rates below 0.1, -0.0041 for 0.1 to 0.3, -0.0071 for 0.3 to 0.7 and -0.0024 for 0.7 and
  above. In simulated draws of nine pairs, the item-level model looked worse than its control in
  10.6% of draws with one source and 0.6% with ten, so a small sample cannot reliably show its
  effect.
- Source pairs were sampled uniformly from the eligible pairs, which may inflate the effect where
  one model appears under several scaffolds. I do not know how many other pairs share a benchmark
  in a real evaluation.

### 6.6 How reliable is an estimate from a few labels?

The estimand is the mean outcome over the pair's evaluation pool (every trial counted), for a pair
on a held-out benchmark. I compare 90% intervals:

- Wilson;
- Clopper-Pearson;
- a pool-aware Wilson that adds the sampling variance of the finite evaluation pool (effective size
  n N / (n + N), from my own derivation);
- an empirical-Bayes predictive interval (a prior from the other benchmarks, a posterior over the
  pair's rate, then the distribution of the pool mean).

Coverage is the share of 200 random draws per pair in which the interval contains the pool mean.

My criterion (H5) was empirical-Bayes coverage between 0.85 and 0.95 at budgets 7, 15 and 31 in
each of four benchmarks. **All 12 estimates lie inside the band, so H5 is supported** (Table 2,
Figure 2). Two cells are marginal: MathArena at 7 labels, 0.860 (interval 0.834 to 0.883), and
Multi-SWE-bench at 7 labels, 0.877 (0.849 to 0.900).

Table 2. Coverage of 90% intervals (empirical-Bayes with a 95% bootstrap interval over model names).

| Benchmark | Labels | Empirical-Bayes | Pool-aware Wilson | Wilson | Clopper-Pearson | Wrong prior |
|---|---|---|---|---|---|---|
| MathArena | 7 | 0.860 (0.834, 0.883) | 0.898 | 0.881 | 0.954 | 0.832 |
| MathArena | 15 | 0.886 (0.876, 0.897) | 0.872 | 0.841 | 0.901 | 0.850 |
| MathArena | 31 | 0.889 (0.876, 0.901) | 0.853 | 0.789 | 0.837 | 0.857 |
| Multi-SWE-bench | 7 | 0.877 (0.849, 0.900) | 0.910 | 0.904 | 0.965 | 0.808 |
| Multi-SWE-bench | 15 | 0.890 (0.872, 0.904) | 0.908 | 0.897 | 0.949 | 0.860 |
| Multi-SWE-bench | 31 | 0.897 (0.888, 0.904) | 0.904 | 0.877 | 0.925 | 0.879 |
| REAL | 7 | 0.927 (0.917, 0.937) | 0.903 | 0.886 | 0.955 | 0.893 |
| REAL | 15 | 0.917 (0.907, 0.927) | 0.893 | 0.857 | 0.911 | 0.902 |
| REAL | 31 | 0.917 (0.906, 0.927) | 0.888 | 0.811 | 0.868 | 0.910 |
| ResearchCodeBench | 7 | 0.921 (0.908, 0.933) | 0.904 | 0.892 | 0.955 | 0.879 |
| ResearchCodeBench | 15 | 0.908 (0.897, 0.919) | 0.896 | 0.868 | 0.929 | 0.883 |
| ResearchCodeBench | 31 | 0.907 (0.897, 0.917) | 0.888 | 0.841 | 0.882 | 0.890 |

Observations beyond the test I specified (exploratory):

- At 1 to 3 labels the empirical-Bayes interval under-covers in MathArena (0.81) and
  Multi-SWE-bench (0.85 to 0.88).
- The standard Wilson interval under-covers at 31 labels (0.789, 0.877, 0.811, 0.841). The
  pool-aware version adds 0.027 to 0.077 coverage there (0.853, 0.904, 0.888, 0.888), which fits
  with the target being a finite sample. Its coverage is still low for MathArena (0.853), where
  items are repeated many times per subject and I approximate the pool by its number of items. I
  did not test that explanation.
- A deliberately wrong prior (the training benchmark with the most different mean rate) lowers
  empirical-Bayes coverage to 0.55 to 0.86 at one label, so the interval depends on the prior
  matching the target benchmark.
- At 31 labels the mean width is 0.23 to 0.29, so 31 labels cannot support a claim about a pair's
  accuracy that is more precise than about plus or minus 0.12 to 0.15.

## 7. Limitations

- The simulation follows the task description but is not the organizers' code: the split shuffle
  differs, and I pick one trial at random when an item has several. The pools in a real evaluation
  may be smaller than many of mine and the benchmarks differ, so my ALC values are not comparable
  with leaderboard scores.
- Four benchmarks have several subjects, and many subjects are one model under different
  scaffolds. The intervals reflect pairs and model names, not benchmarks. SWE-rebench has one pair
  and MMDocRAG is excluded.
- The text-transfer test used one TF-IDF and one embedding model with ridge regression. Other
  methods might behave differently.
- The coverage study is a calibration study under a simple model. "Accuracy" means accuracy over
  recorded evaluation responses. Some benchmarks are sparsely covered per subject (MathArena
  subjects answered about 11% of items), so a claim about a whole benchmark needs assumptions I did
  not check. Items with many repeated trials are approximated by their item count.
- I did not evaluate acquisition policies or the streaming accept/skip rule. For a model with one
  number per pair, selection cannot matter.
- The break-even grid is coarse (1, 3, 10 and 30 source pairs) and the source pairs are drawn
  uniformly.
- The `item_features` field is empty in the training data.

## 8. Reproducibility

Python 3.12, with pinned versions in `requirements.txt`. After requesting access to the dataset:
`pip install -r requirements.txt`, `pip install -e .`, `hf auth login`,
`python scripts/download_data.py --execute`, then `python scripts/run_all.py` (19 stages, about 21
minutes on a CPU). Figures and tables: `python scripts/make_report_assets.py`. The analysis plan,
decision log and protocol notes are in `docs/`. No training data other than the Measurement Data
Bank is used.

Every number in Sections 6.1 to 6.3 and 6.5 to 6.6 was recomputed from the saved result files in
`results/` and `tables/` and matched. The text-transfer numbers in Section 6.4 and the
heterogeneity figures in Section 6.5 come from console output of my runs and are not stored in a
result file. `python scripts/run_all.py --with-embeddings` regenerates them in `logs/run_all`.

## 9. Notes

- The embedding experiment used the public `all-MiniLM-L6-v2` model for analysis only.
- I used an AI coding assistant during development. I ran all the experiments, checked the results
  and made the decisions described here.

## References

- [AEA] Active Evaluation Acquisition for Efficient LLM Benchmarking, arXiv:2410.05952.
- [100I] 100 instances is all you need: predicting the success of a new LLM on unseen data by
  testing on a few instances, arXiv:2409.03563.
- [ATLAS] Adaptive Testing for LLM Evaluation: A Psychometric Alternative to Static Benchmarks,
  arXiv:2511.04689.
- [FLUID] Fluid Language Model Benchmarking, COLM 2025, arXiv:2509.11106.
- [TRUONG] Reliable and Efficient Amortized Model-based Evaluation, ICML 2025, arXiv:2503.13335.
- Truong, N., Truong, S. T., Koyejo, S. The AI Measurement Data Bank. AIMS Lab, Stanford
  University, 2026.
