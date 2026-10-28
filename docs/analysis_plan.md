# Analysis plan

Written 2026-10-04, before the first model comparison. Addenda are dated 2026-10-09 and 2026-10-11.

**A note on when this was written.** This file used to be titled "Pre-registration (frozen
2026-10-04, before any model comparison)". The dates are mine. In the version history the file was
empty in the commit of 2026-10-04 and first held this text in a commit of 2026-10-06, after the
first H1 and shared-evidence runs had been seen. The addenda were not committed before the coverage
experiment ran. Nothing independent records when the text was written, so this is an analysis plan
written before my first comparison, not a verifiable pre-registration.

## Question

With at most 31 revealed labels per subject-benchmark pair, how much do (1) an online per-pair
state, (2) subject information carried over from other benchmarks and (3) selective acquisition
improve calibrated item-level prediction on unseen benchmarks?

## Data and protocol

- Main benchmarks: matharena, multi_swebench, real_webagents, researchcodebench, swe_rebench
  (responses that are exactly 0 or 1). mmdocrag is excluded because its graded scores have no
  obvious 0/1 rule. A threshold analysis would be a separate sensitivity study.
- Eligible pairs: at least 80 distinct items; 50/50 acquisition and evaluation split; budgets 0,
  1, 3, 7, 15, 31; one random trial revealed per acquired item.
- Evaluation is leave-one-benchmark-out. Seed 0 for development, seeds 0 to 4 for reported results.
- Main metric: Brier ALC with weights 0.1, 0.2, 0.2, 0.2, 0.2, 0.1. ECE is a diagnostic.
- Unit of analysis: the subject-benchmark pair, clustered by benchmark. Each benchmark is reported
  separately along with the pair-weighted mean. With four or five benchmarks, claims across
  benchmarks are descriptive, not inferential.

## Hypotheses

- **H1.** Adding a subject ability carried over from the same model name in other benchmarks
  lowers Brier ALC compared with the same model without it. Supported only if the difference is
  negative in a majority of the benchmarks where ability is known for at least half of the
  subjects, and the pooled pair-level bootstrap 95% interval excludes zero.
- **H2.** Uncertainty-driven acquisition lowers Brier ALC compared with random acquisition. Not
  tested.
- **H3.** The gap between streaming and pool-based acquisition grows as the ratio of pool size to
  budget shrinks. Not tested.
- **H4.** A threshold that adapts to surprise helps compared with a fixed threshold. Not tested.
- **Gate for H2 to H4.** A selection policy is only developed if a hindsight-oracle selection
  improves Brier ALC over random by more than 0.01 (a value I proposed before computing the
  oracle). The oracle was never computed.

## Models compared for H1

- `offset_logit_s0`: per-pair offset fitted by MAP, no subject information.
- `offset_logit_s1`: the same plus the transferred subject ability at full scale (primary).
- `offset_logit_s0.5`: a half-scale variant (exploratory).
- Fixed settings: prior sd tau 1.0, shrinkage 5.0, ability clipped at rates 0.02 to 0.98.

## Rules

- Any change to these settings is logged in `docs/decisions.md` with its date.
- Analyses that are not listed here are labelled exploratory in the report.

## Addendum, 2026-10-09 (written before any interval was computed)

Estimand: for a subject-benchmark pair, theta is the mean of the recorded outcomes over the
pair's evaluation pool, counting every trial. Labels come from the acquisition pool by random
acquisition.

**H5** (confirmatory): in leave-one-benchmark-out evaluation, the 90% empirical-Bayes predictive
interval for theta has coverage between 85% and 95% at budgets 7, 15 and 31 in each
multi-subject benchmark. Comparators: Wilson, Clopper-Pearson, and a Wilson interval with a
finite-population correction. Settings are fixed to the existing code. H5 fails if the whole
cluster-bootstrap interval for coverage lies outside 85-95% in any benchmark at any budget of 7
or more. Everything else is exploratory. H2 to H4 remain untested.

### Amendment, 2026-10-11 (before any interval was computed)

The third comparator is a pool-aware Wilson interval. The target, the mean over the evaluation
pool, is itself a mean over a finite pool, so its sampling variance adds to that of the labels.
For two disjoint random subsets of a finite population, the variance of the difference of their
means is S^2 (1/n + 1/N), so the interval uses the effective size n*N/(n+N), where N is the size
of the evaluation pool. This is my own derivation and the coverage experiment itself is the check
on it. Everything else in the addendum is unchanged.
