# Literature notes

Read depth: FULL = full text read; ABSTRACT = abstract and secondary summaries only;
NOT READ = known from search snippets. Claims in the report may rest only on FULL entries
or must be worded as reported by the abstract.

## Closest work

### Active Evaluation Acquisition (AEA), arXiv:2410.05952. FULL
- Problem: estimate a new model's benchmark score from a few acquired prompt outcomes.
- Target: model-level aggregate score (item-level predictions are internal).
- Setting: pool-based, sequential, fixed budget K.
- Before selection: embeddings of all candidate prompts and scores acquired so far.
  After: the selected prompt's score.
- Model: neural process (Set Transformer, latent variable) with an RL acquisition policy.
- Data: Open LLM Leaderboard (2,084 models, 28,659 prompts), MMLU, HELM-Lite, AlpacaEval,
  Chatbot Arena.
- Metric: absolute error of the aggregate score, three seeds.
- Result: the RL policy was best; uncertainty and information-gain policies failed to
  explore; random policies were competitive; static policies degraded under model shift.
- Limits: continual learning left to future work; no item-level calibration metric.
- Relation: direct (acquisition plus prediction), but pool-based and aggregate-score.

### "100 instances is all you need", arXiv:2409.03563. FULL
- Problem: predict a new LLM's success on unseen instances from a few reference instances.
- Target: item-level binary success. Metric: AUC (does not assess calibration).
- Setting: static reference set (not adaptive); selectors tried: clustering, factor
  analysis, IRT, random.
- Model: "generic assessor": XGBoost or L1 logistic regression on instance features
  plus the new LLM's success vector on the references.
- Data: HELM-Lite (30 LLMs, 4,285 instances), KindsOfReasoning (14 OpenAI models,
  37,529 instances).
- Result: generic assessor matches model-specific assessors in distribution; random
  references match advanced selectors; out-of-distribution predictability is low.
- Relation: direct on the prediction side; no acquisition.

### ATLAS, arXiv:2511.04689. FULL
- Problem: estimate model ability with a few items using computerized adaptive testing.
- Target: latent ability; accuracy is reconstructed from it.
- Setting: pool-based CAT with precision-based stopping (minimum 30 items).
- Model: 3PL IRT on a calibrated item bank from thousands of models; Fisher-information
  selection with randomization among the top items.
- Result: 30-89 items matched whole-bank ability; moderate degradation under family and
  temporal holdout.
- Limits: needs item parameters from many models, so no new items or benchmarks.
- Relation: adjacent; the classical CAT baseline for pool-based selection.

### Fluid Language Model Benchmarking (Hofmann et al.), arXiv:2509.11106. ABSTRACT
- IRT (2PL) fitted on existing evaluation results plus dynamic Fisher-information item
  selection; reported effects: IRT improves validity, dynamic selection reduces variance;
  fixed-length adaptive design (as described by the ATLAS comparison).
- Needs a calibrated item bank; evaluates ability estimation, not item-level prediction.
- Relation: adjacent. To read: whether any experiment uses new items or benchmarks.

### Reliable and Efficient Amortized Model-based Evaluation (Truong et al.), arXiv:2503.13335. ABSTRACT
- Trains a model that predicts question difficulty from its content, so new items can be
  calibrated without model responses; reports generalization across datasets when the
  question embedding and dataset context are used; 22 benchmarks, 172-183 LMs.
- Relation: direct for my cold-start text-difficulty result. My ridge model had no dataset
  context, my benchmarks are agentic, and I had 31-104 subjects, so the setups differ.
  To read: how cross-dataset generalization was tested and what the held-out unit was.

## Not read
- tinyBenchmarks (IRT-anchored subsets); metabench; Active Testing (Kossen et al.) and
  LLM extensions; Predictable AI / ADeLe assessors; WILD (multidimensional IRT with
  optimal experimental design; PDF extraction failed).
- Stream-based selective sampling in active learning (classical; budgeted threshold
  strategies exist). Not yet searched: the secretary-problem and online-selection
  literature.

## What this project may and may not claim
- May not claim: novelty of adaptive evaluation, IRT-based prediction, or acquisition.
- The remaining claims are empirical findings under one protocol: base-rate saturation at
  low budgets, the headroom decomposition, cross-benchmark transfer failure for text,
  the break-even density of shared item evidence.
- Checked but not verified: whether similar break-even analyses exist. In writing I say "I did
  not find" one.