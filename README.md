# Sparse Observer

This is a small research project.It is a careful look at one question, with the failures written down
next to the successes.

The task setup is based on the public description of the NeurIPS 2026 Predictive AI Evaluation
Competition (AIMS Lab, Stanford). This repository is an independent study, not an entry, and
none of the numbers in it are leaderboard scores.

## The problem

Take an AI system that has been tested on a benchmark: a long list of tasks, each marked right
or wrong. Now suppose I can only see between 0 and 31 of its results, and I have to guess how it
did on the tasks I have not seen. For each guess I give a probability, like "70% it got this one
right".

The guesses are scored with the Brier score (the average squared error of the probabilities,
lower is better) at 0, 1, 3, 7, 15 and 31 revealed results. The six scores are then combined into
one number, Brier ALC, which gives more weight to the small numbers of results. Guessing 50% every
time scores 0.25. The weights and the rules I followed are in `docs/protocol_notes.md`, including
the places where I had to make an assumption.

## What I wanted to find out

- How much does each revealed result help?
- Does knowing how the same AI did on other benchmarks help?
- Can the text of a task tell me how hard it is, even on a benchmark I have not seen?
- Do results from other AIs on the same benchmark help?
- When I say "the true accuracy is probably between 40% and 60%", can that statement be trusted?

## What I did

I rebuilt the task as a small simulation on the public training data, then tried simple
statistical predictors on it. Every predictor was always tested on a benchmark it had never seen.
I wrote down my hypotheses before my first comparison, kept a dated log of decisions and mistakes,
and saved all results so that every number in the report can be traced to a file.

## What I found

All numbers come from my simulation. They use five benchmarks, 221 subject-benchmark pairs and
five random seeds. Lower is better.

- Guessing 0.5 scores 0.2500. A simple predictor that learns one success rate per pair from the
  revealed results scores 0.2096. That is the one clear gain: 0.0407 better (95% interval 0.0299
  to 0.0508).
- Three other ways of estimating that rate score between 0.2108 and 0.2160, so none of them beats
  the simple one.
- Using how the same AI did on other benchmarks did not help enough to count. My hypothesis H1 was
  not supported (change of -0.0028, interval -0.0091 to +0.0038).
- An oracle that also knows how hard each task is, taken from other AIs' results, would reach a
  Brier score of 0.094 instead of 0.176 for the best constant guess per pair. That oracle uses
  information I would not have in practice, so it is an upper bound and not something I can build.
- The text of a task does not predict its difficulty on a new benchmark. The test I set in advance
  (sentence embeddings) was met on none of the four benchmarks.
- Results from other AIs on the same benchmark only help once there are about ten of them. With
  one other AI the gain is tiny (0.0005).
- My interval method passed the test I set for it: it contained the true accuracy between 86% and
  93% of the time at 7, 15 and 31 results (I allowed 85% to 95%). Two of the twelve checks were
  marginal. With 1 to 3 results the interval is too narrow on two benchmarks.

What I did not do: I did not test strategies for choosing which results to reveal, so those
hypotheses (H2 to H4) are still open. The details and dates are in `docs/decisions.md`.

The full write-up is in `paper/report.md`.

## The data

The data is the Measurement Data Bank (`aims-foundations/measurement-db` on Hugging Face). It is
gated: you have to ask for access on its page. I do not include it here and please do not upload
it anywhere either. The `data/` folder is in `.gitignore` so it cannot be committed by accident.

I only use the item, subject and response tables (48 files, about 64 MB). One benchmark, MMDocRAG,
is left out because its answers are judge scores and there is no obvious way to turn them into
right or wrong. That leaves 221 pairs with enough items (at least 80). Analyses that compare
different AIs on the same benchmark use 220 of them, because SWE-rebench has only one AI.

The files in `results/` and `tables/` are my own computed scores. Subject IDs in them are the
anonymous hashes from the dataset.

## How to run it

You need Python 3.12.

    pip install -r requirements.txt
    pip install -e .
    hf auth login
    python scripts/download_data.py
    python scripts/download_data.py --execute

The first download command only lists the files. The second one downloads about 64 MB into
`data/raw`.

Run the tests:

    python -m pytest tests -q

Run everything, from data checks to figures (19 stages, about 21 minutes on my computer):

    python scripts/run_all.py

Each stage writes a log to `logs/run_all`. Useful options: `--list` shows the stages, `--only NAME`
runs the stages with NAME in their title, and `--with-embeddings` adds the optional text experiment
(install `requirements-embeddings.txt` first).

To rebuild only the report's figures and tables from the saved results:

    python scripts/make_report_assets.py

My result files were written on Windows with Windows line endings and later converted to Unix
ones. If you compare your rerun with them, compare the numbers (for example with pandas) rather
than file hashes.

## What is where

    configs/data.yaml        how answers become right or wrong, per benchmark
    docs/analysis_plan.md    what I planned to test, written before my first comparison
    docs/decisions.md        dated log of decisions, mistakes and results, grouped by topic
    docs/protocol_notes.md   how my simulation follows the task description
    literature/notes.md      related papers and how closely I read each one
    paper/report.md          the technical report
    scripts/                 downloading data, running everything, making figures
    src/sparse_observer/     the code: data, simulator, models, evaluation, experiments, features
    tests/                   checks for the metric, the splits, the models and the intervals
    results/, tables/, figures/   saved outputs behind the report

`src/sparse_observer/models/runtime.py` is a standard-library-only version of the item-level
model, written so it can be copied on its own.

## Limits you should know about

- My simulation follows the task description, but it is not the organizers' code. For example, when
  a task has several recorded attempts I reveal one at random, and the description does not say
  what they do.
- Only four benchmarks have several AIs, and many of those are the same model run with different
  tools. My intervals account for that, but not for differences between benchmarks.
- I used seed 0 while developing, and it is also one of the five seeds in the results.
- The text test used one simple method and one embedding model. Other methods could behave
  differently.
- The `item_features` column is empty in the training data.
- The interval study is a calibration check inside my simulation. It says nothing about any
  hidden data.
- A few numbers in the report (the text experiment and one table) were saved only in my console
  output. Rerunning the `heterogeneity`, `item_features` and `text_transfer` stages (the last one
  with `--with-embeddings`) regenerates them in `logs/run_all`.

## About my plan and my log

My analysis plan was written before my first comparison, but it is not a verifiable
pre-registration: the version history only shows the text a few days later. I say this plainly in
`docs/decisions.md`, along with every correction I had to make, so the record is accurate rather
than impressive.

## License and citation

The code is under the MIT license (see `LICENSE`). The data keeps its own license and I do not
redistribute it. Please cite the dataset as:

Truong, N., Truong, S. T., Koyejo, S. The AI Measurement Data Bank. AIMS Lab, Stanford University, 2026.
