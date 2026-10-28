# Protocol notes

How my simulation maps to the task description it is based on (the public starting-kit README
of the NeurIPS 2026 Predictive AI Evaluation Competition, which I read on 2026-10-05). Items
marked "assumption" are not stated in that description.

## Stated in the description

- Eligible pairs have at least 80 distinct items.
- Each pair gets its own persistent random 50/50 split into an acquisition pool and a fixed
  evaluation pool. Repeated runs of an item stay in the same pool. If the count is odd, the extra
  item goes to acquisition.
- Acquisition candidates arrive one at a time in a fixed random order. At each one the predictor
  can ask for the recorded 0/1 response or skip it for good.
- At most 31 labels per pair. The budgets 0, 1, 3, 7, 15 and 31 are nested prefixes of one
  sequence, and every pair finishes a budget before any pair gets labels for the next one.
- The predictor sees all evidence bought so far across the sampled pairs, each entry tagged with
  its benchmark. A prediction must not depend on earlier calls.
- Every recorded run of an evaluation item is scored, and identical inputs share one prediction.
- A draw consists of complete (benchmark, subject) groups, up to about 1,000 source items.

## Assumptions in the simulation

- Which trial is revealed when an acquired item has several recorded responses: one is picked at
  random.
- Items without a valid 0/1 label are dropped before the split.
- The shuffle seed for the split is derived from the benchmark, the subject and a global seed. The
  organizers' shuffle is different, so individual pairs will not match their splits.
- In the experiments with shared evidence, the other pairs on the same benchmark are sampled at
  random from the eligible pairs (1, 3, 10 or 30 of them). How many pairs share a benchmark in
  the real evaluation is unknown to me.

## Open questions

- How graded responses (for example judge scores in steps of 0.04) become 0/1 targets.
- Whether an item can be an evaluation target for one subject and an acquisition label for
  another on the same benchmark. The description implies yes, because each pair is split on its
  own.
