"""Interface shared by the predictors of the simulation."""


class Predictor:
    """A predictor is fitted on training benchmarks only and then predicts one pair from the
    labels revealed so far.
    """
    name = "predictor"

    def fit(self, train_benchmarks, rules):
        """Fit on the training benchmarks; the evaluated benchmark is never used (leave-one-benchmark-out)."""
        return self

    def predict_pair(self, benchmark, subject_id, item_ids, labeled):
        """Probabilities for the evaluation items; `labeled` is a list of (item, label) revealed so far."""
        raise NotImplementedError
