import numpy as np
import pytest

from sparse_observer.models import runtime
from sparse_observer.models.item_pooled import ItemPooled
from sparse_observer.models.online_state import OffsetLogit, sigmoid
from sparse_observer.models.pooled import PooledOffset

PRIOR = -0.4
TAU = 1.0
PARAMS = {"prior_logit": PRIOR, "tau": TAU, "slope": 1.0}
SUBJECT = {"normalized_name": "a"}
OTHER = {"normalized_name": "b"}


def item(text, benchmark="b"):
    return {"item_content": text, "benchmark_id": benchmark}


def evidence(subject, labels, benchmark="b"):
    return [[[subject, item(f"text {i}", benchmark)], y] for i, y in labels]


def offset_model(tau=TAU, prior=PRIOR):
    model = OffsetLogit(scale=0, tau=tau)
    model.prior_logit = prior
    model._names = {"b": {}}  # skip reading subject tables from disk
    return model


def item_model(slope, tau=TAU, prior=PRIOR):
    model = ItemPooled(slope=slope)
    model.prior_logit = prior
    model.bench_spread = 0.0
    model.pair_spread = tau
    return model


def test_flat_prior_reproduces_the_sample_mean():
    model = offset_model(tau=100.0, prior=0.0)
    labels = list(enumerate([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
    assert model.predict_pair("b", "s", [0], labels)[0] == pytest.approx(3 / 15, abs=1e-3)


def test_no_labels_returns_the_prior():
    model = offset_model()
    assert model.predict_pair("b", "s", [0], [])[0] == pytest.approx(sigmoid(PRIOR))


def test_more_successes_raise_the_prediction():
    model = offset_model()
    low = model.predict_pair("b", "s", [0], [(0, 0), (1, 0), (2, 1)])[0]
    high = model.predict_pair("b", "s", [0], [(0, 1), (1, 1), (2, 1)])[0]
    assert high > low


def test_pooled_offset_weights_other_pairs_labels():
    model = PooledOffset(weight=0.5)
    model.prior_logit = 0.0
    model.bench_spread = 100.0
    model.pair_spread = 100.0
    own = list(enumerate([1, 0, 0, 0, 1]))
    p = model.predict_pair("b", "s", [0], own, [0, 0, 0, 0, 0])[0]
    assert p == pytest.approx(2 / 7.5, abs=1e-3)


def test_item_effect_moves_known_items_and_leaves_unseen_items():
    model = item_model(slope=1.0, tau=100.0, prior=0.0)
    own = [(0, 1), (1, 0), (2, 0), (3, 1), (4, 0)]
    pooled = [(9, 1)] * 10 + [(8, 0)] * 10
    easy, hard, unseen = model.predict_items([9, 8, 7], own, pooled)
    assert easy > 0.4 > hard
    assert unseen == pytest.approx(0.4, abs=1e-3)


def test_slope_zero_ignores_pooled_labels():
    model = item_model(slope=0.0)
    own = [(0, 1), (1, 0), (2, 1)]
    with_pool = model.predict_items([0, 5], own, [(5, 1)] * 5)
    without = model.predict_items([0, 5], own, [])
    assert np.allclose(with_pool, without)


@pytest.mark.parametrize("slope", [0.0, 1.0])
@pytest.mark.parametrize("n_own,n_pooled", [(0, 0), (3, 0), (7, 10), (15, 40), (31, 100)])
def test_runtime_matches_the_simulator_model(slope, n_own, n_pooled):
    rng = np.random.default_rng(n_own * 1000 + n_pooled)
    n_items = 40
    texts = [f"text {i}" for i in range(n_items)]
    keys = [runtime.item_key(item(t)) for t in texts]
    own_ids = rng.choice(n_items, size=n_own, replace=False)
    pooled_ids = rng.choice(n_items, size=n_pooled, replace=True)
    own = [(int(i), int(rng.integers(2))) for i in own_ids]
    pooled = [(int(i), int(rng.integers(2))) for i in pooled_ids]
    labeled = [[[SUBJECT, item(texts[i])], y] for i, y in own]
    labeled += [[[OTHER, item(texts[i])], y] for i, y in pooled]
    params = dict(PARAMS, slope=slope)
    got = [runtime.predict_probability([SUBJECT, item(t)], labeled, params) for t in texts]
    want = item_model(slope).predict_items(
        keys, [(keys[i], y) for i, y in own], [(keys[i], y) for i, y in pooled]
    )
    assert np.allclose(got, want, atol=1e-9)


@pytest.mark.parametrize("n", [0, 1, 3, 7, 15, 31])
def test_runtime_at_slope_zero_equals_the_per_pair_offset_model(n):
    rng = np.random.default_rng(n)
    labels = [(i, int(rng.integers(2))) for i in range(n)]
    got = runtime.predict_probability(
        [SUBJECT, item("x")], evidence(SUBJECT, labels), dict(PARAMS, slope=0.0)
    )
    want = offset_model().predict_pair("b", "s", [0], labels)[0]
    assert got == pytest.approx(want, abs=1e-9)


def test_runtime_ignores_other_benchmarks():
    own = evidence(SUBJECT, [(0, 1), (1, 0)])
    elsewhere = evidence(SUBJECT, [(i, 1) for i in range(10)], benchmark="c")
    elsewhere += evidence(OTHER, [(i, 0) for i in range(10)], benchmark="c")
    a = runtime.predict_probability([SUBJECT, item("x")], own, PARAMS)
    b = runtime.predict_probability([SUBJECT, item("x")], own + elsewhere, PARAMS)
    assert a == b


def test_runtime_ignores_other_subjects_at_slope_zero():
    own = evidence(SUBJECT, [(0, 1), (1, 0)])
    pooled = evidence(OTHER, [(0, 1), (1, 1), (2, 1)])
    params = dict(PARAMS, slope=0.0)
    a = runtime.predict_probability([SUBJECT, item("x")], own, params)
    b = runtime.predict_probability([SUBJECT, item("x")], own + pooled, params)
    assert a == b


@pytest.mark.parametrize("label", [0, 1])
def test_runtime_returns_a_probability_for_extreme_evidence(label):
    labeled = evidence(SUBJECT, [(i, label) for i in range(31)])
    p = runtime.predict_probability([SUBJECT, item("x")], labeled, PARAMS)
    assert 0.0 <= p <= 1.0


def test_runtime_with_no_evidence_returns_the_prior():
    p = runtime.predict_probability([SUBJECT, item("x")], [], PARAMS)
    assert p == pytest.approx(sigmoid(PRIOR))
    assert runtime.predict_probability([SUBJECT, item("x")], None, PARAMS) == p
