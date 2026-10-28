import numpy as np

from sparse_observer.simulator.protocol import labels_at_budget, pair_rng, split_pair

ITEMS = [f"item_{i}" for i in range(81)]  # an odd count


def test_split_is_disjoint_and_covers_all_items():
    split = split_pair("b", "s", ITEMS)
    assert not set(split.acquisition) & set(split.evaluation)
    assert set(split.acquisition) | set(split.evaluation) == set(ITEMS)


def test_odd_count_gives_the_extra_item_to_acquisition():
    split = split_pair("b", "s", ITEMS)
    assert len(split.acquisition) == 41
    assert len(split.evaluation) == 40


def test_even_count_splits_evenly():
    split = split_pair("b", "s", ITEMS[:80])
    assert len(split.acquisition) == len(split.evaluation) == 40


def test_split_is_deterministic():
    assert split_pair("b", "s", ITEMS, seed=3) == split_pair("b", "s", ITEMS, seed=3)


def test_split_does_not_depend_on_input_order():
    assert split_pair("b", "s", ITEMS) == split_pair("b", "s", list(reversed(ITEMS)))


def test_seed_and_pair_change_the_split():
    base = split_pair("b", "s", ITEMS, seed=0)
    assert base != split_pair("b", "s", ITEMS, seed=1)
    assert base != split_pair("b", "t", ITEMS, seed=0)
    assert base != split_pair("c", "s", ITEMS, seed=0)


def test_pair_rng_is_reproducible():
    first = pair_rng("b", "s", 0).random(3)
    second = pair_rng("b", "s", 0).random(3)
    assert np.array_equal(first, second)


def test_budgets_are_nested_prefixes():
    acquired = [(f"i{k}", k % 2) for k in range(31)]
    assert labels_at_budget(acquired, 0) == []
    assert labels_at_budget(acquired, 7) == acquired[:7]
    assert labels_at_budget(acquired, 31) == acquired
