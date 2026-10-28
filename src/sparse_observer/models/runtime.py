"""Standalone version of the item-pooled predictor.

It uses only the standard library, so the file can be copied on its own.
predict_probability() takes one [subject, item] input and all labels revealed so far
(from many pairs) and returns P(correct). Items are matched across pairs by a hash of
their text. With no labels from other subjects, or with slope 0, it reduces to the
per-pair offset model. tests/test_models.py checks it against the simulator version.
"""

import hashlib
import math
from functools import lru_cache

CLIP = 0.02
SHRINK = 2.0


def logit(p):
    p = min(max(p, CLIP), 1 - CLIP)
    return math.log(p / (1 - p))

def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)

@lru_cache(maxsize=None)
def text_key(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def item_key(item):
    return text_key(item.get("item_content") or "")

def split_evidence(subject, item, labeled):
    benchmark = item.get("benchmark_id")
    own, pooled = [], []
    for entry in labeled or []:
        (other_subject, other_item), label = entry
        if other_item.get("benchmark_id") != benchmark:
            continue
        record = (item_key(other_item), float(label))
        (own if other_subject == subject else pooled).append(record)
    return own, pooled


def item_ease(pooled):
    if not pooled:
        return {}
    q = sum(label for _, label in pooled) / len(pooled)
    base = logit(q)
    sums, counts = {}, {}
    for key, label in pooled:
        sums[key] = sums.get(key, 0.0) + label
        counts[key] = counts.get(key, 0) + 1
    return {k: logit((sums[k] + SHRINK * q) / (counts[k] + SHRINK)) - base for k in sums}


def map_offset(own, ease, slope, prior_logit, tau):
    # Newton steps on the penalized log-likelihood of the pair's own labels
    x = [slope * ease.get(key, 0.0) for key, _ in own]
    y = [label for _, label in own]
    b = prior_logit
    for _ in range(25):
        grad = (b - prior_logit) / tau**2
        hess = 1 / tau**2
        for xi, yi in zip(x, y):
            p = sigmoid(b + xi)
            grad += p - yi
            hess += p * (1 - p)
        b -= grad / hess
    return b


def predict_probability(input, labeled, params):
    subject, item = input
    own, pooled = split_evidence(subject, item, labeled)
    ease = item_ease(pooled)
    slope = params["slope"]
    b = map_offset(own, ease, slope, params["prior_logit"], params["tau"])
    z = b + slope * ease.get(item_key(item), 0.0)
    return min(max(sigmoid(z), 0.0), 1.0)