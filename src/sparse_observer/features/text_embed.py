"""Can item difficulty be predicted from item text across benchmarks? (optional experiment)

TF-IDF and MiniLM embeddings with ridge regression; sentence-transformers is imported
only when embeddings are computed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

from sparse_observer.data.labels import load_labeled_responses, load_rules
from sparse_observer.data.loader import BENCHMARKS, load_table
from sparse_observer.models.online_state import logit

MAX_CHARS = 4000
SHRINK = 2.0
MIN_RESPONSES = 5
ALPHA_TFIDF = 3.0
ALPHA_EMBED = 1.0
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = Path("data/processed/item_embeddings.npz")
MULTI_SUBJECT = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
SKILL_THRESHOLD = 0.05


def item_table(benchmark, rules):
    """Ease of every item relative to its benchmark (shrunk log-odds), for items with at least 5 responses."""
    resp = load_labeled_responses(benchmark, rules).dropna(subset=["label"])
    if resp.empty:
        return None
    bench_rate = resp["label"].mean()
    stats = resp.groupby("item_id")["label"].agg(["sum", "size"])
    rate = (stats["sum"] + SHRINK * bench_rate) / (stats["size"] + SHRINK)
    # positive means easier than the benchmark average
    # positive means easier than the benchmark average
    ease = logit(rate) - logit(bench_rate)
    items = load_table(benchmark, "items", columns=["item_id", "content"]).set_index("item_id")
    out = pd.DataFrame({"ease": ease, "n_responses": stats["size"]}).join(items)
    out = out[out["n_responses"] >= MIN_RESPONSES].copy()
    out["text"] = out["content"].fillna("").str.slice(0, MAX_CHARS)
    out["benchmark"] = benchmark
    return out.reset_index()[["item_id", "benchmark", "text", "ease", "n_responses"]]


def skill(pred, target):
    """1 means perfect, 0 means no better than predicting the benchmark average, below 0 means worse."""
    # 1 means perfect, 0 means no better than the benchmark average, below 0 means worse
    return 1.0 - np.sum((target - pred) ** 2) / np.sum(target**2)


def embed_texts(texts):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return model.encode(list(texts), batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True)


def embeddings_by_benchmark(tables):
    keys = np.array([f"{b}|{i}" for b, t in tables.items() for i in t["item_id"]])
    matrix = None
    if CACHE.exists():
        cached = np.load(CACHE)
        if np.array_equal(cached["keys"], keys):
            matrix = cached["vectors"]
    if matrix is None:
        texts = [x for t in tables.values() for x in t["text"]]
        matrix = embed_texts(texts).astype(np.float32)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, keys=keys, vectors=matrix)
    out, start = {}, 0
    for benchmark, table in tables.items():
        out[benchmark] = matrix[start:start + len(table)]
        start += len(table)
    return out


def benchmark_weights(tables, train):
    """Weights so that every training benchmark carries the same total weight."""
    # each training benchmark carries the same total weight
    total = sum(len(tables[b]) for b in train)
    return np.concatenate([np.full(len(tables[b]), total / (len(train) * len(tables[b])))
                           for b in train])


def cross_scores(tables, features, held_out, alpha, sparse_text=False):
    """Train on the other benchmarks, test on the held-out one: Spearman correlation and skill."""
    train = [b for b in tables if b != held_out]
    y = np.concatenate([tables[b]["ease"].to_numpy() for b in train])
    w = benchmark_weights(tables, train)
    target = tables[held_out]["ease"].to_numpy()
    if sparse_text:
        vec = TfidfVectorizer(max_features=20000, min_df=3, sublinear_tf=True)
        x_train = vec.fit_transform(pd.concat([tables[b]["text"] for b in train]))
        x_test = vec.transform(tables[held_out]["text"])
    else:
        x_train = np.vstack([features[b] for b in train])
        x_test = features[held_out]
    pred = Ridge(alpha=alpha).fit(x_train, y, sample_weight=w).predict(x_test)
    return spearmanr(pred, target).statistic, skill(pred, target)


def within_scores(tables, features, held_out):
    """Five-fold cross-validation inside the held-out benchmark, as an upper reference."""
    target = tables[held_out]["ease"].to_numpy()
    cv = KFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(Ridge(alpha=ALPHA_EMBED), features[held_out], target, cv=cv)
    return spearmanr(pred, target).statistic, skill(pred, target)


def main():
    rules = load_rules()
    tables = {}
    for benchmark in BENCHMARKS:
        table = item_table(benchmark, rules)
        if table is not None and len(table) >= 50:
            tables[benchmark] = table
    features = embeddings_by_benchmark(tables)

    print("items: " + ", ".join(f"{b} {len(t)}" for b, t in tables.items()))
    print("\n                         tfidf cross        embedding cross    embedding within")
    print("held-out           items  rho     skill      rho     skill      rho     skill")
    passed = 0
    for benchmark in tables:
        t_rho, t_skill = cross_scores(tables, features, benchmark, ALPHA_TFIDF, sparse_text=True)
        e_rho, e_skill = cross_scores(tables, features, benchmark, ALPHA_EMBED)
        w_rho, w_skill = within_scores(tables, features, benchmark)
        if benchmark in MULTI_SUBJECT and e_skill > SKILL_THRESHOLD:
            passed += 1
        print(f"{benchmark:<18}{len(tables[benchmark]):>5}  {t_rho:>6.3f} {t_skill:>7.3f}"
              f"    {e_rho:>6.3f} {e_skill:>7.3f}    {w_rho:>6.3f} {w_skill:>7.3f}")

    print(f"\nembedding cross skill above {SKILL_THRESHOLD} in {passed} of "
          f"{len(MULTI_SUBJECT)} multi-subject benchmarks (criterion: at least 3)")


if __name__ == "__main__":
    main()
