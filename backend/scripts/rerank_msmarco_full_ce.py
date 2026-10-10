import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 10
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42

INPUT_PATH = Path(
    "results/msmarco_full/relevance/ce_candidates.json"
)

OUT_DIR = Path(
    "results/msmarco_full/relevance"
)

CHECKPOINT_PATH = (
    OUT_DIR / "ce_rerank_checkpoint.json"
)

RESULT_PATH = (
    OUT_DIR / "ce_rerank_results.json"
)

ARTIFACT_PATH = Path(
    "artifacts/msmarco_full/ce_rerank_summary.json"
)


def save_json(path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


def reciprocal_rank(ids, relevant):
    relevant = set(relevant)

    for rank, doc_id in enumerate(
        ids[:TOP_K],
        1,
    ):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg(ids, relevant):
    relevant = set(relevant)

    dcg = 0.0

    for rank, doc_id in enumerate(
        ids[:TOP_K],
        1,
    ):
        if doc_id in relevant:
            dcg += (
                1.0
                / math.log2(rank + 1)
            )

    ideal_hits = min(
        len(relevant),
        TOP_K,
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1,
        )
    )

    return dcg / idcg


def recall(ids, relevant):
    relevant = set(relevant)

    if not relevant:
        return 0.0

    return (
        len(
            set(ids[:TOP_K])
            & relevant
        )
        / len(relevant)
    )


def metrics(ids, relevant):
    return {
        "mrr_at_10":
            reciprocal_rank(
                ids,
                relevant,
            ),
        "ndcg_at_10":
            ndcg(
                ids,
                relevant,
            ),
        "recall_at_10":
            recall(
                ids,
                relevant,
            ),
    }


def summarize(rows, key):
    return {
        metric: statistics.fmean(
            row[key]["metrics"][metric]
            for row in rows
        )
        for metric in [
            "mrr_at_10",
            "ndcg_at_10",
            "recall_at_10",
        ]
    }


def percentile(values, p):
    values = sorted(values)

    pos = (
        (len(values) - 1)
        * p
    )

    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return values[lo]

    f = pos - lo

    return (
        values[lo] * (1 - f)
        + values[hi] * f
    )


def latency_summary(rows, key):
    values = [
        row[key]["latency_ms"]
        for row in rows
    ]

    return {
        "mean_ms":
            statistics.fmean(values),
        "p50_ms":
            percentile(values, 0.50),
        "p95_ms":
            percentile(values, 0.95),
    }


def paired_bootstrap(
    rows,
    metric,
):
    dense = np.array([
        row["dense_ce"]["metrics"][
            metric
        ]
        for row in rows
    ])

    union = np.array([
        row["union_ce"]["metrics"][
            metric
        ]
        for row in rows
    ])

    delta = union - dense

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    n = len(delta)

    samples = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )

    for i in range(
        BOOTSTRAP_SAMPLES
    ):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        samples[i] = (
            delta[idx].mean()
        )

    lo, hi = np.quantile(
        samples,
        [0.025, 0.975],
    )

    wins = int(
        np.sum(delta > 1e-12)
    )

    ties = int(
        np.sum(
            np.abs(delta)
            <= 1e-12
        )
    )

    losses = int(
        np.sum(delta < -1e-12)
    )

    return {
        "dense_mean":
            float(dense.mean()),
        "union_mean":
            float(union.mean()),
        "delta":
            float(delta.mean()),
        "ci95": [
            float(lo),
            float(hi),
        ],
        "wins":
            wins,
        "ties":
            ties,
        "losses":
            losses,
    }


def main():
    data = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    queries = data["queries"]

    checkpoint = {}

    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(
            CHECKPOINT_PATH.read_text(
                encoding="utf-8"
            )
        )

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(
        f"Cross-Encoder : {MODEL_NAME}"
    )
    print(
        f"Device        : {device}"
    )
    print(
        f"Queries       : {len(queries)}"
    )
    print(
        f"Cached        : {len(checkpoint)}"
    )

    model = CrossEncoder(
        MODEL_NAME,
        device=device,
    )

    started_all = time.perf_counter()

    for i, item in enumerate(
        queries,
        1,
    ):
        qid = str(
            item["query_id"]
        )

        if qid in checkpoint:
            continue

        query = item["query"]

        relevant = [
            int(x)
            for x in item[
                "relevant_document_ids"
            ]
        ]

        dense_candidates = (
            item["dense"]["candidates"]
        )

        lexical_candidates = (
            item["lexical"]["candidates"]
        )

        dense_map = {
            int(x["document_id"]):
                x["content"]
            for x in dense_candidates
        }

        lexical_map = {
            int(x["document_id"]):
                x["content"]
            for x in lexical_candidates
        }

        union_map = dict(dense_map)
        union_map.update(lexical_map)

        # --------------------------------
        # Dense Top-50 -> CE
        # --------------------------------

        dense_pairs = [
            (
                query,
                content,
            )
            for content
            in dense_map.values()
        ]

        dense_ids = list(
            dense_map.keys()
        )

        t0 = time.perf_counter()

        dense_scores = model.predict(
            dense_pairs,
            batch_size=64,
            show_progress_bar=False,
        )

        dense_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        dense_ranked = [
            doc_id
            for doc_id, _
            in sorted(
                zip(
                    dense_ids,
                    dense_scores,
                ),
                key=lambda x: (
                    -float(x[1]),
                    x[0],
                ),
            )
        ]

        # --------------------------------
        # Union -> CE
        # --------------------------------

        union_ids = list(
            union_map.keys()
        )

        union_pairs = [
            (
                query,
                union_map[doc_id],
            )
            for doc_id in union_ids
        ]

        t0 = time.perf_counter()

        union_scores = model.predict(
            union_pairs,
            batch_size=64,
            show_progress_bar=False,
        )

        union_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        union_ranked = [
            doc_id
            for doc_id, _
            in sorted(
                zip(
                    union_ids,
                    union_scores,
                ),
                key=lambda x: (
                    -float(x[1]),
                    x[0],
                ),
            )
        ]

        dense_top10 = (
            dense_ranked[:TOP_K]
        )

        union_top10 = (
            union_ranked[:TOP_K]
        )

        dense_set = set(
            dense_map
        )

        lexical_only = (
            set(lexical_map)
            - dense_set
        )

        relevant_set = set(
            relevant
        )

        lexical_only_relevant = (
            lexical_only
            & relevant_set
        )

        promoted = (
            lexical_only_relevant
            & set(union_top10)
        )

        checkpoint[qid] = {
            "query_id": qid,
            "query": query,
            "relevant_document_ids":
                relevant,

            "dense_ce": {
                "top10":
                    dense_top10,
                "latency_ms":
                    dense_ms,
                "metrics":
                    metrics(
                        dense_top10,
                        relevant,
                    ),
            },

            "union_ce": {
                "top10":
                    union_top10,
                "latency_ms":
                    union_ms,
                "metrics":
                    metrics(
                        union_top10,
                        relevant,
                    ),
            },

            "analysis": {
                "dense_candidate_count":
                    len(dense_map),

                "lexical_candidate_count":
                    len(lexical_map),

                "union_candidate_count":
                    len(union_map),

                "lexical_only_count":
                    len(lexical_only),

                "lexical_only_relevant":
                    sorted(
                        lexical_only_relevant
                    ),

                "promoted_lexical_only_relevant":
                    sorted(promoted),
            },
        }

        save_json(
            CHECKPOINT_PATH,
            checkpoint,
        )

        if (
            i == 1
            or i % 10 == 0
            or i == len(queries)
        ):
            elapsed = (
                time.perf_counter()
                - started_all
            )

            print(
                f"[{i:>3}/{len(queries)}] "
                f"qid={qid}  "
                f"denseCE={dense_ms:.1f}ms  "
                f"unionCE={union_ms:.1f}ms  "
                f"union={len(union_map):>3}  "
                f"elapsed={elapsed:.1f}s"
            )

    rows = [
        checkpoint[
            str(q["query_id"])
        ]
        for q in queries
    ]

    dense_summary = summarize(
        rows,
        "dense_ce",
    )

    union_summary = summarize(
        rows,
        "union_ce",
    )

    paired = {
        metric: paired_bootstrap(
            rows,
            metric,
        )
        for metric in [
            "mrr_at_10",
            "ndcg_at_10",
            "recall_at_10",
        ]
    }

    lexical_rescue_queries = 0
    lexical_rescue_docs = 0
    promoted_queries = 0
    promoted_docs = 0

    for row in rows:
        rel = row["analysis"][
            "lexical_only_relevant"
        ]

        pro = row["analysis"][
            "promoted_lexical_only_relevant"
        ]

        if rel:
            lexical_rescue_queries += 1
            lexical_rescue_docs += len(rel)

        if pro:
            promoted_queries += 1
            promoted_docs += len(pro)

    rescue = {
        "queries_with_lexical_only_relevant":
            lexical_rescue_queries,

        "lexical_only_relevant_documents":
            lexical_rescue_docs,

        "queries_with_promoted_lexical_only_relevant":
            promoted_queries,

        "promoted_lexical_only_relevant_documents":
            promoted_docs,
    }

    output = {
        "protocol": {
            "name":
                "msmarco-full-ce-rerank-v1",

            "corpus_documents":
                8_841_823,

            "queries":
                len(rows),

            "candidate_source":
                str(INPUT_PATH),

            "candidate_k":
                50,

            "candidate_generation_requested_ef_search":
                data["protocol"][
                    "requested_ef_search"
                ],

            "candidate_generation_effective_ef_search":
                data["protocol"][
                    "effective_ef_search"
                ],

            "cross_encoder":
                MODEL_NAME,

            "top_k":
                TOP_K,

            "bootstrap_samples":
                BOOTSTRAP_SAMPLES,

            "bootstrap_seed":
                BOOTSTRAP_SEED,
        },

        "summary": {
            "dense_ce":
                dense_summary,

            "union_ce":
                union_summary,

            "latency": {
                "dense_ce":
                    latency_summary(
                        rows,
                        "dense_ce",
                    ),

                "union_ce":
                    latency_summary(
                        rows,
                        "union_ce",
                    ),
            },

            "paired_bootstrap":
                paired,

            "lexical_rescue":
                rescue,
        },

        "per_query":
            rows,
    }

    save_json(
        RESULT_PATH,
        output,
    )

    compact = {
        "protocol":
            output["protocol"],

        "summary":
            output["summary"],
    }

    save_json(
        ARTIFACT_PATH,
        compact,
    )

    print(
        "\n=== Full MS MARCO "
        "Cross-Encoder Results ==="
    )

    for name, s in [
        (
            "Dense CE",
            dense_summary,
        ),
        (
            "Union CE",
            union_summary,
        ),
    ]:
        print(
            f"{name:<10} "
            f"MRR@10={s['mrr_at_10']:.4f}  "
            f"nDCG@10={s['ndcg_at_10']:.4f}  "
            f"Recall@10={s['recall_at_10']:.4f}"
        )

    print(
        "\n=== Union CE - Dense CE ==="
    )

    for metric, result in paired.items():
        print(
            f"{metric:<12} "
            f"delta={result['delta']:+.4f}  "
            f"95% CI=["
            f"{result['ci95'][0]:+.4f}, "
            f"{result['ci95'][1]:+.4f}]  "
            f"W/T/L="
            f"{result['wins']}/"
            f"{result['ties']}/"
            f"{result['losses']}"
        )

    print(
        "\n=== Lexical Rescue ==="
    )

    for key, value in rescue.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\n=== CE Latency ==="
    )

    for name in [
        "dense_ce",
        "union_ce",
    ]:
        s = output[
            "summary"
        ]["latency"][name]

        print(
            f"{name:<10} "
            f"mean={s['mean_ms']:.1f}ms  "
            f"P50={s['p50_ms']:.1f}ms  "
            f"P95={s['p95_ms']:.1f}ms"
        )

    print(
        f"\nSaved:"
        f"\n  {RESULT_PATH}"
        f"\n  {ARTIFACT_PATH}"
    )


if __name__ == "__main__":
    main()
