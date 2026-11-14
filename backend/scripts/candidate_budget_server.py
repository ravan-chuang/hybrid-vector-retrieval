"""Run the candidate-budget experiment on a separate loopback server."""

import argparse
from typing import Literal, get_args

import uvicorn
from pydantic import BaseModel, Field

from backend.api.main import app
from backend.api.routers.search import retrieve_hybrid_candidates
from backend.api.schemas.search import HybridSearchRequest
from backend.api.services.cross_encoder import (
    get_cross_encoder,
    rerank_candidates_profiled,
    resolve_batch_size,
    resolve_device,
    union_candidates,
)
from backend.api.services.token_budget import estimate_request_tokens
from backend.api.services.weighted_rrf import weighted_rrf_score


Strategy = Literal["rrf40", "quota30+10", "rrf50", "union"]
STRATEGIES = get_args(Strategy)


class Query(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


class Comparison(Query):
    strategy: Strategy


def retrieve(query):
    result = retrieve_hybrid_candidates(HybridSearchRequest(
        query=query, corpus="benchmark", lexical_backend="cpp_bm25",
        stopword_filter=True, top_k=10, candidate_k=50, ef_search=160,
    ))
    return result["lexical_rows"], result["vector_rows"]


def select_candidates(lexical, dense, strategy):
    union = union_candidates(lexical, dense)
    if strategy == "union":
        return union

    by_id = {row["document_id"]: row for row in union}
    if strategy == "quota30+10":
        dense_ids = [row[0] for row in dense]
        dense_members = set(dense_ids)
        lexical_only = [row[0] for row in lexical if row[0] not in dense_members][:10]
        ids = dense_ids[:30] + lexical_only + dense_ids[30:40 - len(lexical_only)]
        return [by_id[pid] for pid in ids]

    return sorted(union, key=lambda row: (
        -weighted_rrf_score(
            row["lexical_rank"], row["vector_rank"],
            lexical_weight=0.4, dense_weight=0.6, rrf_k=60,
        ),
        row["document_id"],
    ))[:40 if strategy == "rrf40" else 50]


@app.post("/_budget/compare", include_in_schema=False)
def compare(payload: Comparison):
    lexical, dense = retrieve(payload.query)
    candidates = select_candidates(lexical, dense, payload.strategy)
    ranked, profile = rerank_candidates_profiled(payload.query, candidates)
    return {
        "top10": [row["document_id"] for row in ranked],
        "candidates": [row["document_id"] for row in candidates],
        "lexical": [row[0] for row in lexical],
        "dense": [row[0] for row in dense],
        "inference_ms": profile["rerank_inference_ms"],
        "device": resolve_device(),
        "model_batch_size": resolve_batch_size(),
    }


@app.post("/_budget/tokens", include_in_schema=False)
def tokens(payload: Query):
    lexical, dense = retrieve(payload.query)
    model = get_cross_encoder()
    results = {}
    for strategy in STRATEGIES:
        candidates = select_candidates(lexical, dense, strategy)
        results[strategy] = {
            "candidates": [row["document_id"] for row in candidates],
            "tokens": estimate_request_tokens(
                model.tokenizer, payload.query, candidates, model.max_seq_length,
            ).total_tokens,
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False)
