"""Integration tests using the real Cross-Encoder tokenizer."""

from backend.api.services.cross_encoder import get_cross_encoder
from backend.api.services.token_budget import estimate_request_tokens


def test_real_tokenizer_matches_direct_encoding():
    model = get_cross_encoder()
    tokenizer = model.tokenizer

    query = "What is dynamic batching?"

    candidates = [
        {"content": "Dynamic batching combines inference requests."},
        {"content": "A short document."},
        {"content": " ".join(["retrieval"] * 1000)},
    ]

    estimate = estimate_request_tokens(
        tokenizer=tokenizer,
        query=query,
        candidates=candidates,
        max_length=model.max_seq_length,
    )

    direct = tokenizer(
        [query] * len(candidates),
        [candidate["content"] for candidate in candidates],
        truncation=True,
        max_length=model.max_seq_length,
        padding=False,
        add_special_tokens=True,
        return_attention_mask=False,
    )

    lengths = [
        len(ids)
        for ids in direct["input_ids"]
    ]

    assert estimate.pair_count == len(candidates)
    assert estimate.total_tokens == sum(lengths)
    assert estimate.max_pair_tokens == max(lengths)
    assert max(lengths) <= 512
    assert lengths[-1] == 512
