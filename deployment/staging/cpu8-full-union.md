# Eight-thread Full Union staging acceptance

Two ten-minute open-loop windows passed the local steady-load criteria at a nominal 2.5 RPS. A repeated ladder passed at 24 and 32 simultaneous requests. At 40 requests one of two trials passed. Both 46-request trials fell below the proposed 95% burst success threshold.

| Measurement | First run | Second run |
| --- | ---: | ---: |
| Actual scheduled arrivals per second | 2.470 | 2.435 |
| Steady HTTP successes | 1,482/1,482 | 1,461/1,461 |
| Scheduled-to-response P95 | 1.590 s | 1.232 s |
| Successful within the steady window | 100.00% | 99.93% |
| 24 simultaneous successes | 24/24 | 24/24 |
| 32 simultaneous successes | 32/32 | 32/32 |
| 40 simultaneous successes | 40/40 | 35/40 |
| 46 simultaneous successes | 37/46 | 34/46 |
| 46 paced successes | 46/46 | 46/46 |
| 46 paced time from first release to last response | 11.661 s | 11.477 s |

All 26 burst failures were gateway 504s while waiting for upstream response headers. Fourteen requests had yet to begin inference and twelve were already running. The trace audit recorded zero inference starts after 504 responses. The 32-request result describes the largest tested burst size that passed both repetitions. It applies to the recorded machine and query mix.

The run used API commit `0d38c9d`, a single worker, eight CPU inference threads, a model batch size of 32, and the full candidate union over 8,841,823 passages. The [recorded protocol](../../artifacts/evaluations/cpu8-full-union/protocol.json) contains the versions, model revisions, binary and query-pool digests, and load settings.

## Check the recorded results

From the repository root, run with Python 3.13.

```bash
python -m backend.scripts.reproduce_cpu8_staging --check
```

This reads 3,591 reduced HTTP observations across 12 trials, including steady warmup. It recomputes status and success counts, scheduled-to-response P95, in-window completions, and burst cohort timing. The [expected results](../../artifacts/evaluations/cpu8-full-union/expected.json) also record the original UUID and search-history audit totals. Repeating that audit uses live gateway and API traces.

## Run against your staging instance

Follow the [staging setup](README.md) with an isolated full-corpus database, a loopback HTTPS gateway, and its matching CA and netrc. Build or provision the MS MARCO v1 corpus, PostgreSQL indexes and C++ corpus using the [full-corpus setup](../../README.md#reproduction). Use the recorded model revisions and native binary digest for a matched replay. Record the hardware and any version differences alongside the output. Keep the isolated staging history writable so it can be reconciled with the HTTP results.

From the repository root, export the [CPU profile](cpu8-full-union.env) before starting the protected API wrapper. Set the database URL, proxy token, and corpus paths as described in the staging setup. Set `STAGING_RERANK_TELEMETRY` to a new writable file path.

```bash
set -a
. deployment/staging/cpu8-full-union.env
set +a
PYTHONPATH="$PWD" python deployment/staging/serve_api.py --port "${STAGING_API_PORT:-8010}"
```

In a second shell, set `STAGING_GATEWAY`, `STAGING_CA`, and `STAGING_NETRC` for the verified staging origin. The tracked [query pool](rerank-query-pool.json) supplies the input order. Save each run to a fresh output directory.

```bash
mkdir -p results/cpu8-staging-replay
run_rerank() {
  python deployment/staging/benchmark_rerank.py \
    --gateway "$STAGING_GATEWAY" --ca "$STAGING_CA" --netrc "$STAGING_NETRC" \
    --route batched --timeout 15 "$@"
}

run_rerank --rate 2.5 --seconds 600 --warmup 60 --seed 141025 \
  --output results/cpu8-staging-replay/steady-1.json
sleep 12
run_rerank --rate 2.5 --seconds 600 --warmup 60 --seed 141026 \
  --output results/cpu8-staging-replay/steady-2.json
sleep 12

for size in 24 32 40 46; do
  run_rerank --burst "$size" --output "results/cpu8-staging-replay/burst-$size-1.json"
  sleep 12
done
run_rerank --burst 46 --burst-group-size 4 --burst-interval 1 \
  --output results/cpu8-staging-replay/paced-46-1.json
sleep 12
for size in 46 40 32 24; do
  run_rerank --burst "$size" --output "results/cpu8-staging-replay/burst-$size-2.json"
  sleep 12
done
run_rerank --burst 46 --burst-group-size 4 --burst-interval 1 \
  --output results/cpu8-staging-replay/paced-46-2.json
```

At steady load use P95 ≤ 3 seconds, failure rate ≤ 1%, and an in-window success share ≥ 95%. For the burst diagnostic, use at least 95% HTTP success, zero new inference after a gateway 504, and a history row for each successful response. Join client `trace_id` to the Caddy `uuid` and API trace to audit timeout behavior. With pacing, each request's latency starts at its own scheduled release. The final cohort response uses the first release as its origin.
