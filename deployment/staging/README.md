# Controlled staging

Run the staging gateway on a loopback-only host port. It serves the production Vite build and proxies `/api/` over the same HTTPS origin. The entire site requires an operator login. Internal benchmark routes and API schema pages are blocked by the gateway. This configuration grants authenticated operators access to document administration and search history.

## Components

| Component | Configuration |
| --- | --- |
| Frontend | Build with `VITE_API_BASE=/api` for same-origin requests. A separate staging API can instead supply its full URL at build time. |
| API | Run `deployment/staging/serve_api.py` from the repository root with that root on `PYTHONPATH`. Configure `DATABASE_URL`, `CPP_CORPUS_PATH`, and optional `RETRIEVAL_CPP_EXTENSION`. Set `RETRIEVAL_CORS_ORIGINS` to the allowed origins, separated by commas. |
| Gateway | Use the [Caddyfile](Caddyfile) with Caddy 2.10. Set `STAGING_USER`, `STAGING_PASSWORD_HASH`, `STAGING_BACKEND`, and `STAGING_PROXY_TOKEN`. Use the same randomly generated proxy token in the gateway and the API wrapper. |
| Database | Use a disposable corpus copy or build a fresh database. Give the API role read access to benchmark passages, write access to the document and tag tables, and SELECT and INSERT on search history. Initialize an empty application data set before admitting operators. |

Keep credentials and the gateway CA private to the environment. Generate the password hash with Caddy's password-hashing command and store the gateway settings in an owner-readable environment file outside the repository. Pass the path to the environment file in `docker run --env-file`. Preserve the gateway's `/data` volume across restarts. Distribute the generated root certificate only to staging clients that need it. [Caddy Basic authentication](https://caddyserver.com/docs/caddyfile/directives/basic_auth)

## Deploy and roll back

1. Record the Git commit, database snapshot and schema, model revisions, BM25 binary SHA-256, and gateway image digest. Build the frontend with `VITE_API_BASE=/api` into a new release directory. Under the gateway volume, point `current` to that directory with a relative symlink.
2. Start the gateway with `deployment/staging/Caddyfile` mounted at `/etc/caddy/Caddyfile`, the release directories mounted at `/srv`, and port 8443 published only on `127.0.0.1`. Point `STAGING_BACKEND` at the backend's private address. For Docker Desktop and a loopback backend on the host, use `host.docker.internal:<port>`.
3. Start the protected API with an isolated database URL and the gateway token. Check that unauthenticated document writes and history reads yield 401, direct API calls without the proxy token yield 403, and authenticated requests complete document CRUD, search, and history.
4. To roll back the application, rebuild the prior commit's frontend with the gateway's API base, switch `current` back, and restart the API with the prior checkout. Keep the same staging gateway and preserve the database while verifying the previous release. If a release changes the database schema incompatibly, restore that release's snapshot into a separate volume and verify it before switching the database connection.

The gateway applies the same 10-second upstream response-header timeout to both standard and batched reranking. For an open-loop comparison, use the same caller timeout and arrival schedule on both routes. A timeout may leave API work in progress, so wait for work to finish before measuring another route. [Caddy proxy timeouts](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy#the-http-transport)

## Trace rerank requests

The gateway generates a UUID for each proxied rerank request, including requests that time out at the gateway. Find it in the `X-Request-ID` response header and the gateway's JSON access log as `uuid`. The gateway replaces any client-supplied value. Search the API's `rerank_trace` JSON log entries for the same `request_id`.

The API records receipt, response completion, inference start and end, and the standard lock-wait or batched enqueue event. The scheduler adds its batch ID and records its own wait timeout. Compare the gateway completion timestamp with those events to see whether the request was still waiting or running after the gateway returned. Compare clocks before doing this across hosts. Inference end marks the end of the model call, including an exception. Use the API response status to determine the result.

The staging trace carries IDs, timestamps, queue metadata and status. Request and response headers are omitted from the rerank access logs, and the trace does not record search text or passages. When rolling back to a revision that predates the trace module, run the API wrapper from the earlier staging release.

The staging wrapper observes upstream connection loss after reading the rerank request body. Standard requests stop waiting for the inference lock. Batched jobs are excluded from further work before the model call, including jobs in a batch waiting for the shared lock. A model call already in progress can finish and return scores to connected requests in the batch. The wrapper logs `api_disconnected`, and a discarded request is recorded as an API-side 499. The client still receives the gateway's status, such as 504 when the gateway times out. Already-disconnected rerank results are discarded before recording search history.

## Sample a staging load test

Set `STAGING_RERANK_TELEMETRY` to a writable, private JSONL path before starting the protected API wrapper. The wrapper appends one sample per second with wall and Python performance-counter timestamps, the API process's cumulative CPU time and peak RSS, event-loop delay, and the waiting batch deque's depth, pair count and oldest item age. The inference lock flag describes the shared lock at the sample instant. A job already removed from the deque for batch formation or inference is outside the depth counter.

On macOS the sampler also records Mach host CPU ticks, compressed pages and cumulative VM counters. Derive utilization and paging activity from differences between successive samples. Use a separate file per run. Match the same host monotonic clock to the load window and the existing request UUIDs to gateway and model traces. The telemetry file contains aggregate counters and timing.

## Replay staging acceptance traffic

Run the [live replay driver](benchmark_rerank.py) from a clean checkout. It replays the [fixed 120-query pool](rerank-query-pool.json) against the full MS MARCO database and the C++ lexical index. It validates the manifest digest, the returned Top-10, and each successful response's gateway request ID. The driver needs Python, the gateway's trusted CA certificate, and an owner-readable netrc file with a `machine` entry for the hostname. The output carries request IDs, rankings, and timings.

Configure the target with the full corpus and index, four CPU inference threads, one API worker, pair batching, and history writes enabled to match the reference below. Set `STAGING_RERANK_TELEMETRY` on the API to capture resource and queue observations. Record the API revision, machine, and gateway configuration alongside the output.

```bash
python deployment/staging/benchmark_rerank.py \
  --gateway "$STAGING_GATEWAY" --ca "$STAGING_CA" --netrc "$STAGING_NETRC" \
  --seconds 1800 --warmup 60 --rate 2.5 --seed 141025 --output steady.json

python deployment/staging/benchmark_rerank.py \
  --gateway "$STAGING_GATEWAY" --ca "$STAGING_CA" --netrc "$STAGING_NETRC" \
  --burst 46 --route batched --output burst-batched.json
```

Repeat the burst with `--route standard` as needed. Correlate the driver IDs with the gateway's `uuid`, `status`, and `gateway_error` and the API's rerank events. With a driver on the API host, use the measurement window's Python monotonic timestamp to join the telemetry. Across hosts, align their wall clocks and verify the join using request IDs.

The steady acceptance thresholds are scheduled-to-response P95 at most 3 seconds, failures at most 1%, and successful completions during the window for at least 95% of scheduled arrivals. Keep failures and results that finish after the window in the raw measurement. The burst exercises timeout and disconnect behavior.

## Local verification evidence

- Two matched 46-request bursts at two CPU threads showed 27 requests starting inference after a gateway 504 before disconnect cancellation, and zero after the change. Totals of 22 and 21 gateway timeouts by route were unchanged. Already-running model calls completed.
- An alternating before-and-after replay at four threads and 2.5 offered RPS completed 1,792 requests across four 180-second windows. Before/after P95 was 2.48/3.53 s on one schedule and 2.98/1.72 s on the other. Top-10 document sets matched for all 896 paired successes. Separate gateway controls identified connection refusal and upstream header timeout.
- Two subsequent 600-second windows with cancellation and telemetry enabled completed 3,005 requests. Both met the local criteria with P95 of 2.13 and 1.62 seconds. Sampled queue depth peaked at 8 and 7. In the preceding 10 seconds, arrivals exceeded completions by 11 and 8. These measurements used a Mac with four CPU threads and one API worker.
