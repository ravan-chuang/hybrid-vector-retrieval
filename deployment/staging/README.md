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

Set `STAGING_RERANK_TELEMETRY` to a writable, private JSONL path before starting the protected API wrapper. The wrapper appends one sample per second with wall and monotonic timestamps, the API process's cumulative CPU time and peak RSS, event-loop delay, and the waiting batch deque's depth, pair count and oldest item age. The inference lock flag describes the shared lock at the sample instant. A job already removed from the deque for batch formation or inference is outside the depth counter.

On macOS the sampler also records Mach host CPU ticks, compressed pages and cumulative VM counters. Derive utilization and paging activity from differences between successive samples. Use a separate file per run. Match the same host monotonic clock to the load window and the existing request UUIDs to gateway and model traces. The telemetry file contains aggregate counters and timing.
