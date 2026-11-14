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
