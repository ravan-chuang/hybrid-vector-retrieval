"""Replay live benchmark-corpus queries through a protected staging gateway."""

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
import json
import math
import netrc
from pathlib import Path
import ssl
import sys
from tempfile import NamedTemporaryFile
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import (
    HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend/scripts"))
import benchmark_open_loop_duration as benchmark

ROUTES = {
    "standard": "/api/search/rerank",
    "batched": "/api/search/rerank/batched",
}
PROFILE_KEYS = (
    "batch_id", "batch_request_count", "batch_pair_count", "batch_queue_wait_ms",
    "batch_backlog_wait_ms", "batch_formation_wait_ms", "batch_inference_ms",
    "token_counting_ms", "batch_token_count", "batch_token_utilization",
    "lexical_ms", "vector_ms", "embedding_ms", "rerank_ms", "total_ms",
)
LIMITS = {"p95_ms": 3000, "failure_rate": 0.01, "window_success_share": 0.95}


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, message, headers, newurl):
        return None


def read_pool():
    pool = json.loads(Path(__file__).with_name("rerank-query-pool.json").read_text())
    raw = (ROOT / "artifacts/msmarco_full/dev_small_manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != pool["manifest_sha256"]:
        raise ValueError("The MS MARCO manifest does not match the query pool.")
    texts = {row["query_id"]: row["query"] for row in json.loads(raw)["queries"]}
    return pool, [
        {"query_id": query_id, "body": json.dumps({
            "query": texts[query_id], "corpus": "benchmark", "lexical_backend": "cpp_bm25",
            "stopword_filter": True, "top_k": 10, "candidate_k": 50, "ef_search": 160,
        }).encode()}
        for query_id in pool["query_ids"]
    ]


def request_sender(opener, authorization, route):
    def send(base_url, endpoint, item, request_id, scheduled, start, timeout):
        delay = scheduled - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        sent = time.perf_counter()
        status = trace_id = result = error = None
        request = Request(base_url + endpoint, data=item["body"], method="POST",
                          headers={"Content-Type": "application/json",
                                   "Authorization": authorization})
        try:
            with opener.open(request, timeout=timeout) as response:
                status = response.status
                trace_id = response.headers.get("X-Request-ID")
                result = json.load(response)
            ids = [row["document_id"] for row in result["results"]]
            if not trace_id or len(ids) != 10 or len(set(ids)) != 10:
                raise ValueError("Missing request ID or invalid Top-10 response")
            if result["effective_ef_search"] != 160 or (
                route == "batched" and not result.get("batch_id")
            ):
                raise ValueError("Unexpected search setting or missing batch ID")
        except HTTPError as exc:
            with exc:
                status = exc.code
                trace_id = exc.headers.get("X-Request-ID")
            error = f"HTTP {status}"
        except (OSError, HTTPException, ValueError, TypeError, KeyError) as exc:
            error = type(exc).__name__
        ended = time.perf_counter()
        success = status == 200 and error is None
        return {
            "request_id": request_id, "trace_id": trace_id, "query_id": item["query_id"],
            "status_code": status, "error": error, "success": success,
            "top10": ids if success else None,
            "candidate_count": result.get("candidate_count") if success else None,
            "profile": {key: result.get(key) for key in PROFILE_KEYS} if success else None,
            "scheduled_offset_ms": 1000 * (scheduled - start),
            "dispatch_offset_ms": 1000 * (sent - start),
            "completed_offset_ms": 1000 * (ended - start),
            "dispatch_lag_ms": 1000 * (sent - scheduled),
            "http_latency_ms": 1000 * (ended - sent),
            "scheduled_latency_ms": 1000 * (ended - scheduled),
        }

    return send


def steady(args, payloads, send):
    shared = dict(base_url=args.gateway.rstrip("/"), endpoint=ROUTES[args.route],
                  payloads=payloads, arrival_rate=args.rate, workers=args.workers,
                  timeout=args.timeout, request_sender=send)
    warmup = (
        benchmark.run_phase(phase="warmup", duration=args.warmup, seed=args.seed + 10000, **shared)
        if args.warmup else None
    )
    measured = benchmark.run_phase(phase="measurement", duration=args.seconds, seed=args.seed, **shared)
    summary = benchmark.summarize_measurement(measured, args.rate)
    records = measured["records"]
    summary["capacity"] = benchmark.calculate_capacity_metrics(records, args.seconds)
    if records:
        summary["window_success_share"] = (
            summary["capacity"]["window_goodput_rps"] * args.seconds / len(records)
        )
        summary["failure_rate"] = summary["failure_count"] / len(records)
        summary["meets_targets"] = (
            summary["scheduled_latency_ms"]["p95"] <= LIMITS["p95_ms"]
            and summary["failure_rate"] <= LIMITS["failure_rate"]
            and summary["window_success_share"] >= LIMITS["window_success_share"]
        )
    else:
        summary["meets_targets"] = False
    return {"mode": "steady", "summary": summary, "warmup": warmup, "measurement": measured}


def burst(args, payloads, send):
    start = time.perf_counter() + 0.5
    with ThreadPoolExecutor(max_workers=args.burst) as executor:
        records = list(executor.map(
            lambda pair: send(args.gateway.rstrip("/"), ROUTES[args.route], pair[1],
                              pair[0], start, start, args.timeout),
            enumerate(payloads[:args.burst]),
        ))
    return {
        "mode": "burst", "records": records,
        "summary": {
            "scheduled_request_count": len(records),
            "success_count": sum(row["success"] for row in records),
            "failure_count": sum(not row["success"] for row in records),
            "status_counts": dict(Counter(str(row["status_code"]) if row["status_code"] is not None
                                          else "transport_error" for row in records)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", required=True, help="HTTPS origin of the staging gateway")
    parser.add_argument("--ca", required=True, type=Path)
    parser.add_argument("--netrc", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--route", choices=ROUTES, default="batched")
    parser.add_argument("--rate", type=float, default=2.5)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--seed", type=int, default=141025)
    parser.add_argument("--burst", type=int, metavar="REQUESTS")
    args = parser.parse_args()

    origin = urlparse(args.gateway)
    if (origin.scheme != "https" or not origin.hostname or origin.username is not None
            or origin.path.strip("/") or origin.params or origin.query or origin.fragment):
        parser.error("--gateway must be an HTTPS origin")
    if (not all(math.isfinite(t) and t > 0 for t in (args.rate, args.timeout))
            or min(args.seconds, args.workers) < 1 or args.warmup < 0):
        parser.error("Invalid rate, duration, worker count, or timeout")
    pool, payloads = read_pool()
    if args.burst is not None and not 1 <= args.burst <= len(payloads):
        parser.error(f"--burst must be between 1 and {len(payloads)}")
    auth = netrc.netrc(args.netrc).hosts.get(origin.hostname)
    if not auth or not auth[0] or not auth[2]:
        parser.error("The netrc file needs a login and password for the gateway hostname")
    authorization = "Basic " + base64.b64encode(f"{auth[0]}:{auth[2]}".encode()).decode()
    opener = build_opener(ProxyHandler({}), HTTPSHandler(context=ssl.create_default_context(cafile=str(args.ca))),
                          NoRedirects())
    send = request_sender(opener, authorization, args.route)
    started = datetime.now(timezone.utc).isoformat()
    result = burst(args, payloads, send) if args.burst is not None else steady(args, payloads, send)
    result["limits"] = LIMITS
    result["protocol"] = {
        "started_at_utc": started, "route": args.route, "query_pool_size": len(payloads),
        "manifest_sha256": pool["manifest_sha256"], "timeout_seconds": args.timeout,
    }
    if args.burst is not None:
        result["protocol"]["burst_requests"] = args.burst
    else:
        result["protocol"].update(seed=args.seed, rate_rps=args.rate, warmup_seconds=args.warmup,
                                  measurement_seconds=args.seconds, max_workers=args.workers)
    with NamedTemporaryFile(mode="w", dir=args.output.parent, prefix=".rerank-", delete=False) as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    Path(file.name).replace(args.output)
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
