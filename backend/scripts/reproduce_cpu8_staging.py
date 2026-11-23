"""Recompute CPU8 staging acceptance summaries from archived request observations."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from statistics import quantiles

from backend.scripts.reproduce_candidate_contribution import verify


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "artifacts/evaluations/cpu8-full-union"


def load_observations(protocol):
    pool = ROOT / "deployment/staging/rerank-query-pool.json"
    if hashlib.sha256(pool.read_bytes()).hexdigest() != protocol["query_pool_sha256"]:
        raise ValueError("Staging query pool digest mismatch")
    packed = (ARCHIVE / "observations.csv.gz").read_bytes()
    if hashlib.sha256(packed).hexdigest() != protocol["observations_sha256"]:
        raise ValueError("Observation archive digest mismatch")
    rows = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(io.StringIO(gzip.decompress(packed).decode())):
        trial, phase = row["trial"], row["phase"]
        status = row["status"]
        success = int(row["success"])
        scheduled, completed = float(row["scheduled_ms"]), float(row["completed_ms"])
        if (phase not in ("warmup", "measurement") or success not in (0, 1)
                or success and status != "200" or not 0 <= scheduled <= completed):
            raise ValueError(f"Invalid request observation in {trial}")
        rows[trial][phase].append((status, success, scheduled, completed))
    return rows


def summarize(protocol):
    trials = load_observations(protocol)
    count = 0
    result = {}
    for trial, phases in trials.items():
        measured = phases["measurement"]
        warmup = phases.get("warmup", [])
        if len(measured) < 2:
            raise ValueError(f"Insufficient observations in {trial}")
        latencies = [completed - scheduled for _, _, scheduled, completed in measured]
        item = {
            "requests": len(measured),
            "successes": sum(success for _, success, _, _ in measured),
            "status_counts": dict(Counter(status for status, _, _, _ in measured)),
            "scheduled_p95_ms": round(quantiles(latencies, n=100, method="inclusive")[94], 3),
            "warmup_successes": sum(success for _, success, _, _ in warmup),
        }
        if trial.startswith("steady-"):
            cutoff = 1000 * protocol["steady"]["seconds"]
            item["window_successes"] = sum(success for _, success, _, finished in measured
                                           if finished < cutoff)
        else:
            if warmup:
                raise ValueError(f"Unexpected warmup in burst trial {trial}")
            item.update(
                successes_in_first_10s=sum(success for _, success, _, finished in measured
                                           if finished <= 10_000),
                last_response_ms=round(max(finished for _, _, _, finished in measured), 3),
            )
        result[trial] = item
        count += len(measured) + len(warmup)
    return {"request_count": count, "trials": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the archived request metrics")
    args = parser.parse_args()
    protocol = json.loads((ARCHIVE / "protocol.json").read_text())
    result = summarize(protocol)
    if args.check:
        reference = json.loads((ARCHIVE / "expected.json").read_text())
        verify(result, {key: reference[key] for key in result}, path="CPU8 staging")
        audit = reference["trace_audit"]
        histories = {name: row["successes"] + row["warmup_successes"] for name, row in result["trials"].items()}
        verify(histories, {name: row["history_rows"] for name, row in audit["by_trial"].items()},
               path="audited history counts")
        verify(sum(histories.values()), audit["history_total"], path="audited history total")
        print(f"Verified {result['request_count']:,} requests across {len(result['trials'])} staging trials.")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
