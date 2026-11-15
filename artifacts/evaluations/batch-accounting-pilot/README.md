# Historical frozen-candidate batch accounting

This archive covers the 60-second, nominal 10 RPS fixed-candidate pilot. Each route completed 574 measured requests. The pair policy reported 353 complete batches and the token32 policy reported 330 complete batches. Inference served 50,511 pairs under each policy, at 1,093.678 and 1,071.894 pairs per inference second, respectively. The experiment measures scheduler and model execution over fixed candidates.

The two input files retain the successful-response indicator, candidate count and batch profile fields used by the analyzer. See [sources.json](sources.json) for the original and archived SHA-256 digests, fixed-candidate dataset digest, and load settings. The archived fields reproduce the analyzer output for both original reports.

From the repository root, with NumPy installed in the Python environment, run

```bash
python backend/scripts/analyze_batch_accounting.py \
  artifacts/evaluations/batch-accounting-pilot/pair_10rps_pilot.json \
  artifacts/evaluations/batch-accounting-pilot/token32_10rps_pilot.json \
  | diff -u artifacts/evaluations/batch-accounting-pilot/expected.txt -
```
