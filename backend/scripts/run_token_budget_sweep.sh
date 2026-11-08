#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="$PWD/.venv/bin/python"
BENCHMARK="$PWD/backend/scripts/benchmark_frozen_http.py"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"

OUTPUT_DIR="$PWD/reports/performance/cross_encoder/token_budget_sweep/$RUN_ID"

mkdir -p "$OUTPUT_DIR"

if [ ! -f .env ]; then
    echo "ERROR: .env not found."
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python virtual environment not found."
    exit 1
fi

set -a
source .env
set +a

export ENABLE_FROZEN_BENCHMARK=1

export CROSS_ENCODER_DEVICE=mps
export CROSS_ENCODER_BATCH_SIZE=32

export RERANK_BATCH_POLICY=token

export RERANK_MAX_BATCH_REQUESTS=4
export RERANK_MAX_BATCH_PAIRS=800
export RERANK_MAX_BATCH_WAIT_MS=5
export RERANK_BATCH_TIMEOUT_SECONDS=60

PORT=8000
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

trap cleanup EXIT INT TERM

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN \
    2>/dev/null | grep -q LISTEN; then
    echo "ERROR: Port $PORT is occupied."
    exit 1
fi

run_configuration() {
    local round="$1"
    local config="$2"
    local budget="$3"

    export RERANK_MAX_BATCH_TOKENS="$budget"

    local prefix="$OUTPUT_DIR/round${round}_${config}"
    local log_file="${prefix}.log"
    local output_file="${prefix}.json"

    echo
    echo "========================================"
    echo "ROUND=$round CONFIG=$config"
    echo "TOKEN_BUDGET=$budget"
    echo "========================================"

    "$PYTHON" -m uvicorn backend.api.main:app \
        --host 127.0.0.1 \
        --port "$PORT" \
        --workers 1 \
        > "$log_file" 2>&1 &

    SERVER_PID=$!

    local ready=0

    for i in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: Server exited unexpectedly."
            tail -n 80 "$log_file"
            exit 1
        fi

        if curl -fsS \
            "http://127.0.0.1:$PORT/docs" \
            >/dev/null 2>&1; then
            ready=1
            break
        fi

        sleep 2
    done

    if [ "$ready" -ne 1 ]; then
        echo "ERROR: Server startup timeout."
        tail -n 80 "$log_file"
        exit 1
    fi

    "$PYTHON" "$BENCHMARK" \
        --requests 100 \
        --repetitions 1 \
        --concurrency 4 8 \
        --pipelines batched \
        --warmup-requests 16 \
        --output "$output_file"

    cleanup

    sleep 5
}

echo "Experiment output: $OUTPUT_DIR"

# Rotating order reduces systematic configuration-order bias.
run_configuration 1 T12 12288
run_configuration 1 T16 16384
run_configuration 1 T20 20480
run_configuration 1 T24 24576
run_configuration 1 T28 28672
run_configuration 1 T32 32768

run_configuration 2 T20 20480
run_configuration 2 T24 24576
run_configuration 2 T28 28672
run_configuration 2 T32 32768
run_configuration 2 T12 12288
run_configuration 2 T16 16384

run_configuration 3 T28 28672
run_configuration 3 T32 32768
run_configuration 3 T12 12288
run_configuration 3 T16 16384
run_configuration 3 T20 20480
run_configuration 3 T24 24576

echo
echo "========================================"
echo "PHASE 7D COMPLETED"
echo "========================================"
echo "Results: $OUTPUT_DIR"
