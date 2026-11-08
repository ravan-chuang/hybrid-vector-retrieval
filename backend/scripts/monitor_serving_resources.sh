#!/bin/bash

set -u

OUTPUT="reports/performance/cross_encoder/diagnostics/resources.csv"

mkdir -p "$(dirname "$OUTPUT")"

echo "timestamp,pid,cpu_percent,rss_kb" > "$OUTPUT"

PID=$(pgrep -f 'uvicorn backend.api.main:app' | head -1)

if [ -z "$PID" ]; then
    echo "ERROR: Uvicorn process not found."
    exit 1
fi

echo "Monitoring PID=$PID"
echo "Output=$OUTPUT"

while kill -0 "$PID" 2>/dev/null; do
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    ps -p "$PID" -o pid=,%cpu=,rss= \
      | awk -v ts="$TIMESTAMP" \
        'NF == 3 {print ts "," $1 "," $2 "," $3}' \
      >> "$OUTPUT"

    sleep 1
done
