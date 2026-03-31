#!/bin/bash
GPU_ID="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for i in $(seq -w 1 20); do
  task run "job_${i}" "$SCRIPT_DIR/job_${i}.sh $GPU_ID"
done

wait
