#!/usr/bin/env bash
set -euo pipefail

pids=()
for subject in {1..10}; do
  if [[ "$subject" -eq 8 ]]; then
    echo "Skipping subject 8" >&2
    continue
  fi
  echo "Starting subject $subject" >&2
  uv run preprocess/process_eeg_whiten.py --subject "$subject" &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

exit "$fail"
