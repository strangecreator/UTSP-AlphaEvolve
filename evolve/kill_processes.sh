#!/usr/bin/env bash

set -euo pipefail


# List of regex patterns (match against full command line)
PATTERNS=(
  '^/workspace/dataspace/alpha_evolve'
  '^python3 launch\.py'
)

pids=()

for pattern in "${PATTERNS[@]}"; do
    # pgrep -f -> match full command line
    while IFS= read -r pid; do
        [ -n "$pid" ] && pids+=("$pid")
    done < <(pgrep -f -- "$pattern" || true)
done

if [ "${#pids[@]}" -eq 0 ]; then
    echo "No matching processes."
    exit 0
fi

# Deduplicate PIDs
mapfile -t unique_pids < <(printf '%s\n' "${pids[@]}" | sort -u)

echo "Killing with SIGKILL: ${unique_pids[*]}"
kill -9 "${unique_pids[@]}"