#!/bin/sh
# Whole plans with fix `order` (probe_allocation_fix.py --full) from saved chains of other works.
# Each seed is the original allocation request's seed - 1, so the original run is the `now` control;
# konbini-base is also rerun with `now` to check that the control is reproduced.
# Usage: scripts/story/run_allocation_order.sh WORKS_DIR
set -u
works=$1
mkdir -p "$works/alloc-plan"
for row in konbini-base:base:now konbini-base:base:order konbini-a3l:a3l:order hidamari-base:base:order \
    hidamari-a3l:a3l:order nagi-base:base:order sekai-base-s4:base:order sekai-a3l:a3l:order \
    sajutsu-base-s2:base:order sajutsu-a3l-s2:a3l:order sajutsu-a3l-s4:a3l:order; do
  chain=${row%%:*}; rest=${row#*:}; arm=${rest%%:*}; fix=${rest#*:}
  work=${chain%%-*}
  seed=$(.venv/Scripts/python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['request']['seed'] - 1)" \
    "$works/$chain/requests/chapter-allocation-1.json")
  out="$works/alloc-plan/$chain-$fix"
  [ -f "$out/plot.json" ] && continue
  echo "$(date +%T) $chain $fix $seed"
  .venv/Scripts/python scripts/story/probe_allocation_fix.py --input "$works/inputs/$work.json" --full \
    --chain-from "$works/$chain" --arm "$arm" --fix "$fix" --seed "$seed" --output-dir "$out" > "$out.log" 2>&1
done
