#!/bin/sh
# probe_allocation_fix.py over saved 鉄と血 chains: every fix, the first seed for every chain first.
# The first seed is the original run's seed + 1, so `now` repeats the original allocation request.
# Usage: scripts/story/run_allocation_fix.sh WORKS_DIR
set -u
works=$1
for round in 0 10 20; do
  for row in failed-tetsu-a2-s3:a2:4 failed-tetsu-a3i:a3i:2 failed-tetsu-a3l-s4:a3l:5 tetsu-base-s3:base:4 tetsu-a3l:a3l:2; do
    chain=${row%%:*}; rest=${row#*:}; arm=${rest%%:*}; seed=$(( ${rest#*:} + round ))
    case $chain in failed-*) ;; *) [ "$round" = 0 ] || continue ;; esac
    for fix in now pick code order; do
      name=${chain#failed-}
      out="$works/alloc/${name#tetsu-}-$fix-s$seed"
      [ -f "$out/chapter-allocation.json" ] || [ -f "$out.done" ] && continue
      echo "$(date +%T) $chain $fix $seed"
      .venv/Scripts/python scripts/story/probe_allocation_fix.py --input "$works/inputs/tetsu.json" \
        --chain-from "$works/$chain" --arm "$arm" --fix "$fix" --seed "$seed" --output-dir "$out" > "$out.log" 2>&1
      touch "$out.done"
    done
  done
done
