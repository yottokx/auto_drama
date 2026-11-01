#!/bin/sh
# Scripts of one plan under each arm of probe_script_drama.py (no staging), one after another.
# A run that stops at the session deadline is resumed (up to three sessions).
# Writes WORKS_DIR/PLAN_RUN-script.done when every arm has been tried.
# Usage: scripts/story/run_script_drama.sh WORKS_DIR WORK PLAN_RUN PLAN_ARM ARM...
set -u
works=$1; work=$2; plan=$3; plan_arm=$4
shift 4
for arm in "$@"; do
  out="$works/$plan-script-$arm"
  for session in 1 2 3; do
    grep -q '"status": "script_complete"' "$out.log" 2>/dev/null && break
    echo "$(date +%T) $plan $arm session $session"
    .venv/Scripts/python scripts/story/probe_script_drama.py --input "$works/inputs/$work.json" \
      --plan-from "$works/$plan" --plan-arm "$plan_arm" --arm "$arm" --output-dir "$out" \
      --skip-staging >> "$out.log" 2>&1
  done
done
date +%T > "$works/$plan-script.done"
