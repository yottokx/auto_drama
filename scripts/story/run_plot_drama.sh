#!/bin/sh
# Every work and arm of probe_plot_drama.py, one after another; a finished run is skipped.
# A work without a saved cast plan gets one in its base run, which a and b then reuse.
# Usage: scripts/story/run_plot_drama.sh WORKS_DIR [WORK ...]
set -u
works=$1
shift
names=${*:-$(ls "$works/inputs" | sed 's/\.json$//')}
for work in $names; do
  for arm in base a b; do
    out="$works/$work-$arm"
    [ -f "$out/plot.json" ] && continue
    extra=""
    if ! grep -q '"cast_plan"' "$works/inputs/$work.json" && [ "$arm" != base ]; then
      extra="--cast-from $works/$work-base"
    fi
    resume=""
    [ -d "$out" ] && resume="--resume"
    echo "$(date +%T) $work $arm"
    .venv/Scripts/python scripts/story/probe_plot_drama.py --input "$works/inputs/$work.json" \
      --arm "$arm" --output-dir "$out" $extra $resume > "$out.log" 2>&1 || echo "  failed: $out.log"
  done
done
