#!/usr/bin/env bash
# E7, one command:  bash experiments/E7_gpu/run.sh
# Checks the machine, runs the smoke stage, runs the full experiment, writes the report.
# Rerunning the same command resumes. Use tmux or screen, so a dropped connection does not stop it.
# Options: --mini (CPU validation only), --out DIR, and any e7.py option such as --per-gpu 2.
set -euo pipefail

die() { echo; echo "ERROR: $*" >&2; exit 1; }

# Results live only on this machine, so put them on screen too: an instance can be destroyed.
show_results() {
  local dir="$1"
  tar czf "$dir/e7_results.tar.gz" -C "$dir" report.txt results_compact.csv selection.json jobs 2>/dev/null || true
  echo
  echo "=================== COPY EVERYTHING FROM HERE ==================="
  cat "$dir/report.txt"
  echo
  echo "---------------------- results_compact.csv ----------------------"
  cat "$dir/results_compact.csv"
  echo "=================== COPY EVERYTHING TO HERE ====================="
  echo
  echo "!!! NOTHING IS UPLOADED ANYWHERE. These results exist only on this machine. !!!"
  echo "!!! Do NOT destroy the instance until you have copied the text above.       !!!"
  echo "    Everything, including per-job files: $dir/e7_results.tar.gz"
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
E7="$HERE/e7.py"
ANALYZE="$HERE/analyze.py"
OUT="$REPO/runs/e7"
MINI=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --mini) MINI=1; shift ;;
    --out) [ $# -ge 2 ] || die "--out needs a directory"; OUT="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python3 >/dev/null 2>&1; then PY=python3
  elif command -v python >/dev/null 2>&1; then PY=python
  else die "no python found. Install Python 3.9 or newer."; fi
fi
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 9))' \
  || die "Python 3.9 or newer is needed; $PY is $("$PY" --version 2>&1)."
echo "python: $("$PY" --version 2>&1) at $(command -v "$PY")"

if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  echo "torch is missing; installing it with pip."
  "$PY" -m pip install torch || die "could not install torch. Install it, then rerun."
fi
"$PY" -c 'import sys, torch
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
sys.exit(v < (2, 3))' \
  || die "torch 2.3 or newer is needed. Upgrade with: $PY -m pip install --upgrade torch"
if ! "$PY" -c 'import numpy, scipy' >/dev/null 2>&1; then
  echo "numpy or scipy is missing; installing."
  "$PY" -m pip install -r "$HERE/requirements.txt" \
    || die "could not install numpy and scipy. Install them, then rerun."
fi
"$PY" -c 'import numpy, scipy, torch; torch.from_numpy(numpy.zeros(1))' \
  || die "numpy and torch do not work together here. Try: $PY -m pip install --upgrade numpy torch"

if [ "$MINI" = 1 ]; then
  MINI_OUT="${OUT}_mini"
  "$PY" "$E7" --mini --out "$MINI_OUT" ${EXTRA[@]+"${EXTRA[@]}"} || die "the mini run failed. Logs: $MINI_OUT/logs"
  "$PY" "$ANALYZE" --out "$MINI_OUT" || die "analysis of the mini run failed."
  show_results "$MINI_OUT"
  echo; echo "Mini run finished."
  exit 0
fi

"$PY" "$E7" --preflight || die "the machine check failed; the message above says why."

SMOKE_OUT="${OUT}_smoke"
echo; echo "=== Smoke stage ==="
"$PY" "$E7" --smoke --out "$SMOKE_OUT" ${EXTRA[@]+"${EXTRA[@]}"} \
  || die "the smoke stage failed, so the full run will not start. Logs: $SMOKE_OUT/logs"
"$PY" "$ANALYZE" --out "$SMOKE_OUT" \
  || die "analysis failed in the smoke stage, so the full run will not start."

echo; echo "=== Full experiment ==="
STATUS=0
"$PY" "$E7" --out "$OUT" ${EXTRA[@]+"${EXTRA[@]}"} || STATUS=$?

echo; echo "=== Analysis ==="
"$PY" "$ANALYZE" --out "$OUT" || die "analysis failed. Job results are safe in $OUT/jobs"

show_results "$OUT"
echo
if [ "$STATUS" != 0 ]; then
  echo "ERROR: some jobs did not finish (exit $STATUS). The report above is partial." >&2
  echo "Run the same command again to resume and retry them. Logs: $OUT/logs" >&2
else
  echo "E7 finished."
fi
exit "$STATUS"
