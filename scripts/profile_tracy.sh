#!/usr/bin/env bash
# One MPI rank / one local device. Run with the TT-Metal Python environment active.
set -euo pipefail

nbody_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
metal_root="${TT_METAL_HOME:-$(dirname "$nbody_root")}"
build_dir="${NBODY_BUILD_DIR:-$nbody_root/build-wh-tracy}"
python_bin="${PYTHON_BIN:-python}"
perf_bin="${TT_PERF_REPORT:-$(command -v tt-perf-report || true)}"
if [[ -z "$perf_bin" && -x "$HOME/.local/bin/tt-perf-report" ]]; then
    perf_bin="$HOME/.local/bin/tt-perf-report"
fi
[[ -x "$perf_bin" ]] || { echo 'Set TT_PERF_REPORT to the tt-perf-report executable.' >&2; exit 1; }
particles="${1:-1024}"
cycles="${2:-3}"
[[ "$particles" =~ ^[1-9][0-9]*$ && "$cycles" =~ ^[0-9]+$ ]] || {
    echo 'Usage: bash scripts/profile_tracy.sh [N>0] [Ncycle>=0]' >&2; exit 1;
}
output_dir="${3:-$metal_root/generated/nbody-tracy-$(date +%Y%m%d-%H%M%S)}"
[[ ! -e "$output_dir" ]] || { echo "Output already exists: $output_dir" >&2; exit 1; }
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
export TT_METAL_HOME="$metal_root"
export PYTHONPATH="$metal_root/tools:$metal_root${PYTHONPATH:+:$PYTHONPATH}"
export TRACY_NO_WEB_SERVER=1
cd "$metal_root"
"$python_bin" -m tracy -o "$output_dir" -r -p \
    --tracy-tools-folder "$build_dir/tools/profiler/bin" \
    "$build_dir/nbody_app" --N "$particles" --Ncycle "$cycles" \
    2>&1 | tee "$output_dir/run.log"

# A successful wrapper exit can still leave an empty or incomplete operation CSV.
report="$("$python_bin" - "$output_dir" "$cycles" <<'PY'
import sys
from pathlib import Path
import pandas as pd
reports = list(Path(sys.argv[1]).glob('reports/**/ops_perf_results*.csv'))
if len(reports) != 1:
    raise SystemExit(f'Expected one operation CSV, found {len(reports)}')
df = pd.read_csv(reports[0])
expected = int(sys.argv[2]) + 1
if len(df) != expected or df['GLOBAL CALL COUNT'].nunique() != expected:
    raise SystemExit(f'Expected {expected} distinct N-body operations, found {len(df)} rows')
if not df['OP CODE'].str.startswith('NBody').all():
    raise SystemExit('Unexpected operation names in N-body report')
durations = pd.to_numeric(df['DEVICE KERNEL DURATION [ns]'], errors='coerce')
if durations.isna().any() or (durations <= 0).any() or (durations >= 2**64 - 1).any():
    raise SystemExit('Invalid device durations; check device clock and profiler capture')
print(reports[0])
PY
)"
cd "$(dirname "$report")"
"$perf_bin" "$(basename "$report")" --no-advice --csv perf_report.csv
echo "Operation CSV: $report"
echo "Summary: $PWD/perf_report.csv"
