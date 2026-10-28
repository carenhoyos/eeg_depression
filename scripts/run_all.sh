#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
CFG=${1:-config/config.yaml}
for s in 01_inspect 02_standardize_qc 03_features 04_train_eval 05_interpret; do
  echo "##### $s"; $PY scripts/$s.py --config "$CFG"
done
