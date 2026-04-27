#!/bin/bash
set -e

PRED_DIR="./predictions/cifs"
OUT_CSV="./evaluation/test_eval.csv"
LOG_FILE="./evaluation/test_eval.log"

# Run CCDC COMPACK to evaluate crystal similarity
python ./oxtal/metrics/crystal_similarity.py \
"$PRED_DIR" \
--truth-map ./evaluation/truth_map.csv \
--packing-size 1 15 \
--out-csv "$OUT_CSV" \
--workers 6 2>&1 | tee "$LOG_FILE"

# Extract metrics from the COMPACK output
python ./evaluation/get_metrics.py \
"$OUT_CSV" \
--txt-dir ./data/datasets \
> ./evaluation/metric_summary.txt
