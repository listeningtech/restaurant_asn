#!/usr/bin/env bash
set -euo pipefail

# K=6 cIRM pilot experiment.
# Use this on a machine where the torch environment can see a GPU.

ROOT="${ROOT:-/home/rrame12/Desktop/Research/ASN/V7}"
DATA="${DATA:-$ROOT/k6_cirm_dataset}"
RUNS="${RUNS:-$ROOT/k6_cirm_runs}"

PY_GEN="${PY_GEN:-/home/rrame12/anaconda3/envs/audio/bin/python}"
PY_TORCH="${PY_TORCH:-/home/rrame12/anaconda3/envs/torch/bin/python}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_asn}"

N_TRAIN="${N_TRAIN:-500}"
N_VAL="${N_VAL:-50}"
N_TEST="${N_TEST:-50}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-6060}"

cd "$ROOT"
mkdir -p "$RUNS"

echo "Generating K=6 dataset at $DATA"
"$PY_GEN" generate_restaurant_dataset_v6_graph_required.py \
  --out "$DATA" \
  --n_tables 6 \
  --n_train "$N_TRAIN" --n_val "$N_VAL" --n_test "$N_TEST" --shard_size 25 \
  --mics_per_table 1 \
  --min_speakers_per_table 1 --max_speakers_per_table 3 \
  --hard_scene_prob 0.9 \
  --min_hard_tables 3 \
  --enable_local_mic_corruption \
  --mic_corrupt_prob 0.35 \
  --enable_local_target_masking \
  --target_mask_prob 0.25 \
  --seed "$SEED"

COMMON_TRAIN=(
  --data "$DATA"
  --batch "$BATCH"
  --epochs "$EPOCHS"
  --lr 3e-4
  --patience 8
  --crop_s 2.0
  --nfft 1024
  --hop 256
  --num_workers "$WORKERS"
  --base 32
  --depth 4
  --graph_dim 128
  --graph_heads 4
)

COMMON_EVAL=(
  --data "$DATA"
  --split test
  --batch "$BATCH"
  --num_workers "$WORKERS"
  --save_scenes 0
)

echo "Training K=6 local cIRM"
"$PY_TORCH" train_graph_cirm_v10.py \
  "${COMMON_TRAIN[@]}" \
  --out "$RUNS/local" \
  --disable_graph

echo "Training K=6 Neighbourhood shared network cIRM"
"$PY_TORCH" train_graph_cirm_v10.py \
  "${COMMON_TRAIN[@]}" \
  --out "$RUNS/neighbourhood"

echo "Training K=6 fully shared network cIRM"
"$PY_TORCH" train_graph_alltoall_v10.py \
  "${COMMON_TRAIN[@]}" \
  --out "$RUNS/full"

echo "Evaluating K=6 local cIRM"
"$PY_TORCH" eval_graph_cirm_v10.py \
  --run_dir "$RUNS/local" \
  --out "$RUNS/local/eval_test" \
  "${COMMON_EVAL[@]}"

echo "Evaluating K=6 Neighbourhood shared network cIRM"
"$PY_TORCH" eval_graph_cirm_v10.py \
  --run_dir "$RUNS/neighbourhood" \
  --out "$RUNS/neighbourhood/eval_test" \
  "${COMMON_EVAL[@]}"

echo "Evaluating K=6 fully shared network cIRM"
"$PY_TORCH" eval_graph_alltoall_v10.py \
  --run_dir "$RUNS/full" \
  --out "$RUNS/full/eval_test" \
  "${COMMON_EVAL[@]}"

echo "Done. Summaries:"
for model in local neighbourhood full; do
  echo "$model:"
  "$PY_TORCH" - <<PY
import json
path = "$RUNS/$model/eval_test/metrics_summary.json"
with open(path) as f:
    m = json.load(f)
print(f"  in={m['sisdr_in_db']:.3f} dB  out={m['sisdr_out_db']:.3f} dB  delta={m['delta_db']:+.3f} dB")
PY
done
