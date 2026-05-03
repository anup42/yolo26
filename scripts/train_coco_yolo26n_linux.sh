#!/usr/bin/env bash
# Create a Linux GPU TensorFlow environment, prepare COCO, train scratch YOLO26n,
# validate with COCOeval, and export/reload TFLite.
#
# Default profile is full COCO scratch training with conservative GPU-stable
# settings. Use YOLO26_COCO_PROFILE=small for a quick smoke run.

set -euo pipefail

export KERAS_BACKEND="${KERAS_BACKEND:-tensorflow}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_XLA_FLAGS="${TF_XLA_FLAGS:---tf_xla_auto_jit=0}"
if [[ "${YOLO26_COCO_CUDA_SYNC:-0}" == "1" ]]; then
  export CUDA_LAUNCH_BLOCKING=1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${YOLO26_COCO_VENV:-$ROOT_DIR/.venv-coco-train}"
DATA_DIR="${YOLO26_COCO_DATA:-$ROOT_DIR/datasets/coco}"
OUT_DIR="${YOLO26_COCO_OUT:-$ROOT_DIR/runs/train/yolo26n_tf_coco}"
PROFILE="${YOLO26_COCO_PROFILE:-full}"
TENSORFLOW_PACKAGE="${YOLO26_COCO_TENSORFLOW:-tensorflow[and-cuda]==2.15.1}"
NUMPY_PACKAGE="${YOLO26_COCO_NUMPY:-numpy>=1.23.5,<2.0}"
BATCH="${YOLO26_COCO_BATCH:-16}"
IMGSZ="${YOLO26_COCO_IMGSZ:-640}"
SUBSET="${YOLO26_COCO_SUBSET:-100}"
VAL_SUBSET="${YOLO26_COCO_VAL_SUBSET:-100}"
EPOCHS_SMALL="${YOLO26_COCO_EPOCHS_SMALL:-2}"
EPOCHS_FULL="${YOLO26_COCO_EPOCHS_FULL:-300}"
PROJECT="${YOLO26_COCO_PROJECT:-$OUT_DIR}"
NAME="${YOLO26_COCO_NAME:-scratch_${PROFILE}}"
CACHE_IMAGES="${YOLO26_COCO_CACHE_IMAGES:-auto}"
CACHE_RAM_GB="${YOLO26_COCO_CACHE_RAM_GB:-32}"
USE_TFRECORD="${YOLO26_COCO_USE_TFRECORD:-1}"
COMPILE_STEP="${YOLO26_COCO_COMPILE:-0}"
AMP="${YOLO26_COCO_AMP:-0}"
FAST_DATA="${YOLO26_COCO_FAST_DATA:-0}"
PREFETCH_DATA="${YOLO26_COCO_PREFETCH_DATA:-1}"
FAST_NMS="${YOLO26_COCO_FAST_NMS:-1}"
PROFILE_SPEED="${YOLO26_COCO_PROFILE_SPEED:-1}"
PROFILE_STAGE="${YOLO26_COCO_PROFILE_STAGE:-0}"
PROFILE_BATCHES="${YOLO26_COCO_PROFILE_BATCHES:-0}"
SYNC_PROFILE_STAGE="${YOLO26_COCO_SYNC_PROFILE_STAGE:-0}"
GPU_MONITOR="${YOLO26_COCO_GPU_MONITOR:-0}"
GPU_MONITOR_INTERVAL="${YOLO26_COCO_GPU_MONITOR_INTERVAL:-5}"
OPTIMIZER="${YOLO26_COCO_OPTIMIZER:-sgd}"
EMA_UPDATE_INTERVAL="${YOLO26_COCO_EMA_UPDATE_INTERVAL:-10}"

mkdir -p "$OUT_DIR"
LOG_FILE="${YOLO26_COCO_LOG:-$OUT_DIR/train_coco_yolo26n.log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
GPU_MONITOR_PID=""
GPU_STATS="$OUT_DIR/gpu_stats.csv"

cleanup() {
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

start_gpu_monitor() {
  if [[ "$GPU_MONITOR" != "1" ]]; then
    return
  fi
  require_cmd nvidia-smi
  echo "timestamp,gpu_util,memory_util,memory_used_mb,power_w,temp_c" > "$GPU_STATS"
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits >> "$GPU_STATS" 2>/dev/null || true
      sleep "$GPU_MONITOR_INTERVAL"
    done
  ) &
  GPU_MONITOR_PID="$!"
  echo "GPU monitor writing to $GPU_STATS every ${GPU_MONITOR_INTERVAL}s"
}

summarize_gpu_stats() {
  if [[ "$GPU_MONITOR" != "1" || ! -s "$GPU_STATS" ]]; then
    return
  fi
  python - "$GPU_STATS" <<'PY'
import csv
import sys

path = sys.argv[1]
rows = []
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try:
            rows.append({
                "gpu": float(row["gpu_util"]),
                "mem": float(row["memory_util"]),
                "used": float(row["memory_used_mb"]),
                "power": float(row["power_w"]),
                "temp": float(row["temp_c"]),
            })
        except Exception:
            pass
if not rows:
    print("GPU monitor summary: no samples")
    raise SystemExit
avg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
print(
    "GPU monitor summary: "
    f"samples={len(rows)}, avg_gpu_util={avg['gpu']:.1f}%, avg_mem_util={avg['mem']:.1f}%, "
    f"avg_mem_used_mb={avg['used']:.0f}, avg_power_w={avg['power']:.1f}, avg_temp_c={avg['temp']:.1f}"
)
if avg["gpu"] >= 70:
    print("GPU monitor interpretation: high GPU utilization; bottleneck is likely model compute.")
else:
    print("GPU monitor interpretation: low GPU utilization; inspect stage_profile.csv for CPU/TensorFlow sync bottlenecks.")
PY
}

download_file() {
  local url="$1"
  local dst="$2"
  if [[ -f "$dst" ]]; then
    echo "Using existing $dst"
    return
  fi
  mkdir -p "$(dirname "$dst")"
  echo "Downloading $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --retry 5 --retry-delay 5 -o "$dst" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$dst" "$url"
  else
    echo "Install curl or wget to download COCO." >&2
    exit 1
  fi
}

require_cmd "$PYTHON_BIN"
require_cmd unzip

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install "$NUMPY_PACKAGE" "$TENSORFLOW_PACKAGE"
python -m pip install -e "$ROOT_DIR[convert,dev]" "pycocotools>=2.0.7" "tqdm>=4.66" "$NUMPY_PACKAGE"
python -m pip install --force-reinstall "$NUMPY_PACKAGE"
python -m pip check

echo "Repository root: $ROOT_DIR"
if command -v git >/dev/null 2>&1 && git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Repository commit: $(git -C "$ROOT_DIR" rev-parse --short HEAD)"
fi

python - "$ROOT_DIR" <<'PY'
import importlib.metadata as metadata
import inspect
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
import yolo26_tf
from yolo26_tf.trainer import TrainConfig

pkg_path = Path(inspect.getfile(yolo26_tf)).resolve()
cfg = TrainConfig()
required = {
    "prefetch_data": True,
    "sync_profile_stage": False,
    "ema_update_interval": 1,
}
missing = [name for name in required if not hasattr(cfg, name)]
wrong = [f"{name}={getattr(cfg, name)!r}" for name, expected in required.items() if hasattr(cfg, name) and getattr(cfg, name) != expected]
print("YOLO26 package:", pkg_path)
print("YOLO26 package version:", getattr(yolo26_tf, "__version__", "unknown"), "metadata:", metadata.version("yolo26-tf"))
print(
    "YOLO26 speed defaults:",
    f"prefetch_data={getattr(cfg, 'prefetch_data', None)}",
    f"fast_data={cfg.fast_data}",
    f"compile_train_step={cfg.compile_train_step}",
    f"amp={cfg.amp}",
)
try:
    pkg_path.relative_to(root)
except ValueError as exc:
    raise SystemExit(f"yolo26_tf is not imported from this checkout: package={pkg_path}, root={root}") from exc
if missing or wrong:
    raise SystemExit(f"stale yolo26_tf package detected: missing={missing}, wrong={wrong}. Run git pull and rerun this script.")
PY

python - <<'PY'
from importlib.metadata import version
import os

npv = version("numpy")
print("NumPy:", npv)
if int(npv.split(".", 1)[0]) >= 2:
    raise SystemExit(
        f"NumPy {npv} is incompatible with TensorFlow 2.15.x. "
        "The runner pins numpy<2; remove the venv and rerun if this persists."
    )

import tensorflow as tf
tf.config.optimizer.set_jit(False)
print("TensorFlow:", tf.__version__)
print("TensorFlow XLA JIT:", tf.config.optimizer.get_jit(), "TF_XLA_FLAGS=", os.environ.get("TF_XLA_FLAGS", ""))
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if not gpus:
    raise SystemExit("No TensorFlow GPU detected. This training script is GPU-only.")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass
with tf.device("/GPU:0"):
    layer = tf.keras.layers.Conv2D(8, 3, padding="same")
    y = layer(tf.zeros([1, 64, 64, 3], tf.float32))
    _ = float(tf.reduce_sum(y).numpy())
print("GPU Conv2D sanity check: ok")
if len(gpus) > 1:
    print(f"Multi-GPU training will use MirroredStrategy over {len(gpus)} GPUs")
PY

mkdir -p "$DATA_DIR" "$OUT_DIR"
download_file "http://images.cocodataset.org/zips/train2017.zip" "$DATA_DIR/train2017.zip"
download_file "http://images.cocodataset.org/zips/val2017.zip" "$DATA_DIR/val2017.zip"
download_file "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$DATA_DIR/annotations_trainval2017.zip"

if [[ ! -d "$DATA_DIR/train2017" ]]; then
  echo "Extracting train2017 images"
  unzip -q "$DATA_DIR/train2017.zip" -d "$DATA_DIR"
fi
if [[ ! -d "$DATA_DIR/val2017" ]]; then
  echo "Extracting val2017 images"
  unzip -q "$DATA_DIR/val2017.zip" -d "$DATA_DIR"
fi
if [[ ! -f "$DATA_DIR/annotations/instances_train2017.json" || ! -f "$DATA_DIR/annotations/instances_val2017.json" ]]; then
  echo "Extracting COCO annotations"
  unzip -q "$DATA_DIR/annotations_trainval2017.zip" -d "$DATA_DIR"
fi

PREPARE_ARGS=(
  "$ROOT_DIR/scripts/prepare_coco_yolo.py"
  --coco-root "$DATA_DIR"
  --output-yaml "$DATA_DIR/coco_yolo26.yaml"
  --train-subset "$SUBSET"
  --val-subset "$VAL_SUBSET"
  --summary "$OUT_DIR/prepare_coco_summary.json"
)
if [[ "$USE_TFRECORD" == "0" ]]; then
  PREPARE_ARGS+=(--no-tfrecord)
fi
if [[ "$PROFILE" == "full" ]]; then
  PREPARE_ARGS+=(--full-tfrecord)
fi
python "${PREPARE_ARGS[@]}"

if [[ "$PROFILE" == "full" ]]; then
  DATA_YAML="$DATA_DIR/coco_yolo26.yaml"
  EPOCHS="$EPOCHS_FULL"
else
  DATA_YAML="$DATA_DIR/coco_yolo26_subset${SUBSET}.yaml"
  EPOCHS="$EPOCHS_SMALL"
fi

start_gpu_monitor

python -m yolo26_tf.cli detect train \
  --model yolo26n.yaml \
  --data "$DATA_YAML" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --project "$PROJECT" \
  --name "$NAME" \
  --optimizer "$OPTIMIZER" \
  --lr0 "${YOLO26_COCO_LR0:-0.01}" \
  --lrf "${YOLO26_COCO_LRF:-0.01}" \
  --momentum "${YOLO26_COCO_MOMENTUM:-0.937}" \
  --weight-decay "${YOLO26_COCO_WEIGHT_DECAY:-0.0005}" \
  --warmup-epochs "${YOLO26_COCO_WARMUP_EPOCHS:-3.0}" \
  --close-mosaic "${YOLO26_COCO_CLOSE_MOSAIC:-10}" \
  --workers "${YOLO26_COCO_WORKERS:-8}" \
  --cache \
  --cache-images "$CACHE_IMAGES" \
  --cache-ram-gb "$CACHE_RAM_GB" \
  --require-gpu \
  --val-coco \
  $([[ "$AMP" == "1" ]] && echo "--amp" || echo "--no-amp") \
  $([[ "$USE_TFRECORD" == "1" ]] && echo "--use-tfrecord" || echo "--no-tfrecord") \
  $([[ "$COMPILE_STEP" == "1" ]] && echo "--compile" || echo "--no-compile") \
  $([[ "$FAST_DATA" == "1" ]] && echo "--fast-data" || echo "--no-fast-data") \
  $([[ "$PREFETCH_DATA" == "1" ]] && echo "--prefetch-data" || echo "--no-prefetch-data") \
  $([[ "$FAST_NMS" == "1" ]] && echo "--fast-nms" || echo "--no-fast-nms") \
  $([[ "$PROFILE_SPEED" == "1" ]] && echo "--profile-speed" || echo "--no-profile-speed") \
  $([[ "$PROFILE_STAGE" == "1" ]] && echo "--profile-stage" || echo "") \
  $([[ "$SYNC_PROFILE_STAGE" == "1" ]] && echo "--sync-profile-stage" || echo "") \
  --ema-update-interval "$EMA_UPDATE_INTERVAL" \
  --profile-batches "$PROFILE_BATCHES"

cleanup
summarize_gpu_stats

if [[ "$PROFILE_BATCHES" != "0" ]]; then
  echo "Profiling run complete after ${PROFILE_BATCHES} batch(es); skipping final validation and export."
  echo "Profiler outputs: $PROJECT/$NAME/stage_profile.csv and $PROJECT/$NAME/results.csv"
  exit 0
fi

BEST="$PROJECT/$NAME/weights/best.weights.h5"
python -m yolo26_tf.cli detect val \
  --model yolo26n.yaml \
  --weights "$BEST" \
  --data "$DATA_YAML" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --conf 0.001 \
  --iou 0.7 \
  --max-det 300 \
  --coco \
  --save-json \
  $([[ "$FAST_NMS" == "1" ]] && echo "--fast-nms" || echo "--no-fast-nms") \
  --project "$PROJECT/$NAME" \
  --name final_coco_val

TFLITE="$PROJECT/$NAME/weights/best_fp32.tflite"
python -m yolo26_tf.cli detect export \
  --model yolo26n.yaml \
  --weights "$BEST" \
  --format tflite \
  --output "$TFLITE" \
  --imgsz "$IMGSZ"

python - "$TFLITE" "$IMGSZ" <<'PY'
import sys
import numpy as np
import tensorflow as tf

path = sys.argv[1]
imgsz = int(sys.argv[2])
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()
inputs = interpreter.get_input_details()
outputs = interpreter.get_output_details()
interpreter.set_tensor(inputs[0]["index"], np.zeros((1, imgsz, imgsz, 3), dtype=np.float32))
interpreter.invoke()
out = interpreter.get_tensor(outputs[0]["index"])
print("TFLite reload inference ok:", out.shape, out.dtype)
PY

echo "Training outputs: $PROJECT/$NAME"
