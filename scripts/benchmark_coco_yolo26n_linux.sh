#!/usr/bin/env bash
# Create a Linux benchmark environment, download COCO val2017, and run YOLO26n TF COCO mAP.
#
# Usage:
#   bash scripts/benchmark_coco_yolo26n_linux.sh
#   bash scripts/benchmark_coco_yolo26n_linux.sh --limit 100   # quick smoke test
#
# Environment overrides:
#   PYTHON=python3.11
#   YOLO26_BENCH_VENV=.venv-coco-bench
#   YOLO26_BENCH_DATA=datasets/coco
#   YOLO26_BENCH_OUT=runs/benchmark/yolo26n_tf_coco
#   YOLO26_BENCH_BATCH=16
#   YOLO26_BENCH_IMGSZ=640
#   YOLO26_BENCH_TENSORFLOW='tensorflow[and-cuda]==2.15.1'
#   YOLO26_BENCH_NUMPY='numpy>=1.23.5,<2.0'

set -euo pipefail

export KERAS_BACKEND="${KERAS_BACKEND:-tensorflow}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_XLA_FLAGS="${TF_XLA_FLAGS:---tf_xla_auto_jit=0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${YOLO26_BENCH_VENV:-$ROOT_DIR/.venv-coco-bench}"
DATA_DIR="${YOLO26_BENCH_DATA:-$ROOT_DIR/datasets/coco}"
OUT_DIR="${YOLO26_BENCH_OUT:-$ROOT_DIR/runs/benchmark/yolo26n_tf_coco}"
BATCH="${YOLO26_BENCH_BATCH:-16}"
IMGSZ="${YOLO26_BENCH_IMGSZ:-640}"
TENSORFLOW_PACKAGE="${YOLO26_BENCH_TENSORFLOW:-tensorflow[and-cuda]==2.15.1}"
NUMPY_PACKAGE="${YOLO26_BENCH_NUMPY:-numpy>=1.23.5,<2.0}"
WEIGHTS="$OUT_DIR/yolo26n.pt"
TF_WEIGHTS="$OUT_DIR/yolo26n_tf.weights.h5"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
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

echo "Creating/updating virtualenv: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install "$NUMPY_PACKAGE" "$TENSORFLOW_PACKAGE"
python -m pip install -e "$ROOT_DIR[convert]" "pycocotools>=2.0.7" "tqdm>=4.66" "$NUMPY_PACKAGE"
python -m pip install --force-reinstall "$NUMPY_PACKAGE"
python -m pip check

echo "Verifying NumPy/TensorFlow ABI compatibility"
python - <<'PY'
from importlib.metadata import version

npv = version("numpy")
print("NumPy:", npv)
if int(npv.split(".", 1)[0]) >= 2:
    raise SystemExit(
        f"NumPy {npv} is incompatible with TensorFlow 2.15.x. "
        "The runner pins numpy<2; remove the venv and rerun if this persists."
    )
PY

echo "Verifying TensorFlow GPU runtime"
python - <<'PY'
import os
import tensorflow as tf

tf.config.optimizer.set_jit(False)
print("TensorFlow:", tf.__version__)
print("TensorFlow XLA JIT:", tf.config.optimizer.get_jit(), "TF_XLA_FLAGS=", os.environ.get("TF_XLA_FLAGS", ""))
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if not gpus:
    raise SystemExit("No TensorFlow GPU detected. This benchmark is GPU-only.")
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
PY

mkdir -p "$DATA_DIR" "$OUT_DIR"

download_file "http://images.cocodataset.org/zips/val2017.zip" "$DATA_DIR/val2017.zip"
download_file "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$DATA_DIR/annotations_trainval2017.zip"

if [[ ! -d "$DATA_DIR/val2017" ]]; then
  echo "Extracting val2017 images"
  unzip -q "$DATA_DIR/val2017.zip" -d "$DATA_DIR"
fi

if [[ ! -f "$DATA_DIR/annotations/instances_val2017.json" ]]; then
  echo "Extracting COCO annotations"
  unzip -q "$DATA_DIR/annotations_trainval2017.zip" -d "$DATA_DIR"
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Downloading official yolo26n.pt via pinned Ultralytics"
  python - "$WEIGHTS" <<'PY'
from pathlib import Path
import shutil
import sys

from ultralytics.utils.downloads import attempt_download_asset

target = Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
src = Path(attempt_download_asset("yolo26n.pt"))
if src.resolve() != target.resolve():
    shutil.copy2(src, target)
print(target)
PY
fi

echo "Running TensorFlow YOLO26n COCO val2017 benchmark"
python "$ROOT_DIR/scripts/benchmark_coco_yolo26n.py" \
  --coco-root "$DATA_DIR" \
  --weights "$WEIGHTS" \
  --tf-weights "$TF_WEIGHTS" \
  --out "$OUT_DIR" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  "$@"

echo "Benchmark results: $OUT_DIR/results_yolo26n_tf_coco_val2017.json"
