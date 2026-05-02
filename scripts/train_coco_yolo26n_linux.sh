#!/usr/bin/env bash
# Create a Linux GPU TensorFlow environment, prepare COCO, train scratch YOLO26n,
# validate with COCOeval, and export/reload TFLite.
#
# Default profile is a practical COCO-small smoke run.  Use
#   YOLO26_COCO_PROFILE=full bash scripts/train_coco_yolo26n_linux.sh
# for full COCO scratch training.

set -euo pipefail

export KERAS_BACKEND="${KERAS_BACKEND:-tensorflow}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_XLA_FLAGS="${TF_XLA_FLAGS:---tf_xla_auto_jit=0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${YOLO26_COCO_VENV:-$ROOT_DIR/.venv-coco-train}"
DATA_DIR="${YOLO26_COCO_DATA:-$ROOT_DIR/datasets/coco}"
OUT_DIR="${YOLO26_COCO_OUT:-$ROOT_DIR/runs/train/yolo26n_tf_coco}"
PROFILE="${YOLO26_COCO_PROFILE:-small}"
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

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install "$NUMPY_PACKAGE" "$TENSORFLOW_PACKAGE"
python -m pip install -e "$ROOT_DIR[convert,dev]" "pycocotools>=2.0.7" "tqdm>=4.66" "$NUMPY_PACKAGE"
python -m pip install --force-reinstall "$NUMPY_PACKAGE"
python -m pip check

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

python "$ROOT_DIR/scripts/prepare_coco_yolo.py" \
  --coco-root "$DATA_DIR" \
  --output-yaml "$DATA_DIR/coco_yolo26.yaml" \
  --train-subset "$SUBSET" \
  --val-subset "$VAL_SUBSET" \
  --summary "$OUT_DIR/prepare_coco_summary.json"

if [[ "$PROFILE" == "full" ]]; then
  DATA_YAML="$DATA_DIR/coco_yolo26.yaml"
  EPOCHS="$EPOCHS_FULL"
else
  DATA_YAML="$DATA_DIR/coco_yolo26_subset${SUBSET}.yaml"
  EPOCHS="$EPOCHS_SMALL"
fi

python -m yolo26_tf.cli detect train \
  --model yolo26n.yaml \
  --data "$DATA_YAML" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --project "$PROJECT" \
  --name "$NAME" \
  --optimizer "${YOLO26_COCO_OPTIMIZER:-auto}" \
  --lr0 "${YOLO26_COCO_LR0:-0.01}" \
  --lrf "${YOLO26_COCO_LRF:-0.01}" \
  --momentum "${YOLO26_COCO_MOMENTUM:-0.937}" \
  --weight-decay "${YOLO26_COCO_WEIGHT_DECAY:-0.0005}" \
  --warmup-epochs "${YOLO26_COCO_WARMUP_EPOCHS:-3.0}" \
  --close-mosaic "${YOLO26_COCO_CLOSE_MOSAIC:-10}" \
  --workers "${YOLO26_COCO_WORKERS:-4}" \
  --cache \
  --amp \
  --require-gpu \
  --val-coco

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
