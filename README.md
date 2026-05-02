# Ultralytics YOLO26 TensorFlow/Keras Detection Port

This package is a TensorFlow/Keras port of the public Ultralytics YOLO26 object-detection implementation pinned to upstream commit `b4cf7c4751e1d532eb5b0f5a3e9d67b9583964a7`.

Scope: object detection only (`yolo26.yaml`, `yolo26-p2.yaml`, `yolo26-p6.yaml`). Segmentation, pose, OBB, classification, tracking, HUB integrations, and YOLOE are intentionally out of scope.

License: AGPL-3.0, matching the upstream Ultralytics project.

## COCO YOLO26n Benchmark

Ultralytics reports YOLO26n at image size 640 on COCO val2017 with:

- `mAP50-95(B) = 40.9`
- `mAP50-95(B, e2e) = 40.1`

Reference: https://github.com/ultralytics/ultralytics and https://github.com/ultralytics/ultralytics/blob/main/docs/en/tasks/detect.md. Ultralytics reproduces the official PyTorch number with:

```bash
yolo val detect data=coco.yaml device=0
```

For this TensorFlow port, use the Linux benchmark runner below. It creates a fresh virtualenv, installs this package with TensorFlow and conversion dependencies, downloads COCO val2017 images plus annotations, downloads `yolo26n.pt`, converts it to TensorFlow weights, runs pycocotools COCOeval, and writes predictions/results JSON.

```bash
bash scripts/benchmark_coco_yolo26n_linux.sh
```

Useful smoke test before the full 5000-image run:

```bash
bash scripts/benchmark_coco_yolo26n_linux.sh --limit 100
```

Optional environment overrides:

```bash
PYTHON=python3.11 \
YOLO26_BENCH_VENV=.venv-coco-bench \
YOLO26_BENCH_DATA=datasets/coco \
YOLO26_BENCH_OUT=runs/benchmark/yolo26n_tf_coco \
YOLO26_BENCH_BATCH=16 \
YOLO26_BENCH_IMGSZ=640 \
YOLO26_BENCH_NUMPY='numpy>=1.23.5,<2.0' \
YOLO26_BENCH_TENSORFLOW='tensorflow[and-cuda]==2.15.1' \
bash scripts/benchmark_coco_yolo26n_linux.sh
```

Outputs:

- `runs/benchmark/yolo26n_tf_coco/predictions_yolo26n_tf_coco_val2017.json`
- `runs/benchmark/yolo26n_tf_coco/results_yolo26n_tf_coco_val2017.json`
- `runs/benchmark/yolo26n_tf_coco/yolo26n_tf.weights.h5`

Notes:

- COCO train2017 is not required for validation mAP; the script downloads val2017 and `instances_val2017.json`.
- The TensorFlow benchmark defaults to NMS-free end-to-end YOLO26 evaluation with `conf=0.001`, `iou=0.7`, `max_det=300`, and image size 640.
- Add `--nms` if you explicitly want an NMS compatibility run, but compare the default run against the official `mAP50-95(e2e)` target of `40.1`.
- The Linux benchmark is GPU-only and does not fall back to CPU. It verifies TensorFlow GPU visibility and runs a GPU `Conv2D` sanity check before COCO evaluation.
- For NVIDIA driver/CUDA `535.183 / 12.2`, use Python 3.10 or 3.11 and keep the default `YOLO26_BENCH_TENSORFLOW='tensorflow[and-cuda]==2.15.1'`, because TensorFlow 2.15 uses CUDA 12.2. Newer TensorFlow versions may require a newer NVIDIA driver/CUDA runtime.
- TensorFlow 2.15.x requires NumPy 1.x. The runner pins `YOLO26_BENCH_NUMPY='numpy>=1.23.5,<2.0'`, runs `pip check`, and fails before TensorFlow import if NumPy 2.x is present.
- `imgsz` is normalized to a stride multiple. For the official YOLO26n benchmark, keep `YOLO26_BENCH_IMGSZ=640`.

## COCO Scratch Training

The TensorFlow training stack now includes the YOLO26n detection pieces needed for real scratch COCO runs: YOLO/COCO dataset loading, label verification/cache metadata, mosaic/random-perspective/mixup/cutmix/HSV/flips, close-mosaic, multi-scale training, EMA, warmup/cosine LR, gradient clipping/accumulation, AMP, checkpoint resume, COCOeval validation, and TFLite export/reload verification.

Use the Linux GPU-only runner:

```bash
bash scripts/train_coco_yolo26n_linux.sh
```

The default profile is a small COCO subset smoke run so the pipeline can be checked quickly:

```bash
YOLO26_COCO_PROFILE=small \
YOLO26_COCO_SUBSET=100 \
YOLO26_COCO_VAL_SUBSET=100 \
YOLO26_COCO_EPOCHS_SMALL=2 \
bash scripts/train_coco_yolo26n_linux.sh
```

For a full scratch COCO training run:

```bash
YOLO26_COCO_PROFILE=full \
YOLO26_COCO_EPOCHS_FULL=300 \
YOLO26_COCO_BATCH=16 \
YOLO26_COCO_IMGSZ=640 \
YOLO26_COCO_NUMPY='numpy>=1.23.5,<2.0' \
bash scripts/train_coco_yolo26n_linux.sh
```

The script:

- creates a fresh virtualenv;
- installs `tensorflow[and-cuda]==2.15.1` by default for CUDA 12.2-class Linux systems;
- pins NumPy to `<2.0`, runs `pip check`, and fails early if a stale/broken venv still has NumPy 2.x;
- fails early if TensorFlow cannot see/use GPUs;
- downloads COCO `train2017`, `val2017`, and annotations;
- converts `instances_train2017.json` and `instances_val2017.json` into YOLO labels;
- trains scratch `yolo26n.yaml` with `yolo26-tf detect train`;
- validates with pycocotools COCOeval;
- exports `best.weights.h5` to TFLite and reloads the TFLite model for inference.
- streams all shell/training output to `runs/train/yolo26n_tf_coco/train_coco_yolo26n.log` by default.

Important: the code path is now COCO-capable, but the repository does not claim the official 40.1 e2e mAP until a complete full COCO scratch training run has actually been executed and recorded. Use converted-checkpoint validation for direct checkpoint parity, and use the full profile above for scratch reproduction experiments.

## Parity Checks

Use the optional parity harness when PyTorch and Ultralytics are installed:

```bash
python scripts/parity_check_yolo26.py --weights yolo26n.pt --imgsz 64
```

It compares:

- converted TensorFlow forward outputs against the upstream PyTorch checkpoint;
- TensorFlow `TaskAlignedAssigner` behavior against Ultralytics for small-box and multi-GT conflict cases.

The automated test suite also includes these assigner parity checks when the optional upstream dependencies are available.
