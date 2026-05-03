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

The TensorFlow training stack now includes the YOLO26n detection pieces needed for real scratch COCO runs: YOLO/COCO dataset loading, threaded label verification/cache metadata with hash/version checks, segment-aware labels, class filtering, `single_cls`, rectangular validation shapes, flat `batch_idx/cls/bboxes` targets, Ultralytics-style transform objects (`Instances`, `Compose`, `LetterBox`, `Mosaic`, `RandomPerspective`, `CopyPaste`, `MixUp`, `CutMix`, `Albumentations`, `RandomHSV`, `RandomFlip`, `Format`) wired through the trainer iterator, close-mosaic, multi-scale training, EMA, warmup/cosine LR, gradient clipping/accumulation, AMP, class-weight scaling, freeze/time controls, CSV/results logging, checkpoint resume, NaN checkpoint recovery, final best-checkpoint validation, COCOeval validation, and SavedModel/TFLite export/reload verification.

Use the Linux GPU-only runner from the repo root. By default this starts full COCO scratch training with the stable RTX A6000 settings documented below:

```bash
bash scripts/train_coco_yolo26n_linux.sh
```

For a small COCO subset smoke run, override the profile:

```bash
YOLO26_COCO_PROFILE=small \
YOLO26_COCO_SUBSET=100 \
YOLO26_COCO_VAL_SUBSET=100 \
YOLO26_COCO_EPOCHS_SMALL=2 \
bash scripts/train_coco_yolo26n_linux.sh
```

The training runner now defaults to the stability path:

- stable eager TensorFlow gradient step by default: `YOLO26_COCO_COMPILE=0`;
- stable FP32 training by default: `YOLO26_COCO_AMP=0`;
- serial Python data iterator by default: `YOLO26_COCO_FAST_DATA=0`;
- TFRecord generation and training input when available: `YOLO26_COCO_USE_TFRECORD=1`;
- bounded record/image RAM cache: `YOLO26_COCO_CACHE_IMAGES=auto`, `YOLO26_COCO_CACHE_RAM_GB=32`;
- TensorFlow graph NMS during validation: `YOLO26_COCO_FAST_NMS=1`;
- speed profiling in batch logs plus `results.csv`/`results.json`: `speed/data_ms_per_batch`, `speed/train_ms_per_batch`, `speed/images_per_sec`, and `speed/val_ms`.

The default full scratch COCO command is:

```bash
cd /home/anup/git/anup-code/yolo26

bash scripts/train_coco_yolo26n_linux.sh
```

Training logs are streamed to:

```bash
runs/train/yolo26n_tf_coco/train_coco_yolo26n.log
```

`YOLO26_COCO_BATCH=16`, `YOLO26_COCO_AMP=0`, and `YOLO26_COCO_FAST_DATA=0` are the stable starting point for an RTX A6000 48 GB run. After a stable epoch completes, tune in this order: try batch `32`, then `YOLO26_COCO_FAST_DATA=1`, then batch `48`, then `YOLO26_COCO_AMP=1`. Keep `YOLO26_COCO_COMPILE=0` for stable training; `YOLO26_COCO_COMPILE=1` is an experimental speed path that can trigger unrecoverable TensorFlow GPU CUDA faults on some systems. If the GPU still crashes, run one diagnostic pass with `YOLO26_COCO_CUDA_SYNC=1 bash scripts/train_coco_yolo26n_linux.sh`. The full profile writes full COCO TFRecords by default; subset profiles write subset TFRecords. If system RAM is too constrained for the record cache, lower `YOLO26_COCO_CACHE_RAM_GB` or set `YOLO26_COCO_USE_TFRECORD=0` to use the image-file path.

The script:

- creates a fresh virtualenv;
- installs `tensorflow[and-cuda]==2.15.1` by default for CUDA 12.2-class Linux systems;
- pins NumPy to `<2.0`, runs `pip check`, and fails early if a stale/broken venv still has NumPy 2.x;
- fails early if TensorFlow cannot see/use GPUs;
- downloads COCO `train2017`, `val2017`, and annotations;
- converts `instances_train2017.json` and `instances_val2017.json` into YOLO labels;
- writes TFRecords for the selected training profile unless disabled;
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
- TensorFlow `BboxLoss(reg_max=1)` box and DFL/L1 terms against Ultralytics on a controlled synthetic batch.
- TensorFlow YOLO26 `E2ELoss` branch schedule: one-to-many top-k 10, one-to-one top-k 7/top-k2 1, and progressive `o2m/o2o` decay.

Run individual checks when debugging:

```bash
python scripts/parity_check_yolo26.py --weights yolo26n.pt --imgsz 64 --forward
python scripts/parity_check_yolo26.py --weights yolo26n.pt --imgsz 64 --assigner
python scripts/parity_check_yolo26.py --weights yolo26n.pt --imgsz 64 --bbox-loss
python scripts/parity_check_yolo26.py --weights yolo26n.pt --imgsz 64 --e2e-loss
```

The automated test suite also covers dataset cache/rectangular target behavior, trainer transform/collate contracts, AP50-95 stats accumulation, confusion matrix accounting, multi-label postprocess, optimizer grouping, tiny training, prediction, and SavedModel/TFLite export/reload smoke paths.
