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
YOLO26_BENCH_DEVICE=auto \
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
- If TensorFlow fails with `CUDNN_STATUS_NOT_INITIALIZED`, `Could not create cudnn handle`, or `No DNN in stream executor`, rerun with `YOLO26_BENCH_DEVICE=cpu bash scripts/benchmark_coco_yolo26n_linux.sh`. The benchmark also auto-falls back to CPU in `auto` mode for this specific CUDA/cuDNN initialization failure.
- `imgsz` is normalized to a stride multiple. For the official YOLO26n benchmark, keep `YOLO26_BENCH_IMGSZ=640`.
