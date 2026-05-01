"""Benchmark the TensorFlow YOLO26n port on COCO val2017 with pycocotools.

The official Ultralytics YOLO26n e2e target is COCO val2017 mAP50-95=40.1
at image size 640.  This script evaluates the converted TensorFlow model with
the same COCO API metric.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


COCO80_TO_COCO91 = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]


def load_coco_api():
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:
        raise RuntimeError("Install pycocotools first, e.g. `pip install pycocotools`.") from exc
    return COCO, COCOeval


def configure_tensorflow_runtime(args: argparse.Namespace):
    """Configure TensorFlow before importing yolo26_tf modules.

    TensorFlow initializes CUDA visibility at import time.  This must happen
    before importing any yolo26_tf module because those modules lazily import
    TensorFlow at module import.
    """
    force_cpu = args.device == "cpu" or os.environ.get("YOLO26_TF_FORCE_CPU") == "1"
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", str(args.tf_log_level))

    import tensorflow as tf  # type: ignore

    if force_cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            # TensorFlow may already be initialized if the caller imported it.
            pass
        print("TensorFlow device mode: CPU (GPU disabled)", flush=True)
        return tf

    gpus = tf.config.list_physical_devices("GPU")
    if args.device == "gpu" and not gpus:
        raise RuntimeError("Requested --device gpu, but TensorFlow found no visible GPUs.")
    if gpus and args.gpu_memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        print(f"TensorFlow device mode: GPU auto, memory_growth=True, gpus={len(gpus)}", flush=True)
    elif gpus:
        print(f"TensorFlow device mode: GPU auto, memory_growth=False, gpus={len(gpus)}", flush=True)
    else:
        print("TensorFlow device mode: CPU (no visible GPU)", flush=True)
    return tf


def normalize_imgsz(imgsz: int, stride: int = 32) -> int:
    """Match Ultralytics-style image-size validation for stride-multiple inputs."""
    if imgsz % stride == 0:
        return imgsz
    adjusted = int(math.ceil(imgsz / stride) * stride)
    print(f"WARNING: imgsz={imgsz} is not divisible by stride={stride}; using imgsz={adjusted}.", flush=True)
    return adjusted


def is_gpu_dnn_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = ("cudnn_status_not_initialized", "no dnn in stream executor", "could not create cudnn handle")
    return any(x in text for x in needles)


def load_tf_model(weights: Path, tf_weights: Path, imgsz: int, max_det: int, verify_conversion: bool):
    from yolo26_tf.converter import convert_pt_to_tf
    from yolo26_tf.model import build_model

    if tf_weights.exists():
        model = build_model("yolo26n.yaml", nc=80, imgsz=imgsz)
        model.load_weights(str(tf_weights))
    else:
        model = convert_pt_to_tf(weights, output=tf_weights, imgsz=imgsz, nc=80, verify=verify_conversion)
    for layer in getattr(model, "layers_seq", []):
        if hasattr(layer, "max_det"):
            layer.max_det = int(max_det)
    return model


def iter_image_batches(coco, image_ids: list[int], image_dir: Path, imgsz: int, batch_size: int):
    from yolo26_tf.ops import letterbox

    batch, metas = [], []
    for image_id in image_ids:
        info = coco.loadImgs([image_id])[0]
        image_path = image_dir / info["file_name"]
        img0 = np.asarray(Image.open(image_path).convert("RGB"))
        img, ratio, pad = letterbox(img0, imgsz, scaleup=False)
        batch.append(img.astype(np.float32) / 255.0)
        metas.append({"image_id": image_id, "shape": img0.shape[:2], "ratio": ratio, "pad": pad})
        if len(batch) == batch_size:
            yield np.stack(batch, axis=0), metas
            batch, metas = [], []
    if batch:
        yield np.stack(batch, axis=0), metas


def prediction_to_coco_rows(pred: np.ndarray, meta: dict, imgsz: int, conf: float, iou: float, max_det: int, nms: bool) -> list[dict]:
    from yolo26_tf.ops import nms_numpy, scale_boxes_np

    if pred.size == 0:
        return []
    if pred.shape[-1] == 6:
        det = pred[pred[:, 4] >= conf]
        if nms and len(det):
            det = nms_numpy(det, conf=conf, iou=iou, max_det=max_det)
        elif len(det) > max_det:
            det = det[np.argsort(-det[:, 4])[:max_det]]
    else:
        boxes, scores = pred[:, :4], pred[:, 4:]
        cls = scores.argmax(axis=-1)
        score = scores.max(axis=-1)
        det = nms_numpy(np.concatenate([boxes, score[:, None], cls[:, None]], axis=-1), conf=conf, iou=iou, max_det=max_det)
    if len(det) == 0:
        return []

    boxes = scale_boxes_np(det[:, :4], (imgsz, imgsz), meta["shape"], ratio_pad=((meta["ratio"][0], meta["ratio"][1]), meta["pad"]))
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, meta["shape"][1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, meta["shape"][0])
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    valid = (wh[:, 0] > 0) & (wh[:, 1] > 0)
    rows = []
    for box, wh_i, row in zip(boxes[valid], wh[valid], det[valid]):
        cls_idx = int(row[5])
        if cls_idx < 0 or cls_idx >= len(COCO80_TO_COCO91):
            continue
        rows.append(
            {
                "image_id": int(meta["image_id"]),
                "category_id": int(COCO80_TO_COCO91[cls_idx]),
                "bbox": [float(box[0]), float(box[1]), float(wh_i[0]), float(wh_i[1])],
                "score": float(row[4]),
            }
        )
    return rows


def evaluate_coco(coco, coco_eval_cls, image_ids: list[int], predictions: list[dict]) -> dict:
    if not predictions:
        raise RuntimeError("No predictions were produced; cannot run COCOeval.")
    coco_dt = coco.loadRes(predictions)
    evaluator = coco_eval_cls(coco, coco_dt, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats.tolist()
    return {
        "metrics/mAP50-95(B)": float(stats[0]),
        "metrics/mAP50(B)": float(stats[1]),
        "metrics/mAP75(B)": float(stats[2]),
        "metrics/mAP50-95_small(B)": float(stats[3]),
        "metrics/mAP50-95_medium(B)": float(stats[4]),
        "metrics/mAP50-95_large(B)": float(stats[5]),
        "metrics/AR1(B)": float(stats[6]),
        "metrics/AR10(B)": float(stats[7]),
        "metrics/AR100(B)": float(stats[8]),
    }


def run(args: argparse.Namespace) -> dict:
    tf = configure_tensorflow_runtime(args)
    args.imgsz = normalize_imgsz(args.imgsz)

    coco_root = Path(args.coco_root)
    ann_file = coco_root / "annotations" / "instances_val2017.json"
    image_dir = coco_root / "val2017"
    if not ann_file.exists():
        raise FileNotFoundError(f"Missing COCO annotations: {ann_file}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing COCO val images: {image_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    COCO, COCOeval = load_coco_api()
    coco = COCO(str(ann_file))
    image_ids = sorted(coco.getImgIds())
    if args.limit:
        image_ids = image_ids[: args.limit]

    model = load_tf_model(Path(args.weights), Path(args.tf_weights), args.imgsz, args.max_det, args.verify_conversion)
    predictions = []
    start = time.time()
    n_images = 0
    for batch, metas in iter_image_batches(coco, image_ids, image_dir, args.imgsz, args.batch):
        raw = model(tf.convert_to_tensor(batch, tf.float32), training=False).numpy()
        for pred, meta in zip(raw, metas):
            predictions.extend(prediction_to_coco_rows(pred, meta, args.imgsz, args.conf, args.iou, args.max_det, args.nms))
        n_images += len(metas)
        if n_images % max(args.log_every, 1) == 0 or n_images == len(image_ids):
            print(f"processed {n_images}/{len(image_ids)} images, predictions={len(predictions)}", flush=True)

    pred_file = out_dir / "predictions_yolo26n_tf_coco_val2017.json"
    pred_file.write_text(json.dumps(predictions), encoding="utf-8")
    metrics = evaluate_coco(coco, COCOeval, image_ids, predictions)
    result = {
        "model": "yolo26n",
        "backend": "tensorflow",
        "weights": str(args.weights),
        "tf_weights": str(args.tf_weights),
        "coco_root": str(coco_root),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "nms": args.nms,
        "images": len(image_ids),
        "predictions": len(predictions),
        "seconds": time.time() - start,
        "target_ultralytics_yolo26n_mAP50_95_e2e": 0.401,
        "metrics": metrics,
    }
    (out_dir / "results_yolo26n_tf_coco_val2017.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO val2017 benchmark for TensorFlow YOLO26n.")
    parser.add_argument("--coco-root", required=True, help="Directory containing val2017/ and annotations/instances_val2017.json.")
    parser.add_argument("--weights", default="yolo26n.pt", help="Official Ultralytics yolo26n.pt checkpoint.")
    parser.add_argument("--tf-weights", default="runs/benchmark/yolo26n_tf_coco/yolo26n_tf.weights.h5", help="Cached converted TF weights.")
    parser.add_argument("--out", default="runs/benchmark/yolo26n_tf_coco", help="Output directory for predictions and metrics.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0, help="Optional image limit for smoke tests; use 0 for all 5000 val images.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--nms", action="store_true", help="Apply NMS to e2e predictions. Default is NMS-free YOLO26 e2e evaluation.")
    parser.add_argument("--verify-conversion", action="store_true", help="Run one random PyTorch-vs-TF parity check during conversion.")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto", help="TensorFlow device mode. Use cpu to avoid CUDA/cuDNN driver mismatches.")
    parser.add_argument("--gpu-memory-growth", dest="gpu_memory_growth", action="store_true", default=True, help="Enable TensorFlow GPU memory growth before model creation.")
    parser.add_argument("--no-gpu-memory-growth", dest="gpu_memory_growth", action="store_false", help="Disable TensorFlow GPU memory growth.")
    parser.add_argument("--no-cpu-fallback", dest="cpu_fallback", action="store_false", default=True, help="Disable automatic CPU re-exec on CUDA/cuDNN initialization errors.")
    parser.add_argument("--tf-log-level", type=int, default=2, choices=(0, 1, 2, 3), help="TensorFlow C++ log level.")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    try:
        run(parsed)
    except Exception as exc:
        if parsed.cpu_fallback and parsed.device == "auto" and os.environ.get("YOLO26_TF_FORCE_CPU") != "1" and is_gpu_dnn_error(exc):
            print(
                "GPU/cuDNN initialization failed; re-running benchmark with CUDA_VISIBLE_DEVICES=-1. "
                "Use --device cpu to select this explicitly or --no-cpu-fallback to fail instead.",
                file=sys.stderr,
                flush=True,
            )
            env = os.environ.copy()
            env["YOLO26_TF_FORCE_CPU"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = "-1"
            os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
        print(f"benchmark failed: {exc}", file=sys.stderr)
        raise
