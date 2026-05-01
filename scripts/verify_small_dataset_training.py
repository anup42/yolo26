"""Run deterministic YOLO26 TensorFlow overfit checks on a tiny detection dataset.

This is intentionally small and CPU-friendly.  It verifies:
- converted PyTorch yolo26n weights can fine-tune a 1-class model
- a scratch yolo26n model can fit the same tiny training set
- predictions produce class-0 boxes with good IoU on the training images
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_tiny_dataset import create_tiny_dataset
from yolo26_tf.api import YOLO26
from yolo26_tf.converter import convert_pt_to_tf
from yolo26_tf.data import YOLODataset
from yolo26_tf.losses import E2ELoss
from yolo26_tf.model import build_model
from yolo26_tf.ops import letterbox, scale_boxes_np, xywh2xyxy_np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_model(
    model: tf.keras.Model,
    dataset_yaml: Path,
    out_weights: Path,
    *,
    epochs: int,
    batch: int,
    lr: float,
    imgsz: int,
    nc: int,
    augment: bool = False,
) -> dict:
    hyp = {
        "box": 7.5,
        "cls": 0.5,
        "lr0": lr,
        "lrf": 1.0,
        "warmup_epochs": 0.0,
        "weight_decay": 5e-4,
        "mosaic": 0.0 if not augment else 1.0,
        "mixup": 0.0 if not augment else 0.15,
        "cutmix": 0.0 if not augment else 0.15,
        "copy_paste": 0.0,
        "hsv_h": 0.0 if not augment else 0.015,
        "hsv_s": 0.0 if not augment else 0.7,
        "hsv_v": 0.0 if not augment else 0.4,
        "fliplr": 0.0 if not augment else 0.5,
        "flipud": 0.0,
        "degrees": 0.0 if not augment else 10.0,
        "translate": 0.0 if not augment else 0.1,
        "scale": 0.0 if not augment else 0.5,
        "shear": 0.0 if not augment else 2.0,
        "perspective": 0.0,
        "multi_scale": False,
    }
    dataset = YOLODataset(
        dataset_yaml,
        split="train",
        imgsz=imgsz,
        batch=batch,
        augment=augment,
        hyp=hyp,
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=hyp["weight_decay"])
    loss_fn = E2ELoss(model, hyp=hyp)
    losses: list[float] = []

    # Fixed dataset order makes this an overfit test rather than a loader stress test.
    for _ in range(epochs):
        epoch_losses: list[float] = []
        for batch_data in dataset:
            batch_tf = {
                key: tf.convert_to_tensor(value) if key in {"img", "bboxes", "cls", "mask"} else value
                for key, value in batch_data.items()
            }
            with tf.GradientTape() as tape:
                preds = model(batch_tf["img"], training=True)
                loss, _ = loss_fn(preds, batch_tf)
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.clip_by_norm(g, 10.0) if g is not None else None for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(float(loss.numpy()))
        losses.append(float(np.mean(epoch_losses)))

    out_weights.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(out_weights)
    return {
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "losses": losses,
        "weights": str(out_weights),
    }


def load_label(label_path: Path, image_shape: tuple[int, int, int]) -> np.ndarray:
    lines = [x.strip() for x in label_path.read_text().splitlines() if x.strip()]
    if not lines:
        return np.zeros((0, 4), dtype=np.float32)
    labels = np.asarray([[float(v) for v in line.split()] for line in lines], dtype=np.float32)
    h, w = image_shape[:2]
    boxes = xywh2xyxy_np(labels[:, 1:5])
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h
    return boxes


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    area1 = np.maximum(boxes1[:, 2] - boxes1[:, 0], 0) * np.maximum(boxes1[:, 3] - boxes1[:, 1], 0)
    area2 = np.maximum(boxes2[:, 2] - boxes2[:, 0], 0) * np.maximum(boxes2[:, 3] - boxes2[:, 1], 0)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.maximum(rb - lt, 0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-7)


def evaluate_model(
    model: tf.keras.Model,
    dataset_dir: Path,
    *,
    imgsz: int,
    confs: tuple[float, ...] = (0.001, 0.01, 0.05, 0.25),
) -> dict:
    from PIL import Image

    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    image_paths = sorted(images_dir.glob("*.jpg"))
    metrics = {str(conf): {"any": 0, "class0": 0, "iou50": 0, "best_ious": [], "top_confs": []} for conf in confs}
    examples = []

    for image_path in image_paths:
        pil = Image.open(image_path).convert("RGB")
        original = np.asarray(pil)
        gt = load_label(labels_dir / f"{image_path.stem}.txt", original.shape)
        resized, ratio, pad = letterbox(original, (imgsz, imgsz))
        raw = model(resized[None].astype(np.float32) / 255.0, training=False).numpy()[0]
        boxes = scale_boxes_np(raw[:, :4], resized.shape[:2], original.shape[:2], ratio_pad=(ratio, pad))
        conf = raw[:, 4]
        cls = raw[:, 5].astype(np.int32)
        top_idx = int(np.argmax(conf)) if len(conf) else -1
        top = None
        if top_idx >= 0:
            top_iou = float(box_iou(boxes[[top_idx]], gt).max()) if len(gt) else 0.0
            top = {
                "conf": float(conf[top_idx]),
                "cls": int(cls[top_idx]),
                "box": [float(x) for x in boxes[top_idx]],
                "iou": top_iou,
            }
        for c in confs:
            keep = conf >= c
            if np.any(keep):
                metrics[str(c)]["any"] += 1
            class0 = keep & (cls == 0)
            if np.any(class0):
                metrics[str(c)]["class0"] += 1
                best_iou = float(box_iou(boxes[class0], gt).max()) if len(gt) else 0.0
                best_conf = float(conf[class0].max())
            else:
                best_iou = 0.0
                best_conf = 0.0
            if best_iou >= 0.5:
                metrics[str(c)]["iou50"] += 1
            metrics[str(c)]["best_ious"].append(best_iou)
            metrics[str(c)]["top_confs"].append(best_conf)
        examples.append({"image": image_path.name, "top": top})

    n = len(image_paths)
    summary = {}
    for key, value in metrics.items():
        summary[key] = {
            "images": n,
            "any_detections": value["any"],
            "class0_detections": value["class0"],
            "iou50_recall": value["iou50"] / n if n else 0.0,
            "mean_best_iou": float(np.mean(value["best_ious"])) if n else 0.0,
            "mean_top_conf": float(np.mean(value["top_confs"])) if n else 0.0,
        }
    return {"metrics": summary, "examples": examples}


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    dataset_dir = out_dir / "tiny_dataset"
    dataset_yaml = create_tiny_dataset(dataset_dir, n=args.samples, size=args.imgsz)

    tf.keras.backend.clear_session()
    fine_model = convert_pt_to_tf(
        args.pt,
        output=out_dir / "converted_nc1.weights.h5",
        imgsz=args.imgsz,
        nc=1,
        verify=False,
    )
    fine_train = train_model(
        fine_model,
        dataset_yaml,
        out_dir / "finetune20_nc1.weights.h5",
        epochs=args.finetune_epochs,
        batch=args.samples,
        lr=args.finetune_lr,
        imgsz=args.imgsz,
        nc=1,
        augment=False,
    )
    fine_eval = evaluate_model(fine_model, dataset_dir, imgsz=args.imgsz)
    fine_result = {"train": fine_train, "eval": fine_eval}
    (out_dir / "finetune20_nc1_result.json").write_text(json.dumps(fine_result, indent=2))

    tf.keras.backend.clear_session()
    set_seed(args.seed + 1)
    scratch_model = build_model("yolo26n.yaml", nc=1, imgsz=args.imgsz)
    scratch_train = train_model(
        scratch_model,
        dataset_yaml,
        out_dir / "scratch50_nc1.weights.h5",
        epochs=args.scratch_epochs,
        batch=args.scratch_batch,
        lr=args.scratch_lr,
        imgsz=args.imgsz,
        nc=1,
        augment=False,
    )
    scratch_eval = evaluate_model(scratch_model, dataset_dir, imgsz=args.imgsz)
    scratch_result = {"train": scratch_train, "eval": scratch_eval}
    (out_dir / "scratch50_nc1_result.json").write_text(json.dumps(scratch_result, indent=2))

    summary = {
        "dataset": str(dataset_dir),
        "dataset_yaml": str(dataset_yaml),
        "finetune": fine_result,
        "scratch": scratch_result,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/verify/aug_audit_overfit")
    parser.add_argument("--pt", default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--scratch-epochs", type=int, default=50)
    parser.add_argument("--scratch-batch", type=int, default=4)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--scratch-lr", type=float, default=5e-3)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
