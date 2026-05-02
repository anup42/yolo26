"""Validation helpers for YOLO26 TensorFlow detection."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .coco import COCO80_TO_COCO91, coco_image_id_from_path, evaluate_coco_predictions
from .data import YOLODataset, load_data_yaml
from .metrics import ConfusionMatrix, DetMetrics, targets_from_batch
from .ops import nms_numpy, scale_boxes_np
from .tf_import import require_tf

tf = require_tf()


def validate_detection_model(
    model,
    data: str | Path | dict,
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
    rect: bool = True,
    use_coco: bool = False,
    save_json: bool = False,
    save_txt: bool = False,
    save_conf: bool = False,
    single_cls: bool = False,
    agnostic_nms: bool = False,
    multi_label: bool = False,
    half: bool = False,
    project: str | Path = "runs/detect",
    name: str = "val",
    verbose: bool = True,
) -> dict[str, Any]:
    data_dict = load_data_yaml(data)
    ds = YOLODataset(data_dict, "val", imgsz, batch, augment=False, shuffle=False, rect=rect)
    out_dir = Path(project) / name
    if save_json or save_txt:
        out_dir.mkdir(parents=True, exist_ok=True)
    label_dir = out_dir / "labels"
    if save_txt:
        label_dir.mkdir(parents=True, exist_ok=True)
    preds_all, targets_all, coco_rows, image_ids = [], [], [], []
    det_metrics = DetMetrics(names=data_dict.get("names", {}))
    confusion = ConfusionMatrix(data_dict["nc"], conf=conf, iou_thres=iou)
    seen = 0
    t_infer = 0.0
    t_post = 0.0
    for b in ds:
        t0 = time.perf_counter()
        images = tf.convert_to_tensor(b["img"], tf.float16 if half else tf.float32)
        raw = model(images, training=False).numpy()
        t_infer += time.perf_counter() - t0
        t1 = time.perf_counter()
        input_shape = tuple(int(x) for x in b["img"].shape[1:3])
        batch_targets = targets_from_batch(b, input_shape)
        if single_cls:
            batch_targets = [(np.zeros_like(cls), boxes) for cls, boxes in batch_targets]
        for si, (pred, im_file, shape, ratio, pad) in enumerate(zip(raw, b["im_file"], b["ori_shape"], b["ratio"], b["pad"])):
            det = prediction_to_detections(pred, conf=conf, iou=iou, max_det=max_det, agnostic=agnostic_nms, multi_label=multi_label)
            if single_cls and len(det):
                det[:, 5] = 0
            preds_all.append(det.astype(np.float32))
            gt_cls, gt_boxes = batch_targets[si]
            if single_cls and len(gt_cls):
                gt_cls = np.zeros_like(gt_cls)
            confusion.process_batch(det, gt_cls, gt_boxes)
            seen += 1
            if use_coco:
                image_id = coco_image_id_from_path(im_file)
                image_ids.append(image_id)
                coco_rows.extend(detections_to_coco_rows(det, image_id, shape, input_shape, ratio, pad))
            if save_txt:
                save_one_txt(det, shape, input_shape, ratio, pad, label_dir / f"{Path(im_file).stem}.txt", save_conf)
        targets_all.extend(batch_targets)
        t_post += time.perf_counter() - t1
    if use_coco:
        ann = data_dict.get("val_annotations") or data_dict.get("annotations")
        if ann is None:
            raise FileNotFoundError("COCO validation requested, but data YAML has no val_annotations or annotations entry.")
        if save_json:
            (out_dir / "predictions.json").write_text(json.dumps(coco_rows), encoding="utf-8")
        metrics = evaluate_coco_predictions(ann, coco_rows, image_ids=image_ids)
    else:
        det_metrics.update_stats(preds_all, targets_all)
        metrics = det_metrics.process()
    metrics = {
        **metrics,
        "images": int(seen),
        "predictions": int(sum(len(x) for x in preds_all)),
        "confusion_matrix": confusion.matrix.tolist(),
        "speed/inference_ms_per_image": float(t_infer * 1000 / max(seen, 1)),
        "speed/postprocess_ms_per_image": float(t_post * 1000 / max(seen, 1)),
    }
    if verbose:
        print(metrics)
    return metrics


def prediction_to_detections(
    pred: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
    agnostic: bool = False,
    multi_label: bool = False,
) -> np.ndarray:
    if pred.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    if pred.shape[-1] == 6:
        det = pred[pred[:, 4] >= conf]
        if len(det) > max_det:
            det = det[np.argsort(-det[:, 4])[:max_det]]
        return det.astype(np.float32)
    boxes, scores = pred[:, :4], pred[:, 4:]
    if multi_label:
        rows, cols = np.nonzero(scores > conf)
        if len(rows):
            det = np.concatenate([boxes[rows], scores[rows, cols, None], cols[:, None].astype(np.float32)], axis=-1)
        else:
            det = np.zeros((0, 6), dtype=np.float32)
    else:
        cls = scores.argmax(axis=-1)
        score = scores.max(axis=-1)
        det = np.concatenate([boxes, score[:, None], cls[:, None]], axis=-1)
    return nms_numpy(det, conf=conf, iou=iou, max_det=max_det, agnostic=agnostic)


class DetectionValidator:
    """Ultralytics-style validator wrapper around the TensorFlow validation path."""

    def __init__(self, model=None, data: str | Path | dict | None = None, **kwargs):
        self.model = model
        self.data = data
        self.args = kwargs
        self.metrics: DetMetrics | None = None
        self.results_dict: dict[str, Any] = {}
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

    def preprocess(self, batch: dict) -> dict:
        return batch

    def postprocess(self, preds: np.ndarray) -> list[np.ndarray]:
        return [prediction_to_detections(p, **{k: self.args[k] for k in ("conf", "iou", "max_det") if k in self.args}) for p in preds]

    def __call__(self, model=None, data=None, **kwargs) -> dict[str, Any]:
        model = model or self.model
        data = data or self.data
        if model is None or data is None:
            raise ValueError("DetectionValidator requires both model and data.")
        args = {**self.args, **kwargs}
        self.results_dict = validate_detection_model(model, data, **args)
        self.metrics = DetMetrics(names=load_data_yaml(data).get("names", {}))
        self.metrics.results_dict = self.results_dict
        self.speed = {
            "preprocess": float(self.results_dict.get("speed/preprocess_ms_per_image", 0.0)),
            "inference": float(self.results_dict.get("speed/inference_ms_per_image", 0.0)),
            "loss": 0.0,
            "postprocess": float(self.results_dict.get("speed/postprocess_ms_per_image", 0.0)),
        }
        return self.results_dict


def detections_to_coco_rows(det: np.ndarray, image_id: int, ori_shape, input_shape, ratio, pad) -> list[dict]:
    if len(det) == 0:
        return []
    boxes = scale_boxes_np(det[:, :4], input_shape, ori_shape, ratio_pad=((ratio[0], ratio[1]), pad))
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ori_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, ori_shape[0])
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    valid = (wh[:, 0] > 0) & (wh[:, 1] > 0)
    rows = []
    for box, wh_i, row in zip(boxes[valid], wh[valid], det[valid]):
        cls_idx = int(row[5])
        if cls_idx < 0 or cls_idx >= len(COCO80_TO_COCO91):
            continue
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": int(COCO80_TO_COCO91[cls_idx]),
                "bbox": [float(box[0]), float(box[1]), float(wh_i[0]), float(wh_i[1])],
                "score": float(row[4]),
            }
        )
    return rows


def save_one_txt(det: np.ndarray, ori_shape, input_shape, ratio, pad, file: Path, save_conf: bool):
    if len(det) == 0:
        file.write_text("", encoding="utf-8")
        return
    boxes = scale_boxes_np(det[:, :4], input_shape, ori_shape, ratio_pad=((ratio[0], ratio[1]), pad))
    h, w = ori_shape
    xywh = np.zeros_like(boxes)
    xywh[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / w
    xywh[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / h
    xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]) / w
    xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]) / h
    lines = []
    for row, box in zip(det, xywh):
        values = [int(row[5]), *[float(x) for x in box]]
        if save_conf:
            values.append(float(row[4]))
        lines.append(" ".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in values))
    file.write_text("\n".join(lines) + "\n", encoding="utf-8")
