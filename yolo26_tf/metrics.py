"""Ultralytics-style detection metrics for YOLO26 TensorFlow validation."""

from __future__ import annotations

import numpy as np

from .ops import pairwise_iou_np, xywh2xyxy_np


IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10, dtype=np.float32)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP with the same interpolation idea used by Ultralytics/COCO metrics."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def ap_per_class(preds: list[np.ndarray], targets: list[tuple[np.ndarray, np.ndarray]], iou_thres=None) -> dict:
    """Compute precision, recall, mAP50 and mAP50-95 for detection validation."""
    iouv = np.asarray(iou_thres if iou_thres is not None else IOU_THRESHOLDS, dtype=np.float32)
    stats = []
    target_cls_all = []
    for img_i, (pred, (gt_cls, gt_boxes)) in enumerate(zip(preds, targets)):
        gt_cls = gt_cls.astype(np.int64)
        target_cls_all.extend(gt_cls.tolist())
        correct = np.zeros((len(pred), len(iouv)), dtype=bool)
        if len(pred) and len(gt_cls):
            for cls in np.unique(np.concatenate([pred[:, 5].astype(np.int64), gt_cls])):
                pi = np.where(pred[:, 5].astype(np.int64) == cls)[0]
                ti = np.where(gt_cls == cls)[0]
                if len(pi) == 0 or len(ti) == 0:
                    continue
                ious = pairwise_iou_np(pred[pi, :4], gt_boxes[ti])
                matches = np.argwhere(ious >= iouv[0])
                if len(matches):
                    match_rows = []
                    for p_rel, t_rel in matches:
                        match_rows.append((float(ious[p_rel, t_rel]), int(pi[p_rel]), int(ti[t_rel])))
                    match_rows.sort(reverse=True)
                    used_p, used_t = set(), set()
                    for iou, p_abs, t_abs in match_rows:
                        if p_abs in used_p or t_abs in used_t:
                            continue
                        used_p.add(p_abs)
                        used_t.add(t_abs)
                        correct[p_abs] = iou >= iouv
        if len(pred):
            stats.append((correct, pred[:, 4].astype(np.float32), pred[:, 5].astype(np.int64), gt_cls))

    total_gt = len(target_cls_all)
    if not stats or total_gt == 0:
        return empty_metrics(total_gt)

    correct, conf, pred_cls, _ = (np.concatenate(x, axis=0) for x in zip(*stats))
    order = np.argsort(-conf)
    correct, conf, pred_cls = correct[order], conf[order], pred_cls[order]
    unique_classes = np.unique(np.asarray(target_cls_all, dtype=np.int64))
    ap = np.zeros((len(unique_classes), len(iouv)), dtype=np.float32)
    precision_cls = np.zeros(len(unique_classes), dtype=np.float32)
    recall_cls = np.zeros(len(unique_classes), dtype=np.float32)
    nt_per_class = np.array([(np.asarray(target_cls_all) == c).sum() for c in unique_classes], dtype=np.float32)

    for ci, c in enumerate(unique_classes):
        idx = pred_cls == c
        n_l = nt_per_class[ci]
        n_p = idx.sum()
        if n_p == 0 or n_l == 0:
            continue
        tpc = correct[idx].cumsum(0)
        fpc = (1 - correct[idx]).cumsum(0)
        recall = tpc / (n_l + 1e-16)
        precision = tpc / (tpc + fpc + 1e-16)
        precision_cls[ci] = precision[-1, 0]
        recall_cls[ci] = recall[-1, 0]
        for j in range(len(iouv)):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    mp = float(precision_cls.mean()) if len(precision_cls) else 0.0
    mr = float(recall_cls.mean()) if len(recall_cls) else 0.0
    map50 = float(ap[:, 0].mean()) if ap.size else 0.0
    map5095 = float(ap.mean()) if ap.size else 0.0
    fitness = 0.1 * map50 + 0.9 * map5095
    return {
        "metrics/precision(B)": mp,
        "metrics/recall(B)": mr,
        "metrics/mAP50(B)": map50,
        "metrics/mAP50-95(B)": map5095,
        "fitness": float(fitness),
        "nt_per_class": {int(c): int(n) for c, n in zip(unique_classes, nt_per_class)},
        "ap_class_index": [int(x) for x in unique_classes],
    }


def process_batch(detections: np.ndarray, gt_cls: np.ndarray, gt_boxes: np.ndarray, iouv=None) -> np.ndarray:
    """Return the Ultralytics-style correct-prediction matrix for one image."""
    iouv = np.asarray(iouv if iouv is not None else IOU_THRESHOLDS, dtype=np.float32)
    correct = np.zeros((len(detections), len(iouv)), dtype=bool)
    if len(detections) == 0 or len(gt_cls) == 0:
        return correct
    ious = pairwise_iou_np(gt_boxes, detections[:, :4])
    matches = np.argwhere(ious >= iouv[0])
    if len(matches) == 0:
        return correct
    rows = []
    for gt_i, pred_i in matches:
        if int(gt_cls[gt_i]) == int(detections[pred_i, 5]):
            rows.append((float(ious[gt_i, pred_i]), int(gt_i), int(pred_i)))
    rows.sort(reverse=True)
    used_gt, used_pred = set(), set()
    for iou, gt_i, pred_i in rows:
        if gt_i in used_gt or pred_i in used_pred:
            continue
        used_gt.add(gt_i)
        used_pred.add(pred_i)
        correct[pred_i] = iou >= iouv
    return correct


def ap_per_class_from_stats(stats: dict[str, list[np.ndarray]], iou_thres=None) -> dict:
    """Compute metrics from accumulated Ultralytics-style stats arrays."""
    if not stats.get("target_cls"):
        return empty_metrics(0)
    target_cls = np.concatenate(stats["target_cls"], axis=0).astype(np.int64)
    if len(target_cls) == 0:
        return empty_metrics(0)
    unique_classes = np.unique(target_cls)
    nt_per_class = np.array([(target_cls == c).sum() for c in unique_classes], dtype=np.float32)
    nt_per_image = {}
    if stats.get("target_img"):
        for c in unique_classes:
            nt_per_image[int(c)] = int(sum((np.asarray(x) == c).sum() for x in stats["target_img"]))
    if not stats.get("conf") or sum(len(x) for x in stats["conf"]) == 0:
        result = empty_metrics(len(target_cls))
        result.update(
            {
                "nt_per_class": {int(c): int(n) for c, n in zip(unique_classes, nt_per_class)},
                "nt_per_image": nt_per_image,
                "ap_class_index": [int(x) for x in unique_classes],
            }
        )
        return result
    correct = np.concatenate(stats["tp"], axis=0)
    conf = np.concatenate(stats["conf"], axis=0).astype(np.float32)
    pred_cls = np.concatenate(stats["pred_cls"], axis=0).astype(np.int64)
    iouv = np.asarray(iou_thres if iou_thres is not None else IOU_THRESHOLDS, dtype=np.float32)
    order = np.argsort(-conf)
    correct, pred_cls = correct[order], pred_cls[order]
    ap = np.zeros((len(unique_classes), len(iouv)), dtype=np.float32)
    precision_cls = np.zeros(len(unique_classes), dtype=np.float32)
    recall_cls = np.zeros(len(unique_classes), dtype=np.float32)
    for ci, c in enumerate(unique_classes):
        idx = pred_cls == c
        n_l = nt_per_class[ci]
        n_p = idx.sum()
        if n_p == 0 or n_l == 0:
            continue
        tpc = correct[idx].cumsum(0)
        fpc = (1 - correct[idx]).cumsum(0)
        recall = tpc / (n_l + 1e-16)
        precision = tpc / (tpc + fpc + 1e-16)
        precision_cls[ci] = precision[-1, 0]
        recall_cls[ci] = recall[-1, 0]
        for j in range(len(iouv)):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    map50 = float(ap[:, 0].mean()) if ap.size else 0.0
    map5095 = float(ap.mean()) if ap.size else 0.0
    fitness = 0.1 * map50 + 0.9 * map5095
    return {
        "metrics/precision(B)": float(precision_cls.mean()) if len(precision_cls) else 0.0,
        "metrics/recall(B)": float(recall_cls.mean()) if len(recall_cls) else 0.0,
        "metrics/mAP50(B)": map50,
        "metrics/mAP50-95(B)": map5095,
        "fitness": float(fitness),
        "nt_per_class": {int(c): int(n) for c, n in zip(unique_classes, nt_per_class)},
        "nt_per_image": nt_per_image,
        "ap_class_index": [int(x) for x in unique_classes],
    }


def empty_metrics(total_gt: int = 0) -> dict:
    return {
        "metrics/precision(B)": 0.0,
        "metrics/recall(B)": 0.0,
        "metrics/mAP50(B)": 0.0,
        "metrics/mAP50-95(B)": 0.0,
        "fitness": 0.0,
        "nt_per_class": {},
        "ap_class_index": [],
        "targets": int(total_gt),
    }


class ConfusionMatrix:
    """Small detection confusion matrix compatible with Ultralytics-style reporting."""

    def __init__(self, nc: int, conf: float = 0.25, iou_thres: float = 0.45):
        self.nc = int(nc)
        self.conf = float(conf)
        self.iou_thres = float(iou_thres)
        self.matrix = np.zeros((self.nc + 1, self.nc + 1), dtype=np.int64)

    def process_batch(self, detections: np.ndarray, gt_cls: np.ndarray, gt_boxes: np.ndarray):
        detections = detections[detections[:, 4] >= self.conf] if len(detections) else detections
        if len(gt_cls) == 0:
            for dc in detections[:, 5].astype(int):
                self.matrix[self.nc, dc] += 1
            return
        if len(detections) == 0:
            for gc in gt_cls.astype(int):
                self.matrix[gc, self.nc] += 1
            return
        ious = pairwise_iou_np(gt_boxes, detections[:, :4])
        matches = np.argwhere(ious > self.iou_thres)
        used_gt, used_det = set(), set()
        if len(matches):
            rows = sorted([(float(ious[g, d]), int(g), int(d)) for g, d in matches], reverse=True)
            for _, g, d in rows:
                if g in used_gt or d in used_det:
                    continue
                used_gt.add(g)
                used_det.add(d)
                self.matrix[int(gt_cls[g]), int(detections[d, 5])] += 1
        for g, gc in enumerate(gt_cls.astype(int)):
            if g not in used_gt:
                self.matrix[gc, self.nc] += 1
        for d, dc in enumerate(detections[:, 5].astype(int)):
            if d not in used_det:
                self.matrix[self.nc, dc] += 1


def targets_from_batch(batch: dict, img_shape: int | tuple[int, int]) -> list[tuple[np.ndarray, np.ndarray]]:
    if isinstance(img_shape, int):
        h = w = int(img_shape)
    else:
        h, w = int(img_shape[0]), int(img_shape[1])
    out = []
    for boxes, cls, mask in zip(batch["bboxes"], batch["cls"], batch["mask"]):
        boxes = boxes[mask]
        cls = cls[mask]
        if len(boxes):
            xyxy = xywh2xyxy_np(boxes.copy())
            xyxy[:, [0, 2]] *= w
            xyxy[:, [1, 3]] *= h
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
        out.append((cls.astype(np.int64), xyxy.astype(np.float32)))
    return out


class DetMetrics:
    """Ultralytics-like detection metric accumulator.

    The public keys intentionally match ``DetectionValidator`` result names so
    training logs and benchmark JSONs can be compared directly against
    Ultralytics output. Curve arrays are kept lightweight but available for
    downstream plotting/tests.
    """

    keys = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")

    def __init__(self, names: dict[int, str] | None = None):
        self.names = names or {}
        self.nt_per_class: dict[int, int] = {}
        self.ap_class_index: list[int] = []
        self.results_dict = empty_metrics()
        self.curves_results: dict[str, list] = {}
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.stats: dict[str, list[np.ndarray]] = {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": [], "im_name": []}
        self.clear_stats()

    def clear_stats(self):
        self.preds: list[np.ndarray] = []
        self.targets: list[tuple[np.ndarray, np.ndarray]] = []
        self.stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": [], "im_name": []}

    def update_stats(self, preds, targets: list[tuple[np.ndarray, np.ndarray]] | None = None):
        if isinstance(preds, dict):
            for key in self.stats:
                value = preds.get(key)
                if value is not None:
                    self.stats[key].append(np.asarray(value))
            return
        self.preds.extend(preds)
        self.targets.extend(targets or [])

    def process(self) -> dict:
        self.results_dict = ap_per_class_from_stats(self.stats) if self.stats["target_cls"] else ap_per_class(self.preds, self.targets)
        self.nt_per_class = self.results_dict.get("nt_per_class", {})
        self.nt_per_image = self.results_dict.get("nt_per_image", {})
        self.ap_class_index = self.results_dict.get("ap_class_index", [])
        self.curves_results = {
            "names": [self.names.get(i, str(i)) for i in self.ap_class_index],
            "ap_class_index": list(self.ap_class_index),
        }
        return self.results_dict

    @property
    def fitness(self) -> float:
        return float(self.results_dict.get("fitness", self.results_dict.get("metrics/mAP50-95(B)", 0.0)))

    def mean_results(self) -> list[float]:
        return [float(self.results_dict.get(k, 0.0)) for k in self.keys]

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        cls_id = int(self.ap_class_index[i])
        if self.nt_per_class.get(cls_id, 0) == 0:
            return 0.0, 0.0, 0.0, 0.0
        return tuple(self.mean_results())

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict]:
        rows = []
        for cls_id in self.ap_class_index:
            row = {"Class": self.names.get(int(cls_id), str(cls_id)), "Images": len(self.targets), "Instances": self.nt_per_class.get(int(cls_id), 0)}
            row.update({k: round(float(self.results_dict.get(k, 0.0)), decimals) for k in self.keys})
            rows.append(row)
        return rows
