"""Parity checks for the TensorFlow YOLO26 detection port.

This script intentionally depends on the optional PyTorch/Ultralytics stack and
is meant for local audits against the pinned upstream implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from yolo26_tf.converter import convert_pt_to_tf, parity_check
from yolo26_tf.losses import BboxLoss, TaskAlignedAssigner


def check_forward(weights: Path, imgsz: int, nc: int) -> dict:
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    torch_model = yolo.model.eval()
    tf_model = convert_pt_to_tf(weights, imgsz=imgsz, nc=nc, verify=False)
    max_abs_diff = parity_check(torch_model, tf_model, imgsz=imgsz)
    return {"check": "forward", "imgsz": imgsz, "max_abs_diff": max_abs_diff}


def check_assigner() -> dict:
    import torch
    from ultralytics.utils.tal import TaskAlignedAssigner as TorchTaskAlignedAssigner

    pd_scores = np.array([[[0.9, 0.8], [0.85, 0.7], [0.2, 0.95]]], dtype=np.float32)
    pd_bboxes = np.array([[[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]]], dtype=np.float32)
    anchors = np.array([[5, 5], [6, 6], [25, 25]], dtype=np.float32)
    gt_labels = np.array([[[0], [1]]], dtype=np.int64)
    gt_bboxes = np.array([[[0, 0, 12, 12], [2, 2, 14, 14]]], dtype=np.float32)
    mask_gt = np.array([[[True], [True]]])

    torch_assigner = TorchTaskAlignedAssigner(topk=2, num_classes=2, alpha=0.5, beta=6.0, stride=[8, 16, 32])
    _, torch_boxes, torch_scores, torch_fg, _ = torch_assigner(
        torch.tensor(pd_scores),
        torch.tensor(pd_bboxes),
        torch.tensor(anchors),
        torch.tensor(gt_labels),
        torch.tensor(gt_bboxes),
        torch.tensor(mask_gt),
    )

    tf_assigner = TaskAlignedAssigner(topk=2, num_classes=2, alpha=0.5, beta=6.0, stride=[8, 16, 32])
    tf_boxes, tf_scores, tf_fg, _ = tf_assigner(pd_scores, pd_bboxes, anchors, gt_labels, gt_bboxes, mask_gt)
    return {
        "check": "assigner",
        "fg_match": bool(np.array_equal(tf_fg.numpy(), torch_fg.numpy().astype(bool))),
        "boxes_max_abs_diff": float(np.max(np.abs(tf_boxes.numpy() - torch_boxes.numpy()))),
        "scores_max_abs_diff": float(np.max(np.abs(tf_scores.numpy() - torch_scores.numpy()))),
    }


def check_bbox_loss() -> dict:
    import torch
    from ultralytics.utils.loss import BboxLoss as TorchBboxLoss

    pred_dist = np.array([[[1.0, 1.1, 1.2, 1.3], [0.5, 0.5, 1.0, 1.0]]], dtype=np.float32)
    pred_bboxes = np.array([[[4.0, 4.0, 8.0, 8.0], [10.0, 10.0, 13.0, 13.0]]], dtype=np.float32)
    anchor_points = np.array([[6.0, 6.0], [11.0, 11.0]], dtype=np.float32)
    target_bboxes_pixel = np.array([[[33.6, 32.8, 64.8, 65.6], [80.0, 80.0, 100.0, 102.4]]], dtype=np.float32)
    stride = np.array([[8.0], [8.0]], dtype=np.float32)
    target_bboxes_grid = target_bboxes_pixel / stride[None]
    target_scores = np.array([[[0.7, 0.0], [0.0, 0.5]]], dtype=np.float32)
    fg_mask = np.array([[True, True]])
    target_scores_sum = np.array(1.2, dtype=np.float32)
    imgsz = np.array([64.0, 64.0], dtype=np.float32)

    torch_loss = TorchBboxLoss(reg_max=1)
    torch_box, torch_dfl = torch_loss(
        torch.tensor(pred_dist),
        torch.tensor(pred_bboxes),
        torch.tensor(anchor_points),
        torch.tensor(target_bboxes_grid),
        torch.tensor(target_scores),
        torch.tensor(target_scores_sum),
        torch.tensor(fg_mask),
        torch.tensor(imgsz),
        torch.tensor(stride),
    )

    tf_box, tf_dfl = BboxLoss(reg_max=1)(
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes_pixel,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,
        stride,
    )
    return {
        "check": "bbox_loss",
        "box_abs_diff": float(abs(float(tf_box.numpy()) - float(torch_box.numpy()))),
        "dfl_abs_diff": float(abs(float(tf_dfl.numpy()) - float(torch_dfl.numpy()))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyTorch-vs-TensorFlow YOLO26 parity checks.")
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--assigner", action="store_true")
    parser.add_argument("--bbox-loss", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_forward = args.forward or not (args.forward or args.assigner or args.bbox_loss)
    run_assigner = args.assigner or not (args.forward or args.assigner or args.bbox_loss)
    run_bbox_loss = args.bbox_loss or not (args.forward or args.assigner or args.bbox_loss)
    results = []
    if run_forward:
        results.append(check_forward(Path(args.weights), args.imgsz, args.nc))
    if run_assigner:
        results.append(check_assigner())
    if run_bbox_loss:
        results.append(check_bbox_loss())
    text = json.dumps({"results": results}, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
