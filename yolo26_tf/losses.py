"""YOLO26 detection losses rewritten for TensorFlow training.

The loss keeps the public Ultralytics detection behavior that matters for YOLO26:
end-to-end one-to-many/one-to-one branches, TaskAligned assignment, CIoU box
loss, BCE class loss, and the reg_max=1 normalized L1 distance term.
"""

from __future__ import annotations

from .ops import bbox2dist, bbox_iou, dist2bbox, make_anchors
from .tf_import import require_tf

tf = require_tf()


class TaskAlignedAssigner:
    """TensorFlow TaskAlignedAssigner for YOLO detection training."""

    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        stride: list[float] | None = None,
        topk2: int | None = None,
    ):
        self.topk = int(topk)
        self.topk2 = int(topk2 or topk)
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.stride = [float(x) for x in (stride or [8.0, 16.0, 32.0])]
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = 1e-9

    def __call__(self, pd_scores, pd_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        pd_scores = tf.stop_gradient(tf.cast(pd_scores, tf.float32))
        pd_bboxes = tf.stop_gradient(tf.cast(pd_bboxes, tf.float32))
        anchor_points = tf.stop_gradient(tf.cast(anchor_points, tf.float32))
        gt_bboxes = tf.cast(gt_bboxes, tf.float32)
        gt_labels = tf.convert_to_tensor(gt_labels)
        mask_gt = tf.convert_to_tensor(mask_gt)
        gt_labels = tf.cast(tf.squeeze(gt_labels, axis=-1) if gt_labels.shape.rank == 3 else gt_labels, tf.int32)
        mask_gt = tf.cast(tf.squeeze(mask_gt, axis=-1) if mask_gt.shape.rank == 3 else mask_gt, tf.bool)
        mask_gt = mask_gt & (tf.reduce_sum(gt_bboxes, axis=-1) > 0)

        bsz = tf.shape(pd_scores)[0]
        anchors_n = tf.shape(pd_scores)[1]
        max_gt = tf.shape(gt_bboxes)[1]
        has_gt = tf.reduce_any(mask_gt)

        def empty_result():
            target_bboxes = tf.zeros([bsz, anchors_n, 4], dtype=tf.float32)
            target_scores = tf.zeros([bsz, anchors_n, self.num_classes], dtype=tf.float32)
            fg_mask = tf.zeros([bsz, anchors_n], dtype=tf.bool)
            target_gt_idx = tf.zeros([bsz, anchors_n], dtype=tf.int64)
            return target_bboxes, target_scores, fg_mask, target_gt_idx

        def assign_result():
            return self._assign_non_empty(pd_scores, pd_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt)

        return tf.cond(has_gt, assign_result, empty_result)

    def _assign_non_empty(self, pd_scores, pd_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        candidate_gt_bboxes = self._candidate_boxes(gt_bboxes, mask_gt)
        ap = anchor_points[None, None, :, :]
        gtb = candidate_gt_bboxes[:, :, None, :]
        in_gts = (
            (ap[..., 0] > gtb[..., 0])
            & (ap[..., 1] > gtb[..., 1])
            & (ap[..., 0] < gtb[..., 2])
            & (ap[..., 1] < gtb[..., 3])
        )
        in_gts = in_gts & mask_gt[:, :, None]

        ious = tf.maximum(pairwise_iou_tf(gt_bboxes, pd_bboxes, ciou=True), 0.0)
        labels = tf.clip_by_value(gt_labels, 0, self.num_classes - 1)
        label_oh = tf.one_hot(labels, self.num_classes, dtype=pd_scores.dtype)
        cls_scores = tf.reduce_sum(pd_scores[:, None, :, :] * label_oh[:, :, None, :], axis=-1)
        metric = tf.pow(tf.maximum(cls_scores, 0.0), self.alpha) * tf.pow(tf.maximum(ious, 0.0), self.beta)
        metric = tf.where(in_gts, metric, tf.zeros_like(metric))

        topk_mask = tf.tile(mask_gt[:, :, None], [1, 1, self.topk])
        mask_topk = self._select_topk_candidates(metric, topk_mask)
        mask_pos = mask_topk * tf.cast(in_gts, tf.float32) * tf.cast(mask_gt[:, :, None], tf.float32)
        target_gt_idx, fg_mask_f, mask_pos = self._select_highest_overlaps(mask_pos, ious, metric)

        target_bboxes = tf.gather(gt_bboxes, target_gt_idx, batch_dims=1)
        target_labels = tf.gather(labels, target_gt_idx, batch_dims=1)
        fg_mask = fg_mask_f > 0

        target_scores = tf.one_hot(target_labels, self.num_classes, dtype=tf.float32)
        target_scores = tf.where(fg_mask[:, :, None], target_scores, tf.zeros_like(target_scores))

        align_metric = metric * mask_pos
        pos_align_metrics = tf.reduce_max(align_metric, axis=-1, keepdims=True)
        pos_overlaps = tf.reduce_max(ious * mask_pos, axis=-1, keepdims=True)
        norm_align_metric = tf.reduce_max(align_metric * pos_overlaps / (pos_align_metrics + self.eps), axis=1)
        target_scores = target_scores * norm_align_metric[:, :, None]
        target_gt_idx = tf.cast(target_gt_idx, tf.int64)
        return target_bboxes, target_scores, fg_mask, target_gt_idx

    def _candidate_boxes(self, gt_bboxes, mask_gt):
        """Match Ultralytics TAL small-box expansion before anchor-in-GT checks."""
        x1, y1, x2, y2 = tf.split(gt_bboxes, 4, axis=-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        wh = tf.concat([x2 - x1, y2 - y1], axis=-1)
        min_stride = tf.cast(self.stride[0], gt_bboxes.dtype)
        stride_val = tf.cast(self.stride_val, gt_bboxes.dtype)
        wh = tf.where((wh < min_stride) & mask_gt[..., None], stride_val, wh)
        half = wh * 0.5
        return tf.concat([cx - half[..., 0:1], cy - half[..., 1:2], cx + half[..., 0:1], cy + half[..., 1:2]], axis=-1)

    def _select_topk_candidates(self, metrics, topk_mask=None):
        """Match Ultralytics top-k scatter-add selection and duplicate suppression."""
        k = tf.minimum(tf.cast(self.topk, tf.int32), tf.shape(metrics)[-1])
        topk_metrics, topk_idxs = tf.math.top_k(metrics, k=k, sorted=True)
        if topk_mask is None:
            topk_mask = tf.tile(tf.reduce_max(topk_metrics, axis=-1, keepdims=True) > self.eps, [1, 1, k])
        else:
            topk_mask = tf.cast(topk_mask[..., :k], tf.bool)
        topk_idxs = tf.where(topk_mask, topk_idxs, tf.zeros_like(topk_idxs))

        b = tf.shape(metrics)[0]
        m = tf.shape(metrics)[1]
        a = tf.shape(metrics)[2]
        bm = b * m
        flat_idx = tf.reshape(topk_idxs, [bm, k])
        row = tf.tile(tf.range(bm, dtype=tf.int32)[:, None], [1, k])
        scatter_idx = tf.stack([tf.reshape(row, [-1]), tf.reshape(flat_idx, [-1])], axis=-1)
        updates = tf.ones([tf.shape(scatter_idx)[0]], dtype=tf.int32)
        counts = tf.scatter_nd(scatter_idx, updates, [bm, a])
        counts = tf.where(counts > 1, tf.zeros_like(counts), counts)
        return tf.reshape(tf.cast(counts, metrics.dtype), [b, m, a])

    def _select_highest_overlaps(self, mask_pos, overlaps, align_metric):
        """Match Ultralytics multi-GT conflict resolution and optional topk2 pruning."""
        fg_mask = tf.reduce_sum(mask_pos, axis=1)

        def resolve_multi():
            n_max_boxes = tf.shape(mask_pos)[1]
            mask_multi = tf.tile((fg_mask[:, None, :] > 1), [1, n_max_boxes, 1])
            max_overlaps_idx = tf.argmax(overlaps, axis=1, output_type=tf.int32)
            is_max = tf.one_hot(max_overlaps_idx, n_max_boxes, dtype=mask_pos.dtype)
            is_max = tf.transpose(is_max, [0, 2, 1])
            resolved = tf.where(mask_multi, is_max, mask_pos)
            return resolved, tf.reduce_sum(resolved, axis=1)

        mask_pos, fg_mask = tf.cond(tf.reduce_max(fg_mask) > 1, resolve_multi, lambda: (mask_pos, fg_mask))

        if self.topk2 != self.topk:
            k2 = tf.minimum(tf.cast(self.topk2, tf.int32), tf.shape(mask_pos)[-1])
            topk_idx = tf.math.top_k(align_metric * mask_pos, k=k2, sorted=True).indices
            b = tf.shape(mask_pos)[0]
            m = tf.shape(mask_pos)[1]
            a = tf.shape(mask_pos)[2]
            bm = b * m
            flat_idx = tf.reshape(topk_idx, [bm, k2])
            row = tf.tile(tf.range(bm, dtype=tf.int32)[:, None], [1, k2])
            scatter_idx = tf.stack([tf.reshape(row, [-1]), tf.reshape(flat_idx, [-1])], axis=-1)
            updates = tf.ones([tf.shape(scatter_idx)[0]], dtype=mask_pos.dtype)
            topk_mask = tf.reshape(tf.scatter_nd(scatter_idx, updates, [bm, a]), [b, m, a])
            mask_pos = mask_pos * topk_mask
            fg_mask = tf.reduce_sum(mask_pos, axis=1)

        target_gt_idx = tf.argmax(mask_pos, axis=1, output_type=tf.int32)
        return target_gt_idx, fg_mask, mask_pos


def pairwise_iou_tf(boxes1, boxes2, eps: float = 1e-7, ciou: bool = False):
    """Pairwise IoU/CIoU for boxes shaped [B, N, 4] and [B, A, 4]."""
    b1 = boxes1[:, :, None, :]
    b2 = boxes2[:, None, :, :]
    inter_w = tf.maximum(tf.minimum(b1[..., 2], b2[..., 2]) - tf.maximum(b1[..., 0], b2[..., 0]), 0.0)
    inter_h = tf.maximum(tf.minimum(b1[..., 3], b2[..., 3]) - tf.maximum(b1[..., 1], b2[..., 1]), 0.0)
    inter = inter_w * inter_h
    w1 = tf.maximum(b1[..., 2] - b1[..., 0], 0.0)
    h1 = tf.maximum(b1[..., 3] - b1[..., 1], 0.0)
    w2 = tf.maximum(b2[..., 2] - b2[..., 0], 0.0)
    h2 = tf.maximum(b2[..., 3] - b2[..., 1], 0.0)
    iou = inter / (w1 * h1 + w2 * h2 - inter + eps)
    if not ciou:
        return iou
    cw = tf.maximum(b1[..., 2], b2[..., 2]) - tf.minimum(b1[..., 0], b2[..., 0])
    ch = tf.maximum(b1[..., 3], b2[..., 3]) - tf.minimum(b1[..., 1], b2[..., 1])
    c2 = cw * cw + ch * ch + eps
    rho2 = ((b2[..., 0] + b2[..., 2] - b1[..., 0] - b1[..., 2]) ** 2 + (b2[..., 1] + b2[..., 3] - b1[..., 1] - b1[..., 3]) ** 2) / 4
    v = (4 / 3.141592653589793**2) * tf.square(tf.atan(w2 / (h2 + eps)) - tf.atan(w1 / (h1 + eps)))
    alpha = v / (v - iou + (1.0 + eps))
    return iou - (rho2 / c2 + v * alpha)


class BboxLoss:
    def __init__(self, reg_max: int = 1):
        self.reg_max = int(reg_max)

    def __call__(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor):
        weight = tf.reduce_sum(target_scores, axis=-1)
        fg = tf.where(fg_mask)
        return tf.cond(
            tf.shape(fg)[0] > 0,
            lambda: self._loss_non_empty(pred_dist, pred_bboxes, anchor_points, target_bboxes, weight, target_scores_sum, fg, imgsz, stride_tensor),
            lambda: (tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32)),
        )

    def _loss_non_empty(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, weight, target_scores_sum, fg, imgsz, stride_tensor):
        pd = tf.gather_nd(pred_bboxes, fg)
        target_bboxes_grid = target_bboxes / stride_tensor[None, :, :]
        tb = tf.gather_nd(target_bboxes_grid, fg)
        w = tf.gather_nd(weight, fg)[:, None]
        ciou = bbox_iou(pd, tb, ciou=True)
        loss_iou = tf.reduce_sum((1.0 - ciou) * w) / target_scores_sum
        target_ltrb = bbox2dist(anchor_points[None, :, :], target_bboxes_grid)
        pdist = pred_dist
        # YOLO26 has reg_max=1; upstream uses normalized L1 distance when DFL is disabled.
        target_ltrb = target_ltrb * stride_tensor[None, :, :]
        pdist = pdist * stride_tensor[None, :, :]
        norm = tf.reshape(tf.cast([imgsz[1], imgsz[0], imgsz[1], imgsz[0]], pdist.dtype), [1, 1, 4])
        target_ltrb = target_ltrb / norm
        pdist = pdist / norm
        loss_l1 = tf.reduce_mean(tf.abs(tf.gather_nd(pdist, fg) - tf.gather_nd(target_ltrb, fg)), axis=-1, keepdims=True)
        loss_dfl = tf.reduce_sum(loss_l1 * w) / target_scores_sum
        return tf.cast(loss_iou, tf.float32), tf.cast(loss_dfl, tf.float32)


class DetectionLoss:
    """YOLO detection loss for one branch."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, hyp: dict | None = None):
        self.nc = model.nc
        self.stride = list(model.stride)
        self.reg_max = int(model.detect_layer.reg_max)
        self.assigner = TaskAlignedAssigner(tal_topk, self.nc, alpha=0.5, beta=6.0, stride=self.stride, topk2=tal_topk2)
        self.bbox_loss = BboxLoss(self.reg_max)
        self.hyp = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 100}
        if hyp:
            self.hyp.update(hyp)
        class_weights = self.hyp.get("class_weights")
        self.class_weights = None if class_weights is None else tf.reshape(tf.cast(class_weights, tf.float32), [1, 1, -1])

    def preprocess(self, targets, batch_size, scale_tensor):
        """Mirror Ultralytics target preprocessing from flat batch_idx/cls/xywh rows."""
        targets = tf.cast(targets, tf.float32)
        batch_size = tf.cast(batch_size, tf.int32)

        def no_targets():
            return tf.zeros([batch_size, 0, 5], dtype=tf.float32)

        def with_targets():
            batch_idx = tf.cast(targets[:, 0], tf.int32)
            counts = tf.math.bincount(batch_idx, minlength=batch_size, maxlength=batch_size, dtype=tf.int32)
            max_count = tf.reduce_max(counts)
            offsets = tf.concat([[0], tf.cumsum(counts)[:-1]], axis=0)
            within_idx = tf.range(tf.shape(targets)[0], dtype=tf.int32) - tf.gather(offsets, batch_idx)
            indices = tf.stack([batch_idx, within_idx], axis=1)
            out = tf.scatter_nd(indices, targets[:, 1:], [batch_size, max_count, 5])
            xywh = out[..., 1:5] * tf.reshape(tf.cast(scale_tensor, out.dtype), [1, 1, 4])
            x, y, w, h = tf.split(xywh, 4, axis=-1)
            xyxy = tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)
            return tf.concat([out[..., 0:1], xyxy], axis=-1)

        return tf.cond(tf.shape(targets)[0] == 0, no_targets, with_targets)

    def _targets_to_flat(self, batch):
        if {"batch_idx", "flat_cls", "flat_bboxes"}.issubset(batch):
            return tf.concat(
                [
                    tf.cast(batch["batch_idx"], tf.float32),
                    tf.cast(batch["flat_cls"], tf.float32),
                    tf.cast(batch["flat_bboxes"], tf.float32),
                ],
                axis=-1,
            )
        bboxes = tf.cast(batch["bboxes"], tf.float32)
        cls = tf.cast(batch["cls"], tf.float32)
        mask = tf.cast(batch.get("mask", tf.reduce_sum(bboxes, axis=-1) > 0), tf.bool)
        idx = tf.where(mask)
        batch_idx = tf.cast(idx[:, 0:1], tf.float32)
        cls_flat = tf.gather_nd(cls, idx)[:, None]
        bboxes_flat = tf.gather_nd(bboxes, idx)
        return tf.concat([batch_idx, cls_flat, bboxes_flat], axis=-1)

    def _targets_to_xyxy(self, batch, imgsz):
        targets = self._targets_to_flat(batch)
        batch_size = tf.shape(batch["img"])[0]
        scale = tf.cast([imgsz[1], imgsz[0], imgsz[1], imgsz[0]], tf.float32)
        out = self.preprocess(targets, batch_size, scale)
        gt_labels = tf.cast(out[..., 0:1], tf.int64)
        gt_bboxes = out[..., 1:5]
        mask_gt = tf.reduce_sum(gt_bboxes, axis=-1, keepdims=True) > 0
        return gt_labels, gt_bboxes, mask_gt

    def bbox_decode(self, anchor_points, pred_dist):
        return dist2bbox(pred_dist, anchor_points[None, :, :], xywh=False)

    def loss(self, preds: dict, batch: dict):
        pred_dist = tf.cast(preds["boxes"], tf.float32)
        pred_scores = tf.cast(preds["scores"], tf.float32)
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        imgsz = tf.cast(tf.shape(batch["img"])[1:3], pred_scores.dtype)
        pred_bboxes_grid = self.bbox_decode(anchor_points, pred_dist)
        pred_bboxes = pred_bboxes_grid * stride_tensor[None, :, :]
        gt_labels, gt_bboxes, mask_gt = self._targets_to_xyxy(batch, imgsz)
        target_bboxes, target_scores, fg_mask, _ = self.assigner(
            tf.sigmoid(pred_scores), pred_bboxes, anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )
        target_scores_sum = tf.maximum(tf.reduce_sum(target_scores), 1.0)
        cls_loss_raw = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_scores, logits=pred_scores)
        if self.class_weights is not None:
            cls_loss_raw = cls_loss_raw * self.class_weights
        cls_loss = tf.reduce_sum(cls_loss_raw) / target_scores_sum
        box_loss, dfl_loss = self.bbox_loss(
            pred_dist,
            pred_bboxes_grid,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )
        loss_items = tf.stack([box_loss * self.hyp["box"], cls_loss * self.hyp["cls"], dfl_loss * self.hyp["dfl"]])
        return tf.reduce_sum(loss_items) * tf.cast(tf.shape(pred_scores)[0], tf.float32), loss_items

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds
        return self.loss(preds, batch)


class E2ELoss:
    """YOLO26 end-to-end loss combining one-to-many and one-to-one branches."""

    def __init__(self, model, hyp: dict | None = None):
        self.one2many = DetectionLoss(model, tal_topk=10, hyp=hyp)
        self.one2one = DetectionLoss(model, tal_topk=1, hyp=hyp)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds
        loss_m, items_m = self.one2many.loss(preds["one2many"], batch)
        loss_o, items_o = self.one2one.loss(preds["one2one"], batch)
        return loss_m + loss_o, items_m + items_o

    def update(self):
        return None
