import numpy as np

from yolo26_tf.metrics import ConfusionMatrix, DetMetrics, ap_per_class, targets_from_batch
from yolo26_tf.validation import prediction_to_detections


def test_ap_per_class_reports_map50_95():
    preds = [np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float32)]
    targets = [(np.array([0], dtype=np.int64), np.array([[0, 0, 10, 10]], dtype=np.float32))]
    metrics = ap_per_class(preds, targets)
    assert metrics["metrics/precision(B)"] > 0.99
    assert metrics["metrics/recall(B)"] > 0.99
    assert metrics["metrics/mAP50(B)"] > 0.99
    assert metrics["metrics/mAP50-95(B)"] > 0.99


def test_confusion_matrix_counts_background():
    cm = ConfusionMatrix(nc=2, conf=0.25, iou_thres=0.5)
    det = np.array([[0, 0, 10, 10, 0.9, 1]], dtype=np.float32)
    cm.process_batch(det, np.array([0], dtype=np.int64), np.array([[20, 20, 30, 30]], dtype=np.float32))
    assert cm.matrix[0, 2] == 1
    assert cm.matrix[2, 1] == 1


def test_targets_from_batch_uses_rectangular_input_shape():
    batch = {
        "bboxes": np.array([[[0.5, 0.5, 0.5, 0.25]]], dtype=np.float32),
        "cls": np.array([[1]], dtype=np.int64),
        "mask": np.array([[True]]),
    }
    targets = targets_from_batch(batch, (64, 128))
    cls, boxes = targets[0]
    assert cls.tolist() == [1]
    np.testing.assert_allclose(boxes, np.array([[32.0, 24.0, 96.0, 40.0]], dtype=np.float32))


def test_det_metrics_exposes_ultralytics_keys():
    preds = [np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float32)]
    targets = [(np.array([0], dtype=np.int64), np.array([[0, 0, 10, 10]], dtype=np.float32))]
    metrics = DetMetrics(names={0: "object"})
    metrics.update_stats(preds, targets)
    result = metrics.process()
    assert all(k in result for k in DetMetrics.keys)
    assert metrics.fitness > 0.99
    assert metrics.summary()[0]["Class"] == "object"


def test_prediction_to_detections_supports_multi_label_nms():
    pred = np.array([[0, 0, 10, 10, 0.1, 0.8, 0.7]], dtype=np.float32)
    det = prediction_to_detections(pred, conf=0.5, iou=0.5, max_det=10, multi_label=True)
    assert len(det) == 2
    assert set(det[:, 5].astype(int).tolist()) == {1, 2}
