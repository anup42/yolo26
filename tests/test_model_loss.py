from pathlib import Path

import pytest


tf = pytest.importorskip("tensorflow")

from yolo26_tf.model import build_model
from yolo26_tf.losses import E2ELoss


def test_model_shapes_detect_variants():
    for name in ["yolo26n.yaml", "yolo26s.yaml", "yolo26n-p2.yaml", "yolo26n-p6.yaml"]:
        model = build_model(name, nc=2, imgsz=64)
        y = model(tf.zeros([1, 64, 64, 3]), training=False)
        assert y.shape[0] == 1
        assert y.shape[-1] == 6


def test_detect_layer_avoids_keras_dynamic_property():
    model = build_model("yolo26n.yaml", nc=2, imgsz=64)
    assert not hasattr(model.detect_layer, "__dict__") or "dynamic" not in model.detect_layer.__dict__
    assert hasattr(model.detect_layer, "dynamic_grid")


def test_loss_is_finite():
    model = build_model("yolo26n.yaml", nc=1, imgsz=64)
    batch = {
        "img": tf.zeros([1, 64, 64, 3], tf.float32),
        "bboxes": tf.constant([[[0.5, 0.5, 0.25, 0.25]]], tf.float32),
        "cls": tf.constant([[0]], tf.int64),
        "mask": tf.constant([[True]]),
    }
    preds = model(batch["img"], training=True)
    loss, items = E2ELoss(model, hyp={"epochs": 1})(preds, batch)
    assert bool(tf.reduce_all(tf.math.is_finite(loss)))
    assert bool(tf.reduce_all(tf.math.is_finite(items)))
