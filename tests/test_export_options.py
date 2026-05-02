import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from yolo26_tf.export import tf_export_nms


def test_tf_export_nms_preserves_e2e_detection_contract():
    pred = tf.constant(
        [
            [
                [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                [1.0, 1.0, 11.0, 11.0, 0.8, 0.0],
                [20.0, 20.0, 30.0, 30.0, 0.7, 1.0],
            ]
        ],
        tf.float32,
    )
    out = tf_export_nms(pred, conf=0.1, iou=0.5, max_det=4).numpy()
    assert out.shape == (1, 4, 6)
    assert np.isclose(out[0, 0, 4], 0.9)
    assert np.isclose(out[0, 1, 5], 1.0)

