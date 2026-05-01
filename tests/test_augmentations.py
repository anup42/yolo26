import random

import numpy as np

from scripts.make_tiny_dataset import create_tiny_dataset
from yolo26_tf.data import YOLODataset


def test_detection_augmentation_pipeline_preserves_valid_boxes(tmp_path):
    random.seed(0)
    np.random.seed(0)
    data_yaml = create_tiny_dataset(tmp_path / "tiny", n=8, size=64)
    hyp = {
        "mosaic": 1.0,
        "mixup": 1.0,
        "cutmix": 1.0,
        "copy_paste": 1.0,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.2,
        "shear": 2.0,
        "perspective": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 1.0,
        "flipud": 0.0,
        "bgr": 1.0,
    }
    ds = YOLODataset(data_yaml, split="train", imgsz=64, batch=2, augment=True, hyp=hyp, shuffle=False)
    batch = next(iter(ds))

    assert batch["img"].shape == (2, 64, 64, 3)
    assert np.isfinite(batch["img"]).all()
    assert np.isfinite(batch["bboxes"]).all()
    assert batch["mask"].sum() > 0
    valid_boxes = batch["bboxes"][batch["mask"]]
    assert (valid_boxes >= 0.0).all()
    assert (valid_boxes <= 1.0).all()
    assert (valid_boxes[:, 2:] > 0.0).all()
