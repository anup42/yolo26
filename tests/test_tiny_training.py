from pathlib import Path

import pytest


tf = pytest.importorskip("tensorflow")

from scripts.make_tiny_dataset import create_tiny_dataset
from yolo26_tf.api import YOLO26


def test_tiny_training_predict_export(tmp_path):
    data = create_tiny_dataset(tmp_path / "tiny", n=4, size=64)
    y = YOLO26("yolo26n.yaml", nc=1, imgsz=64)
    result = y.train(data=data, epochs=1, imgsz=64, batch=2, project=tmp_path / "runs", name="smoke", mosaic=0.0, mixup=0.0, cutmix=0.0, optimizer="adamw", lr0=1e-4)
    assert Path(result["last"]).exists()
    assert (Path(result["save_dir"]) / "results.csv").exists()
    assert (Path(result["save_dir"]) / "args.json").exists()
    assert (Path(result["save_dir"]) / "final_metrics.json").exists()
    pred = y.predict(tmp_path / "tiny" / "images" / "val", imgsz=64, conf=0.0)
    assert len(pred) >= 1
    exported = y.export(format="saved_model", output=tmp_path / "saved", imgsz=64)
    assert Path(exported).exists()
    assert (Path(exported) / "metadata.json").exists()
    tflite = y.export(format="tflite", output=tmp_path / "model.tflite", imgsz=64)
    assert Path(tflite).exists()
    assert Path(tflite + ".metadata.json").exists()
