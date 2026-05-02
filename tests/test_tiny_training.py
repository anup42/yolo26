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
    assert "speed/train_ms_per_batch" in result["history"][-1]
    assert result["history"][-1]["speed/images_per_sec"] > 0
    pred = y.predict(tmp_path / "tiny" / "images" / "val", imgsz=64, conf=0.0)
    assert len(pred) >= 1
    exported = y.export(format="saved_model", output=tmp_path / "saved", imgsz=64)
    assert Path(exported).exists()
    assert (Path(exported) / "metadata.json").exists()
    tflite = y.export(format="tflite", output=tmp_path / "model.tflite", imgsz=64)
    assert Path(tflite).exists()
    assert Path(tflite + ".metadata.json").exists()


def test_tiny_training_eager_fallback_path(tmp_path):
    data = create_tiny_dataset(tmp_path / "tiny_eager", n=2, size=32)
    y = YOLO26("yolo26n.yaml", nc=1, imgsz=32)
    result = y.train(
        data=data,
        epochs=1,
        imgsz=32,
        batch=1,
        project=tmp_path / "runs",
        name="eager",
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        optimizer="adamw",
        lr0=1e-4,
        compile_train_step=False,
        fast_data=False,
    )
    assert Path(result["last"]).exists()
