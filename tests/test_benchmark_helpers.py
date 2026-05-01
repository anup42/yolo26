from scripts.benchmark_coco_yolo26n import normalize_imgsz
from yolo26_tf.tf_import import configure_tensorflow_env


def test_normalize_imgsz_rounds_to_stride_multiple():
    assert normalize_imgsz(640) == 640
    assert normalize_imgsz(642) == 672


def test_linux_runners_pin_numpy_below_two():
    for path in ["scripts/benchmark_coco_yolo26n_linux.sh", "scripts/train_coco_yolo26n_linux.sh"]:
        text = open(path, "r", encoding="utf-8").read()
        assert "numpy>=1.23.5,<2.0" in text
        assert "pip check" in text
        assert "TF_XLA_FLAGS" in text
        assert "--tf_xla_auto_jit=0" in text


def test_tf_import_defaults_disable_xla(monkeypatch):
    for key in ["TF_XLA_FLAGS", "TF_ENABLE_ONEDNN_OPTS", "TF_FORCE_GPU_ALLOW_GROWTH"]:
        monkeypatch.delenv(key, raising=False)
    configure_tensorflow_env()
    assert "--tf_xla_auto_jit=0" in __import__("os").environ["TF_XLA_FLAGS"]
    assert __import__("os").environ["TF_ENABLE_ONEDNN_OPTS"] == "0"
    assert __import__("os").environ["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
