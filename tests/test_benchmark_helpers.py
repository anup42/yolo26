from scripts.benchmark_coco_yolo26n import normalize_imgsz


def test_normalize_imgsz_rounds_to_stride_multiple():
    assert normalize_imgsz(640) == 640
    assert normalize_imgsz(642) == 672


def test_linux_runners_pin_numpy_below_two():
    for path in ["scripts/benchmark_coco_yolo26n_linux.sh", "scripts/train_coco_yolo26n_linux.sh"]:
        text = open(path, "r", encoding="utf-8").read()
        assert "numpy>=1.23.5,<2.0" in text
        assert "pip check" in text
