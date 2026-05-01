from scripts.benchmark_coco_yolo26n import normalize_imgsz


def test_normalize_imgsz_rounds_to_stride_multiple():
    assert normalize_imgsz(640) == 640
    assert normalize_imgsz(642) == 672
