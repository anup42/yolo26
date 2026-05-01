from scripts.benchmark_coco_yolo26n import is_gpu_dnn_error, normalize_imgsz


def test_normalize_imgsz_rounds_to_stride_multiple():
    assert normalize_imgsz(640) == 640
    assert normalize_imgsz(642) == 672


def test_gpu_dnn_error_detection_matches_tensorflow_messages():
    assert is_gpu_dnn_error(RuntimeError("CUDNN_STATUS_NOT_INITIALIZED: Could not create cudnn handle"))
    assert is_gpu_dnn_error(RuntimeError("No DNN in stream executor. [Op:Conv2D]"))
    assert not is_gpu_dnn_error(RuntimeError("Missing COCO annotations"))
