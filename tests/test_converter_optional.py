import importlib.util

import pytest


def test_converter_optional_imports():
    if not (importlib.util.find_spec("torch") and importlib.util.find_spec("ultralytics") and importlib.util.find_spec("tensorflow")):
        pytest.skip("converter parity requires torch, ultralytics, and tensorflow")
    from yolo26_tf.converter import convert_pt_to_tf

    assert callable(convert_pt_to_tf)
