"""YOLO26 TensorFlow detection port."""

from __future__ import annotations

__version__ = "0.1.5"
PINNED_ULTRALYTICS_COMMIT = "b4cf7c4751e1d532eb5b0f5a3e9d67b9583964a7"


def __getattr__(name):
    if name == "YOLO26":
        from .api import YOLO26

        return YOLO26
    if name in {"DetectionModel", "build_model"}:
        from .model import DetectionModel, build_model

        return {"DetectionModel": DetectionModel, "build_model": build_model}[name]
    raise AttributeError(name)
