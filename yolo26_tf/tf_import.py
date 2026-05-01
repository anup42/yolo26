"""TensorFlow import helper."""

from __future__ import annotations


def require_tf():
    """Import TensorFlow lazily with a clear error for optional installs."""
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "TensorFlow is required for yolo26_tf runtime. Install with `pip install -e .[tf]`."
        ) from exc
    return tf
