"""TensorFlow import helper."""

from __future__ import annotations

import os


def configure_tensorflow_env() -> None:
    """Set safe TensorFlow defaults before the first TensorFlow import."""
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    # Avoid long XLA/ptxas compilation stalls on systems with mismatched CUDA
    # toolkit ptxas and TensorFlow's embedded CUDA target.
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")


def require_tf():
    """Import TensorFlow lazily with a clear error for optional installs."""
    configure_tensorflow_env()
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "TensorFlow is required for yolo26_tf runtime. Install with `pip install -e .[tf]`."
        ) from exc
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    return tf
