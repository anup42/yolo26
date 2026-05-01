"""Export utilities for TensorFlow-native YOLO26 models."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .tf_import import require_tf

tf = require_tf()


class ServingModule(tf.Module):
    def __init__(self, model):
        super().__init__()
        # Do not attach the Keras model as a trackable child. Keras 3 can keep
        # call metadata wrappers from training calls that break SavedModel object
        # graph traversal. Tracking variables and closing over the model is enough
        # for TensorFlow to capture the serving graph and weights.
        self.weights = list(model.weights)

        @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32, name="images")])
        def serve(images):
            return {"detections": model(images, training=False)}

        self.serve = serve


def export_model(model, format: str = "keras", output: str | Path | None = None, imgsz: int = 640, **kwargs):
    fmt = format.lower()
    output = Path(output) if output else Path(f"yolo26n_{fmt}")
    if fmt in {"keras", "h5"}:
        path = output if output.suffix else output.with_suffix(".keras")
        try:
            model.save(path)
            return str(path)
        except Exception:
            # Subclassed custom models may not be fully serializable in all Keras versions.
            weights = path.with_suffix(".weights.h5")
            model.save_weights(str(weights))
            path.with_suffix(".txt").write_text("Keras full-model save failed; weights exported instead. Use DetectionModel(...).load_weights().\n")
            return str(weights)
    if fmt in {"saved_model", "savedmodel"}:
        path = output if not output.suffix else output.with_suffix("")
        path.parent.mkdir(parents=True, exist_ok=True)
        module = ServingModule(model)
        tf.saved_model.save(module, str(path), signatures={"serving_default": module.serve})
        return str(path)
    if fmt == "tflite":
        @tf.function(input_signature=[tf.TensorSpec([1, imgsz, imgsz, 3], tf.float32, name="images")])
        def serve(images):
            return model(images, training=False)

        concrete = serve.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
        if kwargs.get("half", False):
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        if kwargs.get("int8", False):
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        path = output if output.suffix == ".tflite" else output.with_suffix(".tflite")
        path.write_bytes(tflite_model)
        return str(path)
    if fmt == "pb":
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # type: ignore

        func = tf.function(lambda x: model(x, training=False)).get_concrete_function(
            tf.TensorSpec([None, imgsz, imgsz, 3], tf.float32)
        )
        frozen = convert_variables_to_constants_v2(func)
        path = output if output.suffix == ".pb" else output.with_suffix(".pb")
        tf.io.write_graph(frozen.graph, str(path.parent), path.name, as_text=False)
        return str(path)
    if fmt == "onnx":
        try:
            import tf2onnx  # noqa: F401
        except Exception as exc:
            raise ImportError("ONNX export requires `pip install tf2onnx`.") from exc
        path = output if output.suffix == ".onnx" else output.with_suffix(".onnx")
        spec = (tf.TensorSpec([None, imgsz, imgsz, 3], tf.float32, name="images"),)
        import tf2onnx

        tf2onnx.convert.from_keras(model, input_signature=spec, output_path=str(path), opset=13)
        return str(path)
    if fmt == "tfjs":
        saved = export_model(model, "saved_model", output, imgsz=imgsz)
        out_dir = output if not output.suffix else output.with_suffix("")
        cmd = ["tensorflowjs_converter", "--input_format=tf_saved_model", saved, str(out_dir)]
        subprocess.run(cmd, check=True)
        return str(out_dir)
    raise ValueError(f"Unsupported export format '{format}'")


