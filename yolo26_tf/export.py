"""Export utilities for TensorFlow-native YOLO26 models."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .tf_import import require_tf

tf = require_tf()


class ServingModule(tf.Module):
    def __init__(self, model, imgsz: int = 640, dynamic: bool = True):
        super().__init__()
        # Do not attach the Keras model as a trackable child. Keras 3 can keep
        # call metadata wrappers from training calls that break SavedModel object
        # graph traversal. Tracking variables and closing over the model is enough
        # for TensorFlow to capture the serving graph and weights.
        self.weights = list(model.weights)

        shape = [None, None, None, 3] if dynamic else [None, imgsz, imgsz, 3]
        @tf.function(input_signature=[tf.TensorSpec(shape, tf.float32, name="images")])
        def serve(images):
            return {"detections": model(images, training=False)}

        self.serve = serve


def export_model(model, format: str = "keras", output: str | Path | None = None, imgsz: int = 640, **kwargs):
    fmt = format.lower()
    output = Path(output) if output else Path(f"yolo26n_{fmt}")
    if kwargs.get("nms", False):
        raise NotImplementedError("NMS-embedded export is not supported yet; YOLO26 e2e exports use top-k postprocess.")
    if fmt in {"keras", "h5"}:
        path = output if output.suffix else output.with_suffix(".keras")
        try:
            model.save(path)
            write_metadata(path, model, fmt, imgsz, kwargs)
            return str(path)
        except Exception:
            # Subclassed custom models may not be fully serializable in all Keras versions.
            weights = path.with_suffix(".weights.h5")
            model.save_weights(str(weights))
            path.with_suffix(".txt").write_text("Keras full-model save failed; weights exported instead. Use DetectionModel(...).load_weights().\n")
            write_metadata(weights, model, "weights", imgsz, kwargs)
            return str(weights)
    if fmt in {"saved_model", "savedmodel"}:
        path = output if not output.suffix else output.with_suffix("")
        path.parent.mkdir(parents=True, exist_ok=True)
        module = ServingModule(model, imgsz=imgsz, dynamic=kwargs.get("dynamic", True))
        tf.saved_model.save(module, str(path), signatures={"serving_default": module.serve})
        write_metadata(path, model, fmt, imgsz, kwargs)
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
            rep = representative_dataset(kwargs.get("representative_data"), imgsz)
            if rep is not None:
                converter.representative_dataset = rep
                if kwargs.get("full_integer", False):
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.uint8
                    converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
        path = output if output.suffix == ".tflite" else output.with_suffix(".tflite")
        path.write_bytes(tflite_model)
        if kwargs.get("verify", True):
            verify_tflite(path, imgsz)
        write_metadata(path, model, fmt, imgsz, kwargs)
        return str(path)
    if fmt == "pb":
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # type: ignore

        shape = [None, None, None, 3] if kwargs.get("dynamic", False) else [None, imgsz, imgsz, 3]
        func = tf.function(lambda x: model(x, training=False)).get_concrete_function(tf.TensorSpec(shape, tf.float32))
        frozen = convert_variables_to_constants_v2(func)
        path = output if output.suffix == ".pb" else output.with_suffix(".pb")
        tf.io.write_graph(frozen.graph, str(path.parent), path.name, as_text=False)
        write_metadata(path, model, fmt, imgsz, kwargs)
        return str(path)
    if fmt == "onnx":
        try:
            import tf2onnx  # noqa: F401
        except Exception as exc:
            raise ImportError("ONNX export requires `pip install tf2onnx`.") from exc
        path = output if output.suffix == ".onnx" else output.with_suffix(".onnx")
        shape = [None, None, None, 3] if kwargs.get("dynamic", False) else [None, imgsz, imgsz, 3]
        spec = (tf.TensorSpec(shape, tf.float32, name="images"),)
        import tf2onnx

        tf2onnx.convert.from_keras(model, input_signature=spec, output_path=str(path), opset=13)
        write_metadata(path, model, fmt, imgsz, kwargs)
        return str(path)
    if fmt == "tfjs":
        saved = export_model(model, "saved_model", output, imgsz=imgsz)
        out_dir = output if not output.suffix else output.with_suffix("")
        cmd = ["tensorflowjs_converter", "--input_format=tf_saved_model", saved, str(out_dir)]
        subprocess.run(cmd, check=True)
        write_metadata(out_dir, model, fmt, imgsz, kwargs)
        return str(out_dir)
    raise ValueError(f"Unsupported export format '{format}'")


def representative_dataset(data, imgsz: int):
    """Build a TFLite representative dataset from callable, iterable, or numpy array input."""
    if data is None:
        return None

    def normalize(x):
        x = tf.convert_to_tensor(x, tf.float32)
        if x.shape.rank == 3:
            x = x[None]
        x = tf.image.resize(x, [imgsz, imgsz])
        return [x]

    if callable(data):
        return lambda: (normalize(x) for x in data())
    return lambda: (normalize(x) for x in data)


def verify_tflite(path: Path, imgsz: int):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    dtype = input_info["dtype"]
    if dtype.__name__ == "uint8":
        x = tf.zeros([1, imgsz, imgsz, 3], tf.uint8).numpy()
    else:
        x = tf.zeros([1, imgsz, imgsz, 3], tf.float32).numpy()
    interpreter.set_tensor(input_info["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(output_info["index"])
    if out.size == 0:
        raise RuntimeError(f"TFLite verification produced empty output for {path}")


def write_metadata(path: Path, model, fmt: str, imgsz: int, kwargs: dict):
    meta = {
        "format": fmt,
        "imgsz": int(imgsz),
        "task": "detect",
        "nc": int(getattr(model, "nc", 0)),
        "names": getattr(model, "names", None),
        "end2end": bool(getattr(getattr(model, "detect_layer", None), "end2end", True)),
        "reg_max": int(getattr(getattr(model, "detect_layer", None), "reg_max", 1)),
        "options": {k: str(v) for k, v in kwargs.items() if k not in {"representative_data"}},
    }
    meta_path = path / "metadata.json" if path.is_dir() else path.with_suffix(path.suffix + ".metadata.json")
    try:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass


