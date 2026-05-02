"""Export utilities for TensorFlow-native YOLO26 models."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .tf_import import require_tf

tf = require_tf()


class ServingModule(tf.Module):
    def __init__(self, model, imgsz: int = 640, dynamic: bool = True, nms: bool = False, conf: float = 0.25, iou: float = 0.45, max_det: int = 300):
        super().__init__()
        # Do not attach the Keras model as a trackable child. Keras 3 can keep
        # call metadata wrappers from training calls that break SavedModel object
        # graph traversal. Tracking variables and closing over the model is enough
        # for TensorFlow to capture the serving graph and weights.
        self.weights = list(model.weights)

        shape = [None, None, None, 3] if dynamic else [None, imgsz, imgsz, 3]
        @tf.function(input_signature=[tf.TensorSpec(shape, tf.float32, name="images")])
        def serve(images):
            detections = model(images, training=False)
            if nms:
                detections = tf_export_nms(detections, conf=conf, iou=iou, max_det=max_det)
            return {"detections": detections}

        self.serve = serve


def export_model(model, format: str = "keras", output: str | Path | None = None, imgsz: int = 640, **kwargs):
    fmt = format.lower()
    output = Path(output) if output else Path(f"yolo26n_{fmt}")
    nms = bool(kwargs.get("nms", False))
    conf = float(kwargs.get("conf", 0.25))
    iou = float(kwargs.get("iou", 0.45))
    max_det = int(kwargs.get("max_det", 300))
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
        module = ServingModule(model, imgsz=imgsz, dynamic=kwargs.get("dynamic", True), nms=nms, conf=conf, iou=iou, max_det=max_det)
        tf.saved_model.save(module, str(path), signatures={"serving_default": module.serve})
        if kwargs.get("verify", True):
            verify_saved_model(path, imgsz)
        write_metadata(path, model, fmt, imgsz, kwargs)
        return str(path)
    if fmt == "tflite":
        @tf.function(input_signature=[tf.TensorSpec([1, imgsz, imgsz, 3], tf.float32, name="images")])
        def serve(images):
            detections = model(images, training=False)
            return tf_export_nms(detections, conf=conf, iou=iou, max_det=max_det) if nms else detections

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
        func = tf.function(lambda x: tf_export_nms(model(x, training=False), conf=conf, iou=iou, max_det=max_det) if nms else model(x, training=False)).get_concrete_function(tf.TensorSpec(shape, tf.float32))
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


def tf_export_nms(pred, conf: float = 0.25, iou: float = 0.45, max_det: int = 300):
    """TensorFlow-only NMS wrapper for export graphs.

    YOLO26 end-to-end heads usually already emit ``xyxy, conf, cls`` top-k rows;
    this function still applies class-aware NMS when requested. For raw
    ``xyxy + class scores`` tensors it uses TensorFlow's combined NMS.
    """
    pred = tf.cast(pred, tf.float32)
    last = pred.shape[-1]
    if last == 6:
        max_det_t = tf.constant(max_det, dtype=tf.int32)

        def one_image(p):
            boxes = p[:, :4]
            scores = p[:, 4]
            cls = p[:, 5]
            offsets = cls[:, None] * 7680.0
            boxes_for_nms = boxes + tf.concat([offsets, offsets, offsets, offsets], axis=-1)
            idx = tf.image.non_max_suppression(boxes_for_nms, scores, max_output_size=max_det_t, iou_threshold=iou, score_threshold=conf)
            det = tf.gather(p, idx)
            pad = tf.maximum(max_det_t - tf.shape(det)[0], 0)
            det = tf.pad(det, [[0, pad], [0, 0]])[:max_det]
            return det

        return tf.map_fn(one_image, pred, fn_output_signature=tf.TensorSpec([max_det, 6], tf.float32))
    boxes = pred[..., :4]
    scores = pred[..., 4:]
    nms_boxes, nms_scores, nms_classes, _valid = tf.image.combined_non_max_suppression(
        boxes=tf.expand_dims(boxes, axis=2),
        scores=scores,
        max_output_size_per_class=max_det,
        max_total_size=max_det,
        iou_threshold=iou,
        score_threshold=conf,
        clip_boxes=False,
    )
    return tf.concat([nms_boxes, nms_scores[..., None], nms_classes[..., None]], axis=-1)


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


def verify_saved_model(path: Path, imgsz: int):
    loaded = tf.saved_model.load(str(path))
    fn = loaded.signatures.get("serving_default")
    if fn is None:
        raise RuntimeError(f"SavedModel verification found no serving_default signature for {path}")
    out = fn(tf.zeros([1, imgsz, imgsz, 3], tf.float32))
    if not out:
        raise RuntimeError(f"SavedModel verification produced no outputs for {path}")
    first = next(iter(out.values()))
    if int(tf.size(first).numpy()) == 0:
        raise RuntimeError(f"SavedModel verification produced empty output for {path}")


def write_metadata(path: Path, model, fmt: str, imgsz: int, kwargs: dict):
    meta = {
        "format": fmt,
        "imgsz": int(imgsz),
        "task": "detect",
        "nc": int(getattr(model, "nc", 0)),
        "names": getattr(model, "names", None),
        "stride": [float(x) for x in getattr(model, "strides", [])],
        "end2end": bool(getattr(getattr(model, "detect_layer", None), "end2end", True)),
        "reg_max": int(getattr(getattr(model, "detect_layer", None), "reg_max", 1)),
        "options": {k: str(v) for k, v in kwargs.items() if k not in {"representative_data"}},
    }
    meta_path = path / "metadata.json" if path.is_dir() else path.with_suffix(path.suffix + ".metadata.json")
    try:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass


