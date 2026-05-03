"""Command line interface for yolo26-tf."""

from __future__ import annotations

import argparse

from .api import YOLO26
from .data import load_data_yaml


def add_train_args(parser):
    parser.add_argument("--model", default="yolo26n.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--nc", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="train")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-momentum", type=float, default=0.8)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--cache-images", default="auto", choices=("auto", "ram", "off"))
    parser.add_argument("--cache-ram-gb", type=float, default=8.0)
    parser.add_argument("--use-tfrecord", dest="use_tfrecord", action="store_true", default=True)
    parser.add_argument("--no-tfrecord", dest="use_tfrecord", action="store_false")
    parser.add_argument("--tfrecord-dir")
    parser.add_argument("--rebuild-tfrecord", action="store_true")
    parser.add_argument("--rect", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--classes", default=None, help="Comma-separated class indexes to include.")
    parser.add_argument("--single-cls", action="store_true")
    parser.add_argument("--freeze", default=0, help="Number of leading layers or comma-separated layer indexes to freeze.")
    parser.add_argument("--time", type=float, default=0.0, help="Maximum training time in hours.")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--multi-scale", nargs="?", const=0.5, default=0.0, type=float)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--val-coco", action="store_true")
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--compile", dest="compile_train_step", action="store_true", default=False, help="Experimental: compile forward/loss/gradient step with tf.function.")
    parser.add_argument("--no-compile", dest="compile_train_step", action="store_false")
    parser.add_argument("--fast-data", dest="fast_data", action="store_true", default=False, help="Opt-in fast tf.data input pipeline.")
    parser.add_argument("--no-fast-data", dest="fast_data", action="store_false")
    parser.add_argument("--fast-nms", dest="fast_nms", action="store_true", default=True)
    parser.add_argument("--no-fast-nms", dest="fast_nms", action="store_false")
    parser.add_argument("--profile-speed", dest="profile_speed", action="store_true", default=True)
    parser.add_argument("--no-profile-speed", dest="profile_speed", action="store_false")
    parser.add_argument("--profile-interval", type=int, default=0)
    parser.add_argument("--cls-pw", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mosaic-n", type=int, default=4, choices=(3, 4, 9))
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--copy-paste-mode", default="flip", choices=("flip", "mixup"))
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--bgr", type=float, default=0.0)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="yolo26-tf")
    sub = parser.add_subparsers(dest="task")
    detect = sub.add_parser("detect")
    detect_sub = detect.add_subparsers(dest="mode")

    train = detect_sub.add_parser("train")
    add_train_args(train)

    val = detect_sub.add_parser("val")
    val.add_argument("--model", default="yolo26n.yaml")
    val.add_argument("--weights")
    val.add_argument("--data", required=True)
    val.add_argument("--nc", type=int)
    val.add_argument("--imgsz", type=int, default=640)
    val.add_argument("--batch", type=int, default=16)
    val.add_argument("--conf", type=float, default=0.001)
    val.add_argument("--iou", type=float, default=0.7)
    val.add_argument("--max-det", type=int, default=300)
    val.add_argument("--coco", action="store_true")
    val.add_argument("--save-json", action="store_true")
    val.add_argument("--save-txt", action="store_true")
    val.add_argument("--save-conf", action="store_true")
    val.add_argument("--single-cls", action="store_true")
    val.add_argument("--agnostic-nms", action="store_true")
    val.add_argument("--multi-label", dest="multi_label", action="store_true", default=True)
    val.add_argument("--single-label", dest="multi_label", action="store_false")
    val.add_argument("--half", action="store_true")
    val.add_argument("--fast-nms", dest="fast_nms", action="store_true", default=True)
    val.add_argument("--no-fast-nms", dest="fast_nms", action="store_false")
    val.add_argument("--project", default="runs/detect")
    val.add_argument("--name", default="val")

    pred = detect_sub.add_parser("predict")
    pred.add_argument("--model", default="yolo26n.yaml")
    pred.add_argument("--weights")
    pred.add_argument("--nc", type=int)
    pred.add_argument("--source", required=True)
    pred.add_argument("--imgsz", type=int, default=640)
    pred.add_argument("--conf", type=float, default=0.25)
    pred.add_argument("--iou", type=float, default=0.45)
    pred.add_argument("--max-det", type=int, default=300)
    pred.add_argument("--single-cls", action="store_true")
    pred.add_argument("--agnostic-nms", action="store_true")
    pred.add_argument("--multi-label", action="store_true")

    exp = detect_sub.add_parser("export")
    exp.add_argument("--model", default="yolo26n.yaml")
    exp.add_argument("--weights")
    exp.add_argument("--nc", type=int)
    exp.add_argument("--format", default="saved_model")
    exp.add_argument("--output")
    exp.add_argument("--imgsz", type=int, default=640)
    exp.add_argument("--half", action="store_true")
    exp.add_argument("--int8", action="store_true")
    exp.add_argument("--full-integer", action="store_true")
    exp.add_argument("--dynamic", dest="dynamic", action="store_true", default=True)
    exp.add_argument("--static", dest="dynamic", action="store_false")
    exp.add_argument("--nms", action="store_true", help="Embed TensorFlow NMS in SavedModel/PB/TFLite exports.")
    exp.add_argument("--agnostic-nms", action="store_true")
    exp.add_argument("--conf", type=float, default=0.25)
    exp.add_argument("--iou", type=float, default=0.45)
    exp.add_argument("--max-det", type=int, default=300)
    exp.add_argument("--no-verify", dest="verify", action="store_false", default=True)

    bench = detect_sub.add_parser("benchmark-coco")
    bench.add_argument("--model", default="yolo26n.yaml")
    bench.add_argument("--weights", default="yolo26n.pt")
    bench.add_argument("--data", required=True)
    bench.add_argument("--imgsz", type=int, default=640)
    bench.add_argument("--batch", type=int, default=16)
    bench.add_argument("--conf", type=float, default=0.001)
    bench.add_argument("--iou", type=float, default=0.7)
    bench.add_argument("--max-det", type=int, default=300)
    bench.add_argument("--project", default="runs/benchmark")
    bench.add_argument("--name", default="yolo26n_tf_coco")

    args = parser.parse_args(argv)
    if args.task != "detect" or args.mode is None:
        parser.print_help()
        return 2

    if args.mode == "train":
        nc = args.nc if args.nc is not None else load_data_yaml(args.data)["nc"]
        y = YOLO26(args.model, nc=nc, imgsz=args.imgsz)
        y.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            warmup_momentum=args.warmup_momentum,
            warmup_bias_lr=args.warmup_bias_lr,
            patience=args.patience,
            close_mosaic=args.close_mosaic,
            fraction=args.fraction,
            seed=args.seed,
            workers=args.workers,
            gpus=args.gpus,
            cache=args.cache,
            cache_images=args.cache_images,
            cache_ram_gb=args.cache_ram_gb,
            use_tfrecord=args.use_tfrecord,
            tfrecord_dir=args.tfrecord_dir,
            rebuild_tfrecord=args.rebuild_tfrecord,
            rect=args.rect,
            resume=args.resume,
            classes=parse_int_list(args.classes),
            single_cls=args.single_cls,
            freeze=parse_freeze(args.freeze),
            time=args.time,
            amp=args.amp,
            multi_scale=args.multi_scale,
            cos_lr=args.cos_lr,
            require_gpu=args.require_gpu,
            val_coco=args.val_coco,
            save_period=args.save_period,
            log_interval=args.log_interval,
            compile_train_step=args.compile_train_step,
            fast_data=args.fast_data,
            fast_nms=args.fast_nms,
            profile_speed=args.profile_speed,
            profile_interval=args.profile_interval,
            cls_pw=args.cls_pw,
            mosaic=args.mosaic,
            mosaic_n=args.mosaic_n,
            mixup=args.mixup,
            cutmix=args.cutmix,
            copy_paste=args.copy_paste,
            copy_paste_mode=args.copy_paste_mode,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            fliplr=args.fliplr,
            flipud=args.flipud,
            bgr=args.bgr,
        )
    elif args.mode == "val":
        nc = args.nc if args.nc is not None else load_data_yaml(args.data)["nc"]
        y = YOLO26(args.model, nc=nc, imgsz=args.imgsz)
        if args.weights:
            y.model.load_weights(args.weights)
        y.val(
            args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            coco=args.coco,
            save_json=args.save_json,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            single_cls=args.single_cls,
            agnostic_nms=args.agnostic_nms,
            multi_label=args.multi_label,
            half=args.half,
            fast_nms=args.fast_nms,
            project=args.project,
            name=args.name,
        )
    elif args.mode == "benchmark-coco":
        model_ref = args.weights if str(args.weights).endswith(".pt") else args.model
        y = YOLO26(model_ref, imgsz=args.imgsz)
        if args.weights and not str(args.weights).endswith(".pt"):
            y.model.load_weights(args.weights)
        y.val(args.data, imgsz=args.imgsz, batch=args.batch, conf=args.conf, iou=args.iou, max_det=args.max_det, coco=True, save_json=True, project=args.project, name=args.name)
    elif args.mode == "predict":
        y = YOLO26(args.model, nc=args.nc, imgsz=args.imgsz)
        if args.weights:
            y.model.load_weights(args.weights)
        results = y.predict(
            args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            single_cls=args.single_cls,
            agnostic_nms=args.agnostic_nms,
            multi_label=args.multi_label,
        )
        for r in results:
            print(r["path"], len(r["boxes"]), "detections")
    elif args.mode == "export":
        y = YOLO26(args.model, nc=args.nc, imgsz=args.imgsz)
        if args.weights:
            y.model.load_weights(args.weights)
        print(
            y.export(
                format=args.format,
                output=args.output,
                imgsz=args.imgsz,
                half=args.half,
                int8=args.int8,
                full_integer=args.full_integer,
                dynamic=args.dynamic,
                nms=args.nms,
                agnostic_nms=args.agnostic_nms,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                verify=args.verify,
            )
        )
    return 0


def parse_int_list(value):
    if value in (None, ""):
        return None
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_freeze(value):
    if value in (None, ""):
        return 0
    text = str(value)
    if "," in text:
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    return int(text)


if __name__ == "__main__":
    raise SystemExit(main())
