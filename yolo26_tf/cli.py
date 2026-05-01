"""Command line interface for yolo26-tf."""

from __future__ import annotations

import argparse

from .api import YOLO26


def main(argv=None):
    parser = argparse.ArgumentParser(prog="yolo26-tf")
    sub = parser.add_subparsers(dest="task")
    detect = sub.add_parser("detect")
    detect_sub = detect.add_subparsers(dest="mode")

    train = detect_sub.add_parser("train")
    train.add_argument("--model", default="yolo26n.yaml")
    train.add_argument("--data", required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--project", default="runs/detect")
    train.add_argument("--name", default="train")
    train.add_argument("--optimizer", default="auto")
    train.add_argument("--lr0", type=float, default=0.01)
    train.add_argument("--mosaic", type=float, default=1.0)
    train.add_argument("--mixup", type=float, default=0.0)
    train.add_argument("--cutmix", type=float, default=0.0)

    val = detect_sub.add_parser("val")
    val.add_argument("--model", default="yolo26n.yaml")
    val.add_argument("--weights")
    val.add_argument("--data", required=True)
    val.add_argument("--imgsz", type=int, default=640)
    val.add_argument("--batch", type=int, default=16)

    pred = detect_sub.add_parser("predict")
    pred.add_argument("--model", default="yolo26n.yaml")
    pred.add_argument("--weights")
    pred.add_argument("--source", required=True)
    pred.add_argument("--imgsz", type=int, default=640)
    pred.add_argument("--conf", type=float, default=0.25)

    exp = detect_sub.add_parser("export")
    exp.add_argument("--model", default="yolo26n.yaml")
    exp.add_argument("--weights")
    exp.add_argument("--format", default="saved_model")
    exp.add_argument("--output")
    exp.add_argument("--imgsz", type=int, default=640)

    args = parser.parse_args(argv)
    if args.task != "detect" or args.mode is None:
        parser.print_help()
        return 2
    y = YOLO26(args.model, imgsz=getattr(args, "imgsz", 640))
    if getattr(args, "weights", None):
        y.model.load_weights(args.weights)
    if args.mode == "train":
        y.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=args.project, name=args.name, optimizer=args.optimizer, lr0=args.lr0, mosaic=args.mosaic, mixup=args.mixup, cutmix=args.cutmix)
    elif args.mode == "val":
        y.val(args.data, imgsz=args.imgsz, batch=args.batch)
    elif args.mode == "predict":
        results = y.predict(args.source, imgsz=args.imgsz, conf=args.conf)
        for r in results:
            print(r["path"], len(r["boxes"]), "detections")
    elif args.mode == "export":
        print(y.export(format=args.format, output=args.output, imgsz=args.imgsz))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
