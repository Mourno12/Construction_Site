"""
train_yolo.py
Fine-tunes YOLO26n as a direct PPE-item box detector (person + helmet +
vest + gloves + boots + mask) on the merged dataset built by
tools/build_yolo_dataset.py.

Base model is YOLO26n, not YOLO12n - see the "Heads up on YOLOv12" note in
README.md / the docstring on detector.py's YOLOPersonDetector: Ultralytics'
own docs say YOLO12's attention-heavy blocks trade away CPU throughput and
training stability versus YOLO11/YOLO26, for accuracy gains that don't
clearly show up at the nano size this project uses. This project also hit
that instability firsthand - YOLO12n's attention path crashed reproducibly
(device-lost) on this machine's GPU backend during training. YOLO26n is the
sounder default for a model meant to plausibly run on CPU-only edge
hardware too.

This is a different approach from the ViT crop-classifier
(train_vit.py/ppe_classifier.py) that the rest of this repo's live pipeline
uses by default: instead of detecting a person then classifying PPE
presence on the crop, YOLO detects PPE items as their own boxes directly.
Training it here produces a standalone checkpoint - wiring detector.py /
inference_worker.py to consume its boxes instead of (or alongside) the ViT
classifier is a separate integration step, not done automatically by this
script.

Usage:
    python tools/build_yolo_dataset.py --dataset construction-ppe --source-dir ../datasets/construction-ppe
    python tools/build_yolo_dataset.py --dataset sh17              --source-dir ../raw_datasets/sh17
    python train_yolo.py --epochs 100

Produces <project>/<name>/weights/best.pt (project defaults to runs/detect)
- point config.py's YOLO_WEIGHTS at that file to use it (see README's
"Using your own YOLO weights"). Its class ids (0=person, 1=helmet, 2=vest,
3=gloves, 4=boots, 5=mask - see tools/build_yolo_dataset.py's
TARGET_CLASSES) match config.YOLO_PERSON_CLASS_ID's default of 0, so
person tracking keeps working unchanged even if you swap weights in.

Resume-friendly by design: ultralytics saves a `last.pt` under
<project>/<name>/weights/ after every epoch, not just at the end. If that
file already exists when this script starts (e.g. because a previous run
was interrupted - a disconnected Colab session losing its ephemeral disk,
say), this resumes training from it via ultralytics' own `resume=True`
instead of starting over from --model. Point --project at a persistent
location (a mounted Google Drive folder, not local Colab disk) to make
interruptions cheap - losing at most the epoch in progress, not the whole
run.
"""

import os
import argparse

from device_utils import select_device

DEFAULT_DATA_YAML = os.path.join(os.path.dirname(__file__), "datasets", "yolo_ppe", "data.yaml")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26n as a PPE-item box detector")
    parser.add_argument("--data", default=DEFAULT_DATA_YAML, help="Path to the merged data.yaml (see tools/build_yolo_dataset.py)")
    parser.add_argument("--model", default="yolo26n.pt", help="Base weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="cpu / xpu / 0 (cuda idx) - autodetected if omitted")
    parser.add_argument("--name", default="ppe_yolo", help="Run name under --project")
    parser.add_argument("--project", default=None, help="Where <name>/weights/{best,last}.pt gets written - defaults to ultralytics' own runs/detect. Point at a mounted Drive folder in Colab so a disconnect doesn't lose training progress.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stop patience (epochs with no val improvement)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Expected {args.data} to exist. Run tools/build_yolo_dataset.py first "
            "(once per source dataset) to build the merged PPE detection dataset."
        )

    device = args.device or select_device()
    print(f"Training YOLO PPE detector on device={device}")

    from ultralytics import YOLO

    last_pt = os.path.join(args.project or "runs/detect", args.name, "weights", "last.pt")
    if os.path.exists(last_pt):
        print(f"Found an in-progress run at {last_pt} - resuming from it instead of --model.")
        model = YOLO(last_pt)
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            patience=args.patience,
            name=args.name,
            project=args.project,
        )
    best = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"\nTraining complete. Best weights: {best}")
    print("Point config.py's YOLO_WEIGHTS (or the YOLO_WEIGHTS env var) at that "
          "file to use it. Note: the live pipeline still expects PPE items from "
          "the ViT classifier by default - see this script's docstring.")


if __name__ == "__main__":
    main()
