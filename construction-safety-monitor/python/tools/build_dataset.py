"""
build_dataset.py
Bridges public YOLO-format detection datasets (Kaggle/Roboflow/Ultralytics -
images + per-image bounding-box .txt files) into the per-worker-crop CSV
format train_vit.py trains on.

Two source shapes are supported:

  "ppe"  - a dataset with a Person class plus PPE-item classes (helmet,
           vest, gloves, boots, mask, or however that dataset names them).
           For each Person box, this crops the person out and checks which
           PPE-item boxes overlap that crop to derive 1/0 labels. Any label
           this source dataset doesn't have a class for at all is left
           BLANK (unknown), not 0 - see dataset.py's masked-loss handling.

  "fall" - a dataset where each labelled box IS a person, already
           classified as fallen or not (e.g. classes named "fall" /
           "not_fall"). Crops each box directly; no overlap matching needed.

Ready-made mappings are included for the four datasets discussed in
README.md's "Training your own ViT classifier" / dataset-sourcing
discussion: Ultralytics Construction-PPE, SH17, Roboflow "Fall Detection",
and Roboflow "Person Fallen/Not Fallen". IMPORTANT: class-name spelling
varies release to release - this script always prints the class list it
actually found in your downloaded data.yaml/classes.txt before converting
anything, specifically so you can catch a mismatched preset before it
silently mislabels a few thousand images. If the printed names don't match
what CLASS_MAP below expects, fix CLASS_MAP for that preset (or pass
--custom-map) before re-running.

Usage (run once per source dataset - each run appends to the same shared
train/val CSVs, so you build up the combined dataset incrementally):

    python tools/build_dataset.py ppe  --dataset construction-ppe --source-dir ../raw_datasets/construction-ppe
    python tools/build_dataset.py ppe  --dataset sh17              --source-dir ../raw_datasets/sh17
    python tools/build_dataset.py fall --dataset fall-detection     --source-dir ../raw_datasets/fall-detection
    python tools/build_dataset.py fall --dataset person-fallen      --source-dir ../raw_datasets/person-fallen

Then train:
    python train_vit.py --model ppe
    python train_vit.py --model fall
"""

import os
import sys
import csv
import glob
import json
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402

from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402


# ---------------------------------------------------------------------------
# Per-dataset presets
#
# `person_names`   : class names in the source that mean "this box is a person"
#                     (only used for task="ppe"; task="fall" treats every
#                     labelled box as a person already).
# `label_map`      : {our_label: [source class names that mean "present"]}
#                     Only include a key here if the source actually
#                     annotates that label at all - omit keys entirely for
#                     labels it doesn't cover (they'll be left blank).
# `fallen_map`     : task="fall" only - {source class name: 0 or 1}
# ---------------------------------------------------------------------------
PRESETS = {
    "construction-ppe": {
        "task": "ppe",
        "person_names": ["Person", "person"],
        "label_map": {
            "helmet": ["helmet", "Helmet"],
            "vest": ["vest", "Vest"],
            "gloves": ["gloves", "Gloves"],
            "boots": ["boots", "Boots"],
            # no "mask" class in this dataset - left uncovered on purpose
        },
    },
    "sh17": {
        "task": "ppe",
        "person_names": ["Person", "person"],
        "label_map": {
            "helmet": ["Helmet", "helmet"],
            "vest": ["Safety-vest", "safety-vest", "Safety_Vest"],
            "gloves": ["Gloves", "gloves"],
            "mask": ["Face-mask-medical", "face-mask-medical", "Mask"],
            # Deliberately not mapping "Foot"/bare-foot here - a visible bare
            # foot more likely signals boots are ABSENT, not present.
            "boots": ["Shoes", "shoes"],
        },
    },
    "fall-detection": {
        "task": "fall",
        "fallen_map": {
            "fall": 1, "Fall": 1, "fallen": 1, "Fallen": 1, "Fall-Detected": 1,
            "not_fall": 0, "not fallen": 0, "Not-Fallen": 0, "person": 0, "Person": 0,
        },
    },
    "person-fallen": {
        "task": "fall",
        "fallen_map": {
            "fall": 1, "Fall": 1, "fallen": 1, "Fallen": 1,
            "not fallen": 0, "not_fallen": 0, "Not-Fallen": 0, "person": 0, "Person": 0,
        },
    },
}


# ---------------------------------------------------------------------------
# YOLO-format dataset discovery
# ---------------------------------------------------------------------------
def discover_classes(source_dir):
    """
    Reads the class-id -> class-name list from whichever convention this
    dataset uses: a data.yaml with a `names:` list/dict, or a plain
    classes.txt (one name per line, in class-id order).
    """
    yaml_path = None
    for candidate in ("data.yaml", "data.yml"):
        p = os.path.join(source_dir, candidate)
        if os.path.exists(p):
            yaml_path = p
            break

    if yaml_path:
        import yaml
        # encoding="utf-8" explicitly - Windows' default open() uses the
        # system locale (cp1252 here), which chokes on non-ASCII bytes like
        # the emoji in Ultralytics' own data.yaml header comments.
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if isinstance(names, dict):
            return [names[i] for i in sorted(names)]
        if isinstance(names, list):
            return names
        raise ValueError(f"Could not find a usable 'names' list in {yaml_path}")

    for candidate in ("classes.txt", "labels.txt"):
        p = os.path.join(source_dir, candidate)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        # Sometimes nested one level down (e.g. train/classes.txt)
        matches = glob.glob(os.path.join(source_dir, "**", candidate), recursive=True)
        if matches:
            with open(matches[0], encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

    raise FileNotFoundError(
        f"Could not find data.yaml or classes.txt under {source_dir}. "
        "Point --source-dir at the folder that actually contains one of those "
        "(often the top level of the extracted zip)."
    )


def find_image_label_pairs(source_dir):
    """
    Walks the dataset for (image_path, label_txt_path) pairs. Two common
    YOLO export layouts are handled, since datasets are inconsistent about
    which one they use:
      - Ultralytics-style: images/<split>/x.jpg -> labels/<split>/x.txt
        (images/ and labels/ are siblings, each holding split subfolders -
        this is what Construction-PPE uses)
      - Roboflow-style:    <split>/images/x.jpg -> <split>/labels/x.txt
        (the split folder itself holds both images/ and labels/)
      - Flat (no split subfolders at all): images/x.jpg -> labels/x.txt
    A label file with no matching image (or vice versa) is skipped.
    """
    pairs = []
    for dirpath, _dirnames, filenames in os.walk(source_dir):
        # os.walk (not glob) deliberately - a glob pattern like "**/images/*"
        # only recurses as many levels as written into the pattern, and on
        # Windows "**" matching zero segments means "*" lists images/'s
        # direct children (the split folders themselves) rather than the
        # files nested inside them. Walking the whole tree and checking for
        # an "images" path component anywhere avoids that entirely.
        if "images" not in os.path.normpath(dirpath).split(os.sep):
            continue
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            img_path = os.path.join(dirpath, fname)
            stem = os.path.splitext(fname)[0]
            images_dir = dirpath

            split_name = os.path.basename(images_dir)
            images_root = os.path.dirname(images_dir)
            ultralytics_candidate = os.path.join(os.path.dirname(images_root), "labels", split_name, stem + ".txt")
            roboflow_or_flat_candidate = os.path.join(os.path.dirname(images_dir), "labels", stem + ".txt")

            for label_path in (ultralytics_candidate, roboflow_or_flat_candidate):
                if os.path.exists(label_path):
                    pairs.append((img_path, label_path))
                    break

    if not pairs:
        raise FileNotFoundError(
            f"No (image, label) pairs found under {source_dir}. Expected an "
            "images/ folder with a corresponding labels/ folder, in either "
            "Ultralytics-style (images/<split>/, labels/<split>/) or "
            "Roboflow-style (<split>/images/, <split>/labels/) layout."
        )
    return pairs


def read_yolo_boxes(label_path, class_names):
    """Returns [(class_name, cx, cy, w, h)] with normalized (0-1) coordinates."""
    boxes = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id >= len(class_names):
                continue
            cx, cy, w, h = (float(v) for v in parts[1:5])
            boxes.append((class_names[cls_id], cx, cy, w, h))
    return boxes


def denormalize(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


# train_vit.py resizes every crop to config.VIT_IMAGE_SIZE (224px) before
# it ever reaches the model, so storing crops far larger than that is pure
# waste - some source datasets (e.g. SH17's high-res stock photos) produce
# person crops several thousand pixels on a side and multiple MB each
# straight out of a naive crop+save, ballooning the dataset into the tens
# of GB for no accuracy benefit. Cap the long side here instead.
MAX_CROP_SIDE = 512


def save_crop(crop, path):
    crop.thumbnail((MAX_CROP_SIDE, MAX_CROP_SIDE), Image.LANCZOS)
    crop.save(path, quality=90)


def containment_ratio(inner_box, outer_box):
    """Fraction of inner_box's area that falls inside outer_box - used to
    decide "this PPE item box belongs to this person box", which is more
    robust than IoU for a small item box nested inside a much larger person
    box (their IoU is naturally low even for a perfect match)."""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    inter_x1, inter_y1 = max(ix1, ox1), max(iy1, oy1)
    inter_x2, inter_y2 = min(ix2, ox2), min(iy2, oy2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    return inter_area / inner_area


# ---------------------------------------------------------------------------
# CSV output (appends across multiple source-dataset runs)
# ---------------------------------------------------------------------------
def append_rows(out_dir, labels, rows, split_ratio):
    """rows: [(filename, {label: 0/1/None})]. Splits into train/val here so
    every source dataset contributes to both, rather than one dataset
    entirely becoming the val set."""
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * (1 - split_ratio))) if len(rows) > 4 else 0
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    for name, split_rows in (("train_annotations.csv", train_rows), ("val_annotations.csv", val_rows)):
        path = os.path.join(out_dir, name)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["filename"] + labels)
            for filename, label_values in split_rows:
                writer.writerow([filename] + [
                    "" if label_values.get(label) is None else int(label_values[label])
                    for label in labels
                ])
    print(f"  -> appended {len(train_rows)} train / {len(val_rows)} val rows to {out_dir}")


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------
def convert_ppe_source(spec, source_dir, out_dir, split_ratio, min_iou):
    class_names = discover_classes(source_dir)
    print(f"Classes found in {source_dir}:\n  {class_names}")
    label_map = spec["label_map"]
    covered_labels = set(label_map.keys())
    print(f"Labels this source covers: {sorted(covered_labels)} (others left blank/unknown for these rows)")

    pairs = find_image_label_pairs(source_dir)
    print(f"Found {len(pairs)} image/label pairs - cropping (this can take a while for large datasets, with no output until it's done otherwise)")
    images_out = os.path.join(out_dir, "images")
    os.makedirs(images_out, exist_ok=True)

    rows = []
    crop_counter = 0
    for img_path, label_path in tqdm(pairs, desc="Cropping person boxes", unit="img"):
        boxes = read_yolo_boxes(label_path, class_names)
        with Image.open(img_path) as img:
            img_w, img_h = img.size
            person_boxes = [
                denormalize(cx, cy, w, h, img_w, img_h)
                for name, cx, cy, w, h in boxes if name in spec["person_names"]
            ]
            item_boxes = [
                (name, denormalize(cx, cy, w, h, img_w, img_h))
                for name, cx, cy, w, h in boxes if name not in spec["person_names"]
            ]

            for person_box in person_boxes:
                x1, y1, x2, y2 = person_box
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue  # too small to be a useful crop

                label_values = {label: (0 if label in covered_labels else None) for label in config.PPE_LABELS}
                for item_name, item_box in item_boxes:
                    for label, source_names in label_map.items():
                        if item_name in source_names and containment_ratio(item_box, person_box) >= min_iou:
                            label_values[label] = 1

                crop = img.crop(person_box).convert("RGB")
                crop_counter += 1
                filename = f"{spec['name']}_{crop_counter:06d}.jpg"
                save_crop(crop, os.path.join(images_out, filename))
                rows.append((filename, label_values))

    print(f"Cropped {crop_counter} person boxes from {len(pairs)} images.")
    append_rows(out_dir, config.PPE_LABELS, rows, split_ratio)


def convert_fall_source(spec, source_dir, out_dir, split_ratio):
    class_names = discover_classes(source_dir)
    print(f"Classes found in {source_dir}:\n  {class_names}")
    fallen_map = spec["fallen_map"]
    unmapped = [c for c in class_names if c not in fallen_map]
    if unmapped:
        print(f"  WARNING: these classes aren't in fallen_map and will be skipped entirely: {unmapped}")

    pairs = find_image_label_pairs(source_dir)
    print(f"Found {len(pairs)} image/label pairs - cropping (this can take a while for large datasets, with no output until it's done otherwise)")
    images_out = os.path.join(out_dir, "images")
    os.makedirs(images_out, exist_ok=True)

    rows = []
    crop_counter = 0
    for img_path, label_path in tqdm(pairs, desc="Cropping labelled boxes", unit="img"):
        boxes = read_yolo_boxes(label_path, class_names)
        with Image.open(img_path) as img:
            img_w, img_h = img.size
            for name, cx, cy, w, h in boxes:
                if name not in fallen_map:
                    continue
                box = denormalize(cx, cy, w, h, img_w, img_h)
                x1, y1, x2, y2 = box
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue
                crop = img.crop(box).convert("RGB")
                crop_counter += 1
                filename = f"{spec['name']}_{crop_counter:06d}.jpg"
                save_crop(crop, os.path.join(images_out, filename))
                rows.append((filename, {"fallen": fallen_map[name]}))

    print(f"Cropped {crop_counter} labelled boxes from {len(pairs)} images.")
    append_rows(out_dir, config.FALL_LABELS, rows, split_ratio)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["ppe", "fall"])
    parser.add_argument("--dataset", required=True, choices=list(PRESETS), help="Which built-in preset to use")
    parser.add_argument("--source-dir", required=True, help="Path to the downloaded/extracted dataset")
    parser.add_argument("--out-dir", default=None, help="Defaults to python/dataset_ppe or python/dataset_fall")
    parser.add_argument("--split-ratio", type=float, default=0.85, help="Fraction of rows going to train (rest to val)")
    parser.add_argument("--min-overlap", type=float, default=0.5, help="ppe only: min containment ratio to count a PPE box as belonging to a person box")
    args = parser.parse_args()

    preset = PRESETS[args.dataset]
    if preset["task"] != args.task:
        raise SystemExit(f"--dataset {args.dataset} is a '{preset['task']}' dataset, not '{args.task}'")
    preset = dict(preset, name=args.dataset)

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "..", f"dataset_{args.task}")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Converting '{args.dataset}' ({args.task}) from {args.source_dir} -> {out_dir}\n")
    if args.task == "ppe":
        convert_ppe_source(preset, args.source_dir, out_dir, args.split_ratio, args.min_overlap)
    else:
        convert_fall_source(preset, args.source_dir, out_dir, args.split_ratio)

    print(f"\nDone. Run more sources into the same {out_dir} to combine datasets, "
          f"then: python train_vit.py --model {args.task} --data_dir {out_dir}")


if __name__ == "__main__":
    main()
