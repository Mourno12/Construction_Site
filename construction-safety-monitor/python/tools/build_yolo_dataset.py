"""
build_yolo_dataset.py
Merges public YOLO-format detection datasets into ONE combined dataset for
fine-tuning YOLOv12n as a direct PPE-item box detector (person + helmet +
vest + gloves + boots + mask), rather than the ViT crop-classifier approach
used elsewhere in this repo (see ppe_classifier.py / train_vit.py).

Each source dataset uses its own class ids/names (and covers a different
subset of PPE items), so this can't just point YOLO at the raw folders -
every source's label files are re-written with a shared target class id
space, images are hard-linked (no data duplication, same volume) into one
merged images/labels tree, and a single data.yaml ties it together.

Usage (run once per source dataset - each run appends to the same merged
output dir, so you can add more sources later without redoing earlier ones):

    python tools/build_yolo_dataset.py --dataset construction-ppe --source-dir ../datasets/construction-ppe
    python tools/build_yolo_dataset.py --dataset sh17              --source-dir ../raw_datasets/sh17

Then train:
    python train_yolo.py
"""

import os
import sys
import shutil
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image  # noqa: E402
from tools.build_dataset import discover_classes, find_image_label_pairs  # noqa: E402

TARGET_CLASSES = ["person", "helmet", "vest", "gloves", "boots", "mask"]

# YOLO trains at imgsz=640 (see train_yolo.py) with random-scale augmentation
# on top, so anything much bigger than ~1280px on the long side is wasted
# bytes, not accuracy - some source datasets (SH17's stock photos in
# particular) ship images up to 8000px wide straight out of the camera,
# which balloons the merged dataset into double-digit GB for zero training
# benefit and makes it impractical to zip up for Colab. Downscale in place
# instead of hard-linking whenever an image exceeds this. Bounding boxes in
# the label files are normalized (0-1), so they stay valid across the
# resize with no rewriting needed.
MAX_IMAGE_SIDE = 1280

# {source dataset name: {"class_map": {source class name: target class name},
#                         "splits": {source split dir name: target split name}}}
# Any source class not listed in class_map is dropped (not remapped to
# anything - e.g. construction-ppe's "goggles"/"none"/"no_helmet" absence
# annotations, or sh17's "Head"/"Face"/"Tools"/etc). Boxes for a dropped
# class are simply omitted from the rewritten label file; an image can end
# up with an empty label file (a valid YOLO "no objects of interest here"
# negative) if none of its boxes map to a target class.
SOURCES = {
    "construction-ppe": {
        "class_map": {
            "Person": "person",
            "helmet": "helmet",
            "vest": "vest",
            "gloves": "gloves",
            "boots": "boots",
            # no mask class in this dataset
        },
        "splits": {"train": "train", "val": "val", "test": "val"},  # fold test into val
    },
    "sh17": {
        "class_map": {
            "Person": "person",
            "Helmet": "helmet",
            "Safety-vest": "vest",
            "Gloves": "gloves",
            "Shoes": "boots",              # closest available proxy for "boots"
            "Face-mask-medical": "mask",
        },
        "splits": {"train": "train", "val": "val"},
    },
}


def link_or_copy(src, dst):
    """Hard link when possible (instant, no extra disk - same volume) and the
    image is already small enough; otherwise downscale-and-save (see
    MAX_IMAGE_SIDE) or, failing that, plain copy."""
    if os.path.exists(dst):
        return
    try:
        with Image.open(src) as img:
            if max(img.size) > MAX_IMAGE_SIDE:
                img = img.convert("RGB")
                img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
                img.save(dst, quality=90)
                return
    except Exception:
        pass  # not a PIL-readable image (shouldn't normally happen) - fall through to a plain copy
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def remap_label_file(src_label_path, dst_label_path, class_names, class_map):
    """Rewrites one YOLO label file's class ids into the shared target id
    space, dropping any line whose source class isn't in class_map. Leaves
    box coordinates untouched (normalized coords don't depend on class id) -
    copied verbatim from the source line rather than reparsed/reformatted,
    so no precision is lost."""
    kept_lines = []
    with open(src_label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id >= len(class_names):
                continue
            source_name = class_names[cls_id]
            target_name = class_map.get(source_name)
            if target_name is None:
                continue
            target_id = TARGET_CLASSES.index(target_name)
            kept_lines.append(" ".join([str(target_id)] + parts[1:]) + "\n")

    with open(dst_label_path, "w", encoding="utf-8") as f:
        f.writelines(kept_lines)
    return len(kept_lines)


def split_of(img_path):
    """Which source split (train/val/test/...) an image belongs to, from its
    path - the split name is the folder directly above 'images' in the
    Ultralytics-style layout find_image_label_pairs() matches."""
    parts = os.path.normpath(img_path).split(os.sep)
    idx = parts.index("images") if "images" in parts else -1
    return parts[idx + 1] if 0 <= idx < len(parts) - 1 else "train"


def convert(dataset_name, source_dir, out_dir):
    spec = SOURCES[dataset_name]
    class_names = discover_classes(source_dir)
    print(f"Classes found in {source_dir}:\n  {class_names}")
    covered = {n for n in class_names if n in spec["class_map"]}
    dropped = [n for n in class_names if n not in spec["class_map"]]
    print(f"  -> mapping to target classes: {sorted(covered)}")
    if dropped:
        print(f"  -> dropping (no target equivalent): {sorted(dropped)}")

    pairs = find_image_label_pairs(source_dir)
    print(f"Found {len(pairs)} image/label pairs")

    n_written, n_boxes, n_empty = 0, 0, 0
    for img_path, label_path in pairs:
        src_split = split_of(img_path)
        target_split = spec["splits"].get(src_split)
        if target_split is None:
            continue  # split not requested for this source (shouldn't normally happen)

        images_out = os.path.join(out_dir, "images", target_split)
        labels_out = os.path.join(out_dir, "labels", target_split)
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)

        stem, ext = os.path.splitext(os.path.basename(img_path))
        out_name = f"{dataset_name}_{stem}"
        dst_img = os.path.join(images_out, out_name + ext)
        dst_label = os.path.join(labels_out, out_name + ".txt")

        link_or_copy(img_path, dst_img)
        n = remap_label_file(label_path, dst_label, class_names, spec["class_map"])
        n_written += 1
        n_boxes += n
        n_empty += int(n == 0)

    print(f"Wrote {n_written} images ({n_boxes} boxes kept, {n_empty} images ended up with 0 boxes) to {out_dir}")


def write_data_yaml(out_dir):
    path = os.path.join(out_dir, "data.yaml")
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(TARGET_CLASSES))
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# Auto-generated by tools/build_yolo_dataset.py - merges multiple public\n"
            "# PPE detection datasets into one shared class space. Re-run that script\n"
            "# (with a different --dataset/--source-dir) to add more sources; this file\n"
            "# is overwritten each time to keep the class list authoritative.\n"
            "# Deliberately no 'path:' key - this dataset gets zipped up and moved\n"
            "# between machines (e.g. to Colab for GPU training). An absolute path\n"
            "# here would silently break elsewhere, and - less obviously - ultralytics\n"
            "# resolves a *relative* 'path:' (even 'path: .') against the current\n"
            "# working directory the training script happens to be run from, not\n"
            "# against this file's own directory - also wrong the moment training\n"
            "# isn't launched from exactly this folder. Omitting 'path:' entirely\n"
            "# makes ultralytics fall back to resolving train/val relative to wherever\n"
            "# this data.yaml itself actually lives, which is the one thing that's\n"
            "# always true no matter the machine or cwd.\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            f"{names_block}\n"
        )
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(SOURCES))
    parser.add_argument("--source-dir", required=True, help="Path to the downloaded/extracted YOLO-format dataset")
    parser.add_argument("--out-dir", default=None, help="Defaults to python/datasets/yolo_ppe")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "..", "datasets", "yolo_ppe")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Converting '{args.dataset}' from {args.source_dir} -> {out_dir}\n")
    convert(args.dataset, args.source_dir, out_dir)
    write_data_yaml(out_dir)
    print(f"\nDone. Run more sources into the same {out_dir} to combine datasets, then: python train_yolo.py")


if __name__ == "__main__":
    main()
