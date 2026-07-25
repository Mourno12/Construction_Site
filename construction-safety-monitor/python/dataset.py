"""
dataset.py
Dataset loader for fine-tuning the ViT PPE/state classifier.

Expected layout - a CSV annotations file plus an images folder:

    dataset/
      images/
        img_0001.jpg
        img_0002.jpg
        ...
      train_annotations.csv
      val_annotations.csv

Each CSV has one column per label in config.ALL_TRAINABLE_LABELS, plus
`filename` - by default that's:

    filename, helmet, vest, gloves, boots, mask, fallen

  - filename : image file name inside dataset/images/ (a crop of ONE worker)
  - each label column: 1 if present/true, else 0
    (helmet/vest/gloves/boots/mask = PPE item worn; fallen = worker is down,
    not standing)

Tip: you can bootstrap this dataset by running the pipeline in DEMO MODE on
your own site footage, exporting the per-worker crops, and hand-correcting
the heuristic labels rather than annotating everything from scratch.
"""

import os
import csv
from PIL import Image
from torch.utils.data import Dataset

import config


class PPEDataset(Dataset):
    def __init__(self, images_dir, annotations_csv, processor, labels=None):
        self.images_dir = images_dir
        self.processor = processor
        self.labels = labels or config.ALL_TRAINABLE_LABELS
        self.samples = []
        with open(annotations_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((
                    row["filename"],
                    [float(row[label]) for label in self.labels],
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, labels = self.samples[idx]
        path = os.path.join(self.images_dir, filename)
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pixel_values, "labels": labels}


def make_sample_dataset_structure(root):
    """Creates an empty dataset/ skeleton with a README so users know
    exactly where to drop their own images + CSVs."""
    images_dir = os.path.join(root, "images")
    os.makedirs(images_dir, exist_ok=True)
    readme_path = os.path.join(root, "README.md")
    if not os.path.exists(readme_path):
        header = ",".join(["filename"] + config.ALL_TRAINABLE_LABELS)
        sample_row_1 = ",".join(["img_0001.jpg"] + ["1"] * len(config.ALL_TRAINABLE_LABELS))
        sample_row_2 = ",".join(["img_0002.jpg"] + ["0"] * len(config.ALL_TRAINABLE_LABELS))
        with open(readme_path, "w") as f:
            f.write(
                "# PPE Dataset\n\n"
                "Put your worker-crop images in `images/`, then create\n"
                "`train_annotations.csv` and `val_annotations.csv` with columns:\n\n"
                f"```\n{header}\n{sample_row_1}\n{sample_row_2}\n```\n\n"
                "Then run:\n\n```\npython train_vit.py --data_dir ./dataset\n```\n"
            )


if __name__ == "__main__":
    make_sample_dataset_structure(os.path.join(os.path.dirname(__file__), "dataset"))
    print("Dataset skeleton created at python/dataset/. Add images/ + CSVs, then run train_vit.py")
