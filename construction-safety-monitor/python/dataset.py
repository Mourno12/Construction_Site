"""
dataset.py
Dataset loader for fine-tuning the ViT PPE classifier.

Expected layout - a CSV annotations file plus an images folder:

    dataset/
      images/
        img_0001.jpg
        img_0002.jpg
        ...
      train_annotations.csv
      val_annotations.csv

Each CSV has columns:  filename, helmet, vest
  - filename : image file name inside dataset/images/ (a crop of ONE worker)
  - helmet   : 1 if wearing a helmet/hard hat, else 0
  - vest     : 1 if wearing a hi-vis safety vest, else 0

Tip: you can bootstrap this dataset by running the pipeline in DEMO MODE on
your own site footage, exporting the per-worker crops, and hand-correcting
the heuristic labels rather than annotating everything from scratch.
"""

import os
import csv
from PIL import Image
from torch.utils.data import Dataset


class PPEDataset(Dataset):
    def __init__(self, images_dir, annotations_csv, processor):
        self.images_dir = images_dir
        self.processor = processor
        self.samples = []
        with open(annotations_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((
                    row["filename"],
                    float(row["helmet"]),
                    float(row["vest"]),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, helmet, vest = self.samples[idx]
        path = os.path.join(self.images_dir, filename)
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        labels = [helmet, vest]
        return {"pixel_values": pixel_values, "labels": labels}


def make_sample_dataset_structure(root):
    """Creates an empty dataset/ skeleton with a README so users know
    exactly where to drop their own images + CSVs."""
    images_dir = os.path.join(root, "images")
    os.makedirs(images_dir, exist_ok=True)
    readme_path = os.path.join(root, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write(
                "# PPE Dataset\n\n"
                "Put your worker-crop images in `images/`, then create\n"
                "`train_annotations.csv` and `val_annotations.csv` with columns:\n\n"
                "```\nfilename,helmet,vest\nimg_0001.jpg,1,1\nimg_0002.jpg,0,1\n```\n\n"
                "Then run:\n\n```\npython train_vit.py --data_dir ./dataset\n```\n"
            )


if __name__ == "__main__":
    make_sample_dataset_structure(os.path.join(os.path.dirname(__file__), "dataset"))
    print("Dataset skeleton created at python/dataset/. Add images/ + CSVs, then run train_vit.py")
