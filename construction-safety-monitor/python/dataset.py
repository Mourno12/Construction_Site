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

Each CSV has one column per label being trained (either config.PPE_LABELS
or config.FALL_LABELS - see the `labels` argument below), plus `filename` -
e.g. for the PPE model:

    filename, helmet, vest, gloves, boots, mask

  - filename : image file name inside dataset/images/ (a crop of ONE worker)
  - each label column: 1 if present/true, 0 if not, or BLANK if this row's
    source dataset doesn't annotate that label at all (see
    python/tools/build_dataset.py, which combines multiple source datasets
    that each only cover some of the labels). A blank is not the same as a
    0 - it's excluded from the loss/metrics for that row entirely (see
    train_vit.py's masked loss), rather than being guessed as "absent".

Tip: you can bootstrap this dataset by running the pipeline in DEMO MODE on
your own site footage, exporting the per-worker crops, and hand-correcting
the heuristic labels rather than annotating everything from scratch.
"""

import os
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config

# Applied only when PPEDataset(..., augment=True) - i.e. training, never
# validation/inference. The point isn't more of the same clean stock-photo
# look the source datasets share; it's forcing the model to recognize PPE
# items under conditions those datasets barely cover (webcams, indoor
# lighting, off-angle framing) - see the "helmet looked strong on val but
# failed on a real webcam" finding that motivated this. Kept mild: this
# still has to look like a real photo of a person, not a corrupted one.
#
# scale/ratio deliberately narrow (was (0.8, 1.0) / (0.85, 1.15)): the wider
# version measurably hurt helmet (val F1 0.88 -> 0.83 over a 20-epoch run)
# while barely moving vest/boots. Helmet sits right at the top edge of a
# person crop, unlike vest/boots which occupy the central/lower body - a
# more aggressive random crop was very plausibly slicing the top of frame
# off in a meaningful fraction of training images, something vest/boots are
# far more robust to by simple virtue of where they sit in the frame. This
# keeps the crop close to the full original frame so that mechanism can't
# happen, while still keeping the color/blur/rotation augmentation (those
# don't have a "cut the top off" failure mode).
TRAIN_AUGMENTATION = transforms.Compose([
    transforms.RandomResizedCrop(config.VIT_IMAGE_SIZE, scale=(0.92, 1.0), ratio=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.RandomRotation(degrees=5),
])


class PPEDataset(Dataset):
    def __init__(self, images_dir, annotations_csv, processor, labels=None, augment=False):
        self.images_dir = images_dir
        self.processor = processor
        self.labels = labels or config.ALL_TRAINABLE_LABELS
        self.augment = augment
        self.samples = []
        with open(annotations_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels_row, mask_row = [], []
                for label in self.labels:
                    raw = row.get(label, "").strip()
                    if raw == "":
                        labels_row.append(0.0)  # placeholder value, ignored via mask below
                        mask_row.append(0.0)
                    else:
                        labels_row.append(float(raw))
                        mask_row.append(1.0)
                self.samples.append((row["filename"], labels_row, mask_row))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, labels, mask = self.samples[idx]
        path = os.path.join(self.images_dir, filename)
        image = Image.open(path).convert("RGB")
        if self.augment:
            image = TRAIN_AUGMENTATION(image)
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pixel_values, "labels": labels, "label_mask": mask}


def make_sample_dataset_structure(root, labels, model_name):
    """Creates an empty dataset skeleton (images/ + README) for one of the
    two models - so users know exactly where to drop their own images +
    CSVs for that model specifically."""
    images_dir = os.path.join(root, "images")
    os.makedirs(images_dir, exist_ok=True)
    readme_path = os.path.join(root, "README.md")
    if not os.path.exists(readme_path):
        header = ",".join(["filename"] + labels)
        sample_row_1 = ",".join(["img_0001.jpg"] + ["1"] * len(labels))
        sample_row_2 = ",".join(["img_0002.jpg"] + ["0"] * len(labels))
        with open(readme_path, "w") as f:
            f.write(
                f"# {model_name} training dataset\n\n"
                "Put your worker-crop images in `images/`, then create\n"
                "`train_annotations.csv` and `val_annotations.csv` with columns:\n\n"
                f"```\n{header}\n{sample_row_1}\n{sample_row_2}\n```\n\n"
                "A column left blank for a row means \"not annotated\" (excluded\n"
                "from training for that row), not \"0/absent\" - see train_vit.py.\n\n"
                "Tip: python/tools/build_dataset.py can build this automatically\n"
                "from public YOLO-format PPE/fall datasets (Kaggle/Roboflow).\n\n"
                f"Then run:\n\n```\npython train_vit.py --model {'ppe' if model_name == 'PPE' else 'fall'} --data_dir ./{os.path.basename(root)}\n```\n"
            )


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    make_sample_dataset_structure(os.path.join(base, "dataset_ppe"), config.PPE_LABELS, "PPE")
    make_sample_dataset_structure(os.path.join(base, "dataset_fall"), config.FALL_LABELS, "fall")
    print("Dataset skeletons created at python/dataset_ppe/ and python/dataset_fall/. Add images/ + CSVs, then run train_vit.py --model ppe|fall")
