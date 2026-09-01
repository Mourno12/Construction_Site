"""
train_vit.py
Complete transfer-learning pipeline that fine-tunes a Vision Transformer
(google/vit-base-patch16-224-in21k) into a multi-label sigmoid classifier.

Trains ONE of two separate models per run, selected with --model:
    --model ppe   -> config.PPE_LABELS  (helmet, vest, gloves, boots, mask)
    --model fall  -> config.FALL_LABELS (fallen)
See config.py for why these are two separate checkpoints rather than one.

Usage:
    python train_vit.py --model ppe  --data_dir ./dataset_ppe  --epochs 15
    python train_vit.py --model fall --data_dir ./dataset_fall --epochs 15

Some training rows may not have every label annotated (e.g. combining
multiple source PPE datasets where no single one covers all 5 items - see
python/tools/build_dataset.py). dataset.py marks those as blank/masked
rather than guessing 0, and the loss/metrics below skip masked entries
entirely so an unlabelled item never trains the model toward a wrong
answer.

Produces a checkpoint at python/models/vit_ppe_classifier/ or
python/models/vit_fall_classifier/, which ppe_classifier.py automatically
picks up on the next run (switching that piece out of DEMO MODE).

Resume-friendly by design: if --output_dir already contains a saved
checkpoint (e.g. because a previous run was interrupted - a disconnected
Colab session losing its ephemeral disk, say), this loads that checkpoint
and continues fine-tuning from it instead of starting over from the base
pretrained model. Point --output_dir at a persistent location (a mounted
Google Drive folder, not local Colab disk) to make interruptions cheap -
losing at most the epochs since the last improvement, not the whole run.
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import config
from dataset import PPEDataset
from device_utils import select_device

MODEL_PRESETS = {
    "ppe": {
        "labels": config.PPE_LABELS,
        "data_dir": "dataset_ppe",
        "output_dir": config.VIT_PPE_CHECKPOINT_DIR,
    },
    "fall": {
        "labels": config.FALL_LABELS,
        "data_dir": "dataset_fall",
        "output_dir": config.VIT_FALL_CHECKPOINT_DIR,
    },
}


def build_model(labels):
    model = ViTForImageClassification.from_pretrained(
        config.VIT_BASE_MODEL,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label={i: label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
    )
    return model


def _is_existing_checkpoint(output_dir):
    return os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "config.json"))


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.float32)
    label_mask = torch.tensor([b["label_mask"] for b in batch], dtype=torch.float32)
    return {"pixel_values": pixel_values, "labels": labels, "label_mask": label_mask}


def masked_bce_loss(logits, labels, label_mask, pos_weight=None):
    """
    Elementwise BCE, zeroed out wherever label_mask is 0 (that label wasn't
    annotated for this row), then averaged only over the annotated entries -
    so rows from a dataset that doesn't cover every label never push the
    model toward guessing an unannotated one is absent.

    pos_weight (optional, one value per label - see compute_pos_weight):
    scales up the loss on missed positives for labels whose positive class
    is rare in the training data. Without this, a label like "mask" (few
    positive examples relative to negatives) gets a nearly-free ride by
    just always predicting "absent" - which is exactly the precision=1.0/
    recall=0.35 pattern seen on the first trained checkpoint.
    """
    per_element = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none", pos_weight=pos_weight
    )
    masked = per_element * label_mask
    denom = label_mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def compute_pos_weight(dataset, labels, device, cap=10.0):
    """Per-label pos_weight = negative_count / positive_count, counted only
    over annotated (non-blank) rows for that label. Standard correction for
    imbalanced binary labels, fed straight into BCEWithLogitsLoss's own
    pos_weight support. Capped so one extremely rare label (mask, in this
    project's merged dataset) can't blow up training stability by demanding
    a huge loss multiplier."""
    pos_counts = [0.0] * len(labels)
    neg_counts = [0.0] * len(labels)
    for _, label_vals, mask_vals in dataset.samples:
        for i in range(len(labels)):
            if mask_vals[i] == 0.0:
                continue
            if label_vals[i] >= 0.5:
                pos_counts[i] += 1
            else:
                neg_counts[i] += 1

    weights = []
    print("Class balance (train set) and resulting pos_weight:")
    for i, label in enumerate(labels):
        pos, neg = pos_counts[i], neg_counts[i]
        w = min(neg / pos, cap) if pos > 0 else 1.0
        weights.append(w)
        print(f"  {label:<8} pos={int(pos):<6} neg={int(neg):<6} pos_weight={w:.2f}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, optimizer, device, labels, train=True, pos_weight=None):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_masked = 0.0
    all_preds, all_labels, all_masks = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if train else "val", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            batch_labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)

            outputs = model(pixel_values=pixel_values)
            loss = masked_bce_loss(outputs.logits, batch_labels, label_mask, pos_weight=pos_weight)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n_masked = label_mask.sum().item()
            total_loss += loss.item() * n_masked
            total_masked += n_masked
            preds = (torch.sigmoid(outputs.logits) >= 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(batch_labels.cpu())
            all_masks.append(label_mask.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_masks = torch.cat(all_masks).numpy().astype(bool)
    avg_loss = total_loss / max(total_masked, 1.0)

    # Per-label F1, each computed only over that label's annotated rows -
    # this is what actually caught mask's weakness (train_vit.py used to
    # only report one flattened score across all labels combined, which a
    # rare label's poor recall barely moves since common labels dominate
    # the flattened count). macro_f1 (plain average across labels, not
    # weighted by how common each is) is what "best checkpoint" is now
    # selected on, so an improvement on a rare label like mask actually
    # counts instead of being drowned out.
    per_label_f1 = {}
    for i, label in enumerate(labels):
        mask_col = all_masks[:, i]
        if mask_col.sum() == 0:
            per_label_f1[label] = 0.0
            continue
        y_true, y_pred = all_labels[mask_col, i], all_preds[mask_col, i]
        per_label_f1[label] = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = sum(per_label_f1.values()) / len(per_label_f1) if per_label_f1 else 0.0

    flat_preds, flat_labels = all_preds[all_masks], all_labels[all_masks]
    acc = accuracy_score(flat_labels, flat_preds) if flat_labels.size else 0.0

    return avg_loss, acc, macro_f1, per_label_f1


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT for PPE items or fall detection")
    parser.add_argument("--model", choices=list(MODEL_PRESETS), default="ppe",
                         help="Which model to train: 'ppe' (helmet/vest/gloves/boots/mask) or 'fall' (fallen)")
    parser.add_argument("--data_dir", default=None, help="Defaults to python/<preset data_dir> for the chosen --model")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--output_dir", default=None, help="Defaults to the checkpoint dir config.py has for the chosen --model")
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.model]
    labels = preset["labels"]
    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), preset["data_dir"])
    output_dir = args.output_dir or preset["output_dir"]

    device = select_device()
    print(f"Training '{args.model}' model ({', '.join(labels)}) on {device}")

    processor = ViTImageProcessor.from_pretrained(config.VIT_BASE_MODEL)
    images_dir = os.path.join(data_dir, "images")
    train_csv = os.path.join(data_dir, "train_annotations.csv")
    val_csv = os.path.join(data_dir, "val_annotations.csv")

    for p in (images_dir, train_csv, val_csv):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Expected {p} to exist. Run `python dataset.py` first to see the "
                "expected dataset layout, or python/tools/build_dataset.py to build "
                "one from a public PPE/fall dataset."
            )

    # augment=True only for training - validation must stay on the real,
    # unmodified images so the reported metrics reflect actual performance.
    train_ds = PPEDataset(images_dir, train_csv, processor, labels=labels, augment=True)
    val_ds = PPEDataset(images_dir, val_csv, processor, labels=labels, augment=False)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    pos_weight = compute_pos_weight(train_ds, labels, device)

    os.makedirs(output_dir, exist_ok=True)
    resuming = _is_existing_checkpoint(output_dir)
    if resuming:
        print(f"Found an existing checkpoint at {output_dir} - resuming from it instead of the base pretrained model.")
        model = ViTForImageClassification.from_pretrained(output_dir).to(device)
    else:
        model = build_model(labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    if resuming:
        # Establish the resumed checkpoint's own val F1 as the baseline to
        # beat, via one eval-only pass - otherwise the first epoch's save
        # logic (val_f1 > best_f1, starting from 0.0) would happily
        # overwrite an already-good checkpoint with a worse one.
        _, _, best_f1, resumed_per_label = run_epoch(model, val_loader, optimizer, device, labels, train=False)
        print(f"Resumed checkpoint's current macro val_f1={best_f1:.4f} {resumed_per_label} - only saving over it if a later epoch beats this.")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, _ = run_epoch(
            model, train_loader, optimizer, device, labels, train=True, pos_weight=pos_weight
        )
        val_loss, val_acc, val_f1, val_per_label = run_epoch(
            model, val_loader, optimizer, device, labels, train=False, pos_weight=pos_weight
        )

        per_label_str = " ".join(f"{lbl}={f1:.3f}" for lbl, f1 in val_per_label.items())
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_macro_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f} | "
              f"val_per_label: {per_label_str}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            print(f"  -> New best model saved to {output_dir} (val_macro_f1={val_f1:.4f})")

    print(f"Training complete. Best val macro F1: {best_f1:.4f}. Checkpoint at: {output_dir}")
    print("Restart the app (or the Node server) to pick up the fine-tuned model automatically.")


if __name__ == "__main__":
    main()
