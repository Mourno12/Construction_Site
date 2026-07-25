"""
train_vit.py
Complete transfer-learning pipeline that fine-tunes a Vision Transformer
(google/vit-base-patch16-224-in21k) into a 2-head sigmoid classifier for:
    output[0] = P(helmet worn)
    output[1] = P(vest worn)

Usage:
    python train_vit.py --data_dir ./dataset --epochs 15 --batch_size 32

Produces a checkpoint at python/models/vit_ppe_classifier/ which
inference_worker.py automatically picks up on the next run (switching the
whole app out of DEMO MODE).
"""

import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import config
from dataset import PPEDataset


def build_model():
    model = ViTForImageClassification.from_pretrained(
        config.VIT_BASE_MODEL,
        num_labels=2,                      # [helmet, vest]
        problem_type="multi_label_classification",
    )
    return model


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.float32)
    return {"pixel_values": pixel_values, "labels": labels}


def run_epoch(model, loader, optimizer, device, criterion, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if train else "val", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * pixel_values.size(0)
            preds = (torch.sigmoid(outputs.logits) >= 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels.flatten(), all_preds.flatten())
    f1 = f1_score(all_labels.flatten(), all_preds.flatten(), zero_division=0)
    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT for PPE (helmet+vest) classification")
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "dataset"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--output_dir", default=config.VIT_CHECKPOINT_DIR)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    processor = ViTImageProcessor.from_pretrained(config.VIT_BASE_MODEL)
    images_dir = os.path.join(args.data_dir, "images")
    train_csv = os.path.join(args.data_dir, "train_annotations.csv")
    val_csv = os.path.join(args.data_dir, "val_annotations.csv")

    for p in (images_dir, train_csv, val_csv):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Expected {p} to exist. Run `python dataset.py` first to see the "
                "expected dataset layout, or check the README in python/dataset/."
            )

    train_ds = PPEDataset(images_dir, train_csv, processor)
    val_ds = PPEDataset(images_dir, val_csv, processor)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, optimizer, device, criterion, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, optimizer, device, criterion, train=False)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"  -> New best model saved to {args.output_dir} (val_f1={val_f1:.4f})")

    print(f"Training complete. Best val F1: {best_f1:.4f}. Checkpoint at: {args.output_dir}")
    print("Restart the app (or the Node server) to pick up the fine-tuned model automatically.")


if __name__ == "__main__":
    main()
