"""Train the COMP3710 ADNI baseline CNN, using patient-disjoint CSV splits.

Example (PowerShell, run from recognition/ConvNeXt_s4983318):
    python train.py --data-root C:\\...\\AD_NC \
        --manifest-dir C:\\...\\prepared_seed42 \
        --output-dir C:\\...\\runs\\baseline_smoke \
        --epochs 1 --max-train-batches 8 --max-val-batches 4

Run from the project directory where dataset.py and modules.py are located.
No test images are used here: training uses train.csv, model selection uses val.csv.
The test manifest is checked for separation but NOT iterated over.

AI-use disclosure: ChatGPT helped scaffold this training and logging script.
Verify its results yourself and record any AI assistance in the README.
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn

from dataset import make_dataloaders
from modules import BaselineCNN


def classification_metrics(confusion):
    """Return accuracy and macro F1 from a 2x2 [true, predicted] matrix."""
    total = int(confusion.sum().item())
    accuracy = int(confusion.diag().sum().item()) / total if total else 0.0
    f1s = []
    for label in (0, 1):
        true_positive = int(confusion[label, label])
        predicted = int(confusion[:, label].sum())
        actual = int(confusion[label, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        f1s.append(f1)
    return {"accuracy": accuracy, "macro_f1": sum(f1s) / len(f1s)}


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              max_batches=0, log_every=100):
    """Train when optimizer is provided; otherwise perform validation only."""
    training = optimizer is not None
    model.train(training)
    losses = 0.0
    samples = 0
    confusion = torch.zeros((2, 2), dtype=torch.long)
    # During validation, average slice-level P(AD) to obtain patient predictions.
    patients = {}

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, dtype=torch.long, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            # CUDA automatic mixed precision saves memory on the RTX 4060.
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite loss in batch {batch_index}")
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        predicted = logits.detach().argmax(dim=1)
        pairs = labels.detach().cpu() * 2 + predicted.cpu()
        confusion += torch.bincount(pairs, minlength=4).reshape(2, 2)
        batch_size = labels.size(0)
        losses += loss.item() * batch_size
        samples += batch_size

        if not training:
            probabilities = logits.detach().float().softmax(dim=1)[:, 1].cpu().tolist()
            ground_truth = labels.detach().cpu().tolist()
            for pid, prob, label in zip(batch["patient_id"], probabilities, ground_truth):
                if pid in patients:
                    if patients[pid][2] != label:
                        raise ValueError(f"Conflicting labels for patient {pid}")
                    patients[pid][0] += prob
                    patients[pid][1] += 1
                else:
                    patients[pid] = [prob, 1, label]

        if training and log_every > 0 and batch_index % log_every == 0:
            print(f"  training batches: {batch_index}/{len(loader)}", flush=True)

    if samples == 0:
        raise RuntimeError("No batches processed. Check the dataset and batch limits.")

    metrics = classification_metrics(confusion)
    metrics["loss"] = losses / samples
    metrics["samples"] = samples
    metrics["confusion"] = confusion.tolist()

    if not training:
        patient_confusion = torch.zeros((2, 2), dtype=torch.long)
        for sum_prob, n_slices, true_label in patients.values():
            predicted_label = int(sum_prob / n_slices >= 0.5)
            patient_confusion[true_label, predicted_label] += 1
        patient_metrics = classification_metrics(patient_confusion)
        metrics["patient_accuracy"] = patient_metrics["accuracy"]
        metrics["patient_macro_f1"] = patient_metrics["macro_f1"]
        metrics["patients_seen"] = len(patients)
    return metrics


def save_curves(history, output_dir):
    """Write separate loss and accuracy plots to files; never show a GUI window."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is needed for training plots. Install with: "
            "python -m pip install matplotlib"
        ) from exc
    epochs = [row["epoch"] for row in history]
    for train_field, val_field, label, filename in [
        ("train_loss", "val_loss", "Cross-entropy loss", "loss_curve.png"),
        ("train_accuracy", "val_accuracy", "Slice-level accuracy", "accuracy_curve.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, [r[train_field] for r in history], marker="o", label="Train")
        ax.plot(epochs, [r[val_field] for r in history], marker="o", label="Validation")
        ax.set(xlabel="Epoch", ylabel=label, title=f"Baseline CNN: {label}")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="0 = full train loader; positive limit for smoke tests only")
    parser.add_argument("--max-val-batches", type=int, default=0,
                        help="0 = full val loader; positive limit for smoke tests only")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    if (args.epochs < 1 or args.batch_size < 1 or args.image_size < 32
            or args.num_workers < 0 or args.max_train_batches < 0
            or args.max_val_batches < 0 or args.patience < 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error("Invalid numeric arguments; check epochs/batch size/learning rate/limits")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use your COMP3710 .venv Python")

    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    output_dir = args.output_dir.expanduser().resolve()
    # Fresh run directories avoid silently overwriting checkpoints from other experiments.
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {output_dir}; use a new run directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats(device)

    diagnostic = args.max_train_batches > 0 or args.max_val_batches > 0
    if diagnostic:
        print("DIAGNOSTIC RUN: a limited number of batches is used.")
        print("Metrics are NOT representative and must NOT be used for the final report.")
    print(f"Device: {device}; epochs: {args.epochs}; batch size: {args.batch_size}")
    print("Classes: 0=NC, 1=AD; test images will NOT be evaluated.")

    loaders = make_dataloaders(
        data_root=args.data_root, manifest_dir=args.manifest_dir,
        batch_size=args.batch_size, image_size=args.image_size,
        num_workers=args.num_workers, seed=args.seed, augment_train=False,
    )
    train_loader, val_loader = loaders["train"], loaders["val"]
    model = BaselineCNN(num_classes=2).to(device)
    print(f"BaselineCNN parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training: {len(train_loader.dataset):,} slices; validation: {len(val_loader.dataset):,} slices")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    config = {"architecture": "BaselineCNN", "seed": args.seed,
              "epochs_requested": args.epochs, "batch_size": args.batch_size,
              "image_size": args.image_size, "lr": args.lr,
              "weight_decay": args.weight_decay, "device": str(device),
              "torch_version": torch.__version__, "diagnostic": diagnostic,
              "max_train_batches": args.max_train_batches,
              "max_val_batches": args.max_val_batches,
              "split_rule": "patient level; manifests prepared with seed 42",
              "classes": {"0": "NC", "1": "AD"},
              "test_evaluated": False}
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    history = []
    best_loss, best_epoch, wait = float("inf"), None, 0
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = run_epoch(model, train_loader, criterion, device,
                                optimizer=optimizer, scaler=scaler,
                                max_batches=args.max_train_batches)
        val_stats = run_epoch(model, val_loader, criterion, device,
                              max_batches=args.max_val_batches)
        summary = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "train_macro_f1": train_stats["macro_f1"],
            "val_loss": val_stats["loss"],
            "val_accuracy": val_stats["accuracy"],
            "val_macro_f1": val_stats["macro_f1"],
            "val_patient_accuracy": val_stats["patient_accuracy"],
            "val_patient_macro_f1": val_stats["patient_macro_f1"],
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(summary)
        print(f"Epoch {epoch:02d}/{args.epochs}: "
              f"train loss={summary['train_loss']:.4f}, acc={summary['train_accuracy']:.3f} | "
              f"val loss={summary['val_loss']:.4f}, acc={summary['val_accuracy']:.3f}, "
              f"macro-F1={summary['val_macro_f1']:.3f}, "
              f"patient acc={summary['val_patient_accuracy']:.3f} "
              f"({summary['seconds']:.1f}s)", flush=True)

        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerows(history)

        if val_stats["loss"] < best_loss:
            best_loss = val_stats["loss"]
            best_epoch = epoch
            wait = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "BaselineCNN", "epoch": epoch,
                "image_size": args.image_size, "num_classes": 2,
                "val_loss": best_loss, "diagnostic": diagnostic,
                "class_names": {0: "NC", 1: "AD"},
            }, output_dir / "best_baseline.pt")
            print("  Best checkpoint updated (by validation loss).", flush=True)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping: validation loss did not improve for {args.patience} epochs.")
                break

    save_curves(history, output_dir)
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    if device.type == "cuda" else None)
    result = {"best_epoch": best_epoch, "best_val_loss": best_loss,
              "completed_epochs": len(history),
              "total_seconds": time.perf_counter() - start,
              "peak_cuda_memory_mb": peak_vram_mb, "diagnostic": diagnostic,
              "last_train": train_stats, "last_val": val_stats}
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Done. Best epoch: {best_epoch}, peak VRAM (allocated): {peak_vram_mb}")
    print(f"Checkpoints, CSV history, curves and summary saved in: {output_dir}")
    if diagnostic:
        print("SMOKE TEST COMPLETE: do NOT report these partial-run metrics as final performance.")
    else:
        print("FULL TRAINING COMPLETE: test split remains untouched; later evaluate the selected checkpoint.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
