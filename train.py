import argparse
import json
import math
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from data.mrnet_multilabel import (
    DEFAULT_DATA_ROOT, LABEL_NAMES, MRNetDataset, read_records, split_records,
)
from data.packed_views import collate_packed_views
from model import ResNetCBAM, SIPResNetCBAM, prototype_aggregation_loss


def run_epoch(model, loader, loss_fn, device, optimizer=None, prototype_lambda=0.01):
    training = optimizer is not None
    model.train(training)
    labels, scores = [], []
    loss_total = 0.0
    exam_count = 0
    sip = isinstance(model, SIPResNetCBAM)
    bce_total = pa_total = 0.0
    with torch.set_grad_enabled(training):
        for slices, slice_counts, target, _ in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            slices = slices.to(device)
            device_target = target.to(device)
            batch_count = target.shape[0]
            if sip:
                logits, prototypes, global_features, _ = model(slices, slice_counts, return_aux=True)
                bce = loss_fn(logits, device_target)
                pa = prototype_aggregation_loss(device_target, prototypes, global_features)
                loss = bce + prototype_lambda * pa
                bce_total += bce.item() * batch_count
                positive_exams = (target == 1).any(dim=1).sum().item()
                pa_total += pa.item() * positive_exams
            else:
                logits = model(slices, slice_counts)
                loss = loss_fn(logits, device_target)
            if training:
                loss.backward()
                optimizer.step()
            if not sip:
                loss_total += loss.item() * batch_count
            exam_count += batch_count
            labels.append(target)
            scores.append(logits.detach().float().sigmoid().cpu())
    y = torch.cat(labels).numpy()
    p = torch.cat(scores).numpy()
    auc = {
        name: float(roc_auc_score(y[:, i], p[:, i])) if len(set(y[:, i])) == 2 else None
        for i, name in enumerate(LABEL_NAMES)
    }
    macro = sum(auc.values()) / len(auc) if None not in auc.values() else None
    metrics = {"loss": loss_total / exam_count, "auc": auc, "macro_auc": macro}
    if sip:
        # Report per-exam averages, with zero PA contribution for negative-only exams.
        metrics.update(bce_loss=bce_total / exam_count, pa_loss=pa_total / exam_count)
        metrics["loss"] = metrics["bce_loss"] + prototype_lambda * metrics["pa_loss"]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--metadata", type=Path, help="Defaults to the workbook inside data-root")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of exams per batch")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--degrees", type=float, default=10)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--no-cbam", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--mode", choices=("baseline", "sip"), default="baseline")
    parser.add_argument("--prototype-samples", type=int, default=10)
    parser.add_argument("--prototype-lambda", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    checkpoint_args = parser.add_mutually_exclusive_group()
    checkpoint_args.add_argument("--evaluate", type=Path, help="Evaluate a saved checkpoint on the validation split")
    checkpoint_args.add_argument("--resume", type=Path, help="Continue training from a checkpoint into a new output directory")
    args = parser.parse_args()
    if args.epochs < 1 or args.size < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs, size and batch-size must be positive and workers must be nonnegative")
    if args.prototype_samples < 1 or not math.isfinite(args.prototype_lambda) or args.prototype_lambda < 0:
        parser.error("prototype-samples must be positive and prototype-lambda must be finite and nonnegative")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    metadata = args.metadata or args.data_root / "MRNet_Multilabel_Metadata.xlsx"
    train, val = split_records(read_records(metadata))
    checkpoint_path = args.evaluate or args.resume
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if checkpoint_path else None
    config = checkpoint["model_config"] if checkpoint else {
        "cbam": not args.no_cbam, "dropout": args.dropout,
    }
    mode = checkpoint.get("mode", "baseline") if checkpoint else args.mode
    prototype_lambda = checkpoint.get("prototype_lambda", 0.01) if checkpoint else args.prototype_lambda
    if mode == "sip" and checkpoint is None:
        config["prototype_samples"] = args.prototype_samples
    size = checkpoint["size"] if checkpoint else args.size
    val_loader = DataLoader(
        MRNetDataset(val, args.data_root, size), batch_size=args.batch_size,
        num_workers=args.workers, collate_fn=collate_packed_views,
    )
    model_cls = SIPResNetCBAM if mode == "sip" else ResNetCBAM
    model = model_cls(pretrained=checkpoint is None and not args.no_pretrained, **config).to(device)
    targets = torch.tensor([row["target"] for row in train])
    positive = targets.sum(dim=0)
    negative = len(train) - positive
    if torch.any((positive == 0) | (negative == 0)):
        raise ValueError("Each training label must contain both classes")
    # BCE has separate weights for positive and negative targets, with mean weight 1 per label.
    loss_fn = nn.BCEWithLogitsLoss(
        weight=(len(train) / (2 * negative)).to(device),
        pos_weight=(negative / positive).to(device),
    )
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    if args.evaluate:
        print(json.dumps(run_epoch(model, val_loader, loss_fn, device, prototype_lambda=prototype_lambda)))
        return

    train_loader = DataLoader(
        MRNetDataset(train, args.data_root, size, True, args.degrees, args.translate),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=collate_packed_views,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    completed = checkpoint["epoch"] if args.resume else 0
    if completed >= args.epochs:
        parser.error("--epochs is the total target epoch and must exceed the saved epoch")
    legacy = args.resume and "optimizer" not in checkpoint
    if legacy:
        print("Legacy checkpoint: restoring weights and cosine position with fresh Adam state.", flush=True)
        for group in optimizer.param_groups:
            group["initial_lr"] = args.lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, last_epoch=completed - 1 if legacy else -1,
    )
    if args.resume and not legacy:
        if checkpoint["scheduler"]["T_max"] != args.epochs:
            parser.error("--epochs must match the saved cosine schedule")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    default_output = Path("runs/resnet_cbam_sip" if mode == "sip" else "runs/resnet_cbam")
    output = args.output or (default_output.with_name(default_output.name + "_resume") if args.resume else default_output)
    if args.resume and any((output / name).exists() for name in ("metrics.jsonl", "last.pt", "best.pt")):
        parser.error("Choose a new --output directory to preserve previous logs and checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    best = float("-inf")
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(completed + 1, args.epochs + 1):
            metrics = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": run_epoch(model, train_loader, loss_fn, device, optimizer, prototype_lambda),
                "val": run_epoch(model, val_loader, loss_fn, device, prototype_lambda=prototype_lambda),
            }
            scheduler.step()
            line = json.dumps(metrics)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            state = {"model": model.state_dict(), "model_config": config, "size": size, "epoch": epoch}
            state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            if mode == "sip":
                state.update(mode=mode, prototype_lambda=prototype_lambda)
            torch.save(state, output / "last.pt")
            auc = metrics["val"]["macro_auc"]
            if auc is not None and auc > best:
                best = auc
                torch.save(state, output / "best.pt")


if __name__ == "__main__":
    main()
