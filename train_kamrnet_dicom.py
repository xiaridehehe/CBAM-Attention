import argparse
import json
import math
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from data.kamrnet_dicom import DEFAULT_DATA_ROOT, LABEL_CODES, LABEL_NAMES, KAMRNetDicomDataset, read_splits
from model import ResNetCBAM, SIPResNetCBAM, prototype_aggregation_loss

DEFAULT_PROTOTYPE_LAMBDA = 0.0001


class KAMRNetResNetCBAM(ResNetCBAM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = nn.Linear(self.classifier.in_features, len(LABEL_CODES))


def run_epoch(model, loader, loss_fn, device, optimizer=None, prototype_lambda=DEFAULT_PROTOTYPE_LAMBDA):
    training = optimizer is not None
    model.train(training)
    losses, labels, scores = [], [], []
    sip = isinstance(model, SIPResNetCBAM)
    bce_total = pa_total = 0.0
    with torch.set_grad_enabled(training):
        for views, target, _ in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            views = [view[0].to(device) for view in views]
            device_target = target.to(device)
            if sip:
                logits, prototypes, global_features, _ = model(views, return_aux=True)
                bce = loss_fn(logits, device_target)
                pa = prototype_aggregation_loss(device_target, prototypes, global_features)
                loss = bce + prototype_lambda * pa
                bce_total += bce.item()
                pa_total += pa.item()
            else:
                logits = model(views)
                loss = loss_fn(logits, device_target)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            labels.append(target)
            scores.append(logits.detach().float().sigmoid().cpu())
    y = torch.cat(labels).numpy()
    p = torch.cat(scores).numpy()
    auc = {
        name: float(roc_auc_score(y[:, i], p[:, i])) if len(set(y[:, i])) == 2 else None
        for i, name in enumerate(LABEL_NAMES)
    }
    valid_auc = [value for value in auc.values() if value is not None]
    macro = sum(valid_auc) / len(valid_auc) if valid_auc else None
    metrics = {"loss": sum(losses) / len(losses), "auc": auc, "macro_auc": macro}
    if sip:
        metrics.update(bce_loss=bce_total / len(losses), pa_loss=pa_total / len(losses))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--metadata", type=Path, help="Defaults to the workbook inside data-root")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--degrees", type=float, default=10)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--no-cbam", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--mode", choices=("baseline", "sip"))
    parser.add_argument("--prototype-samples", type=int, default=10)
    parser.add_argument("--prototype-lambda", type=float, default=DEFAULT_PROTOTYPE_LAMBDA)
    parser.add_argument("--output", type=Path)
    checkpoint_args = parser.add_mutually_exclusive_group()
    checkpoint_args.add_argument("--evaluate", type=Path, help="Evaluate a saved checkpoint on the validation split")
    checkpoint_args.add_argument("--resume", type=Path, help="Continue from a saved checkpoint into a new output directory")
    args = parser.parse_args()
    if args.epochs < 1 or args.size < 1 or args.workers < 0:
        parser.error("epochs and size must be positive and workers must be nonnegative")
    if args.prototype_samples < 1 or not math.isfinite(args.prototype_lambda) or args.prototype_lambda < 0:
        parser.error("prototype-samples must be positive and prototype-lambda must be finite and nonnegative")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    metadata = args.metadata or args.data_root / "data_split.xlsx"
    train, val = read_splits(metadata)
    checkpoint_path = args.evaluate or args.resume
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if checkpoint_path else None
    if checkpoint is not None and checkpoint.get("label_codes") != LABEL_CODES:
        parser.error("Expected a KAMRNet DICOM checkpoint with matching label codes")
    config = checkpoint["model_config"] if checkpoint else {
        "cbam": not args.no_cbam, "dropout": args.dropout,
    }
    mode = checkpoint.get("mode", "baseline") if checkpoint else args.mode or "baseline"
    if args.mode is not None and args.mode != mode:
        parser.error("--mode must match the checkpoint")
    prototype_lambda = checkpoint["prototype_lambda"] if checkpoint and mode == "sip" else args.prototype_lambda
    if mode == "sip" and checkpoint is None:
        config["prototype_samples"] = args.prototype_samples
    model_cls = KAMRNetResNetCBAM
    if mode == "sip":
        model_cls = SIPResNetCBAM
        config["num_labels"] = len(LABEL_CODES)
    size = checkpoint["size"] if checkpoint else args.size
    val_loader = DataLoader(KAMRNetDicomDataset(val, args.data_root, size), batch_size=1, num_workers=args.workers)
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
        KAMRNetDicomDataset(train, args.data_root, size, True, args.degrees, args.translate),
        # Native slice counts vary by patient and the model fuses one patient at a time.
        batch_size=1, shuffle=True, num_workers=args.workers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    completed = checkpoint["epoch"] if args.resume else 0
    if completed >= args.epochs:
        parser.error("--epochs is the total target epoch and must exceed the saved epoch")
    default_name = "kamrnet_dicom_sip" if mode == "sip" else "kamrnet_dicom_cbam"
    if args.resume:
        default_name += "_resume"
    output = args.output or Path("runs") / default_name
    if args.resume and any((output / name).exists() for name in ("metrics.jsonl", "last.pt", "best.pt")):
        parser.error("Choose a new --output directory to preserve previous metrics and checkpoints")
    if args.resume and "optimizer" not in checkpoint:
        print("Legacy checkpoint: restoring weights and cosine schedule position with fresh Adam state.", flush=True)
        for group in optimizer.param_groups:
            group["initial_lr"] = args.lr
    legacy_epoch = completed - 1 if args.resume and "optimizer" not in checkpoint else -1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, last_epoch=legacy_epoch,
    )
    if args.resume and "optimizer" in checkpoint:
        if checkpoint["scheduler"]["T_max"] != args.epochs:
            parser.error("--epochs must match the saved cosine schedule")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
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
            state = {"model": model.state_dict(), "model_config": config, "label_codes": LABEL_CODES, "size": size, "epoch": epoch}
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
