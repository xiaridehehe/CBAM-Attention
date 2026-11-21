import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from data.kamrnet_dicom import DEFAULT_DATA_ROOT, LABEL_CODES, LABEL_NAMES, KAMRNetDicomDataset, read_splits
from model import ResNetCBAM


class KAMRNetResNetCBAM(ResNetCBAM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = nn.Linear(self.classifier.in_features, len(LABEL_CODES))


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, labels, scores = [], [], []
    with torch.set_grad_enabled(training):
        for views, target, _ in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model([view[0].to(device) for view in views])
            loss = loss_fn(logits, target.to(device))
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
    return {"loss": sum(losses) / len(losses), "auc": auc, "macro_auc": macro}


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
    parser.add_argument("--output", type=Path, default=Path("runs/kamrnet_dicom_cbam"))
    checkpoint_args = parser.add_mutually_exclusive_group()
    checkpoint_args.add_argument("--evaluate", type=Path, help="Evaluate a saved checkpoint on the validation split")
    checkpoint_args.add_argument("--resume", type=Path, help="Continue from a saved checkpoint into a new output directory")
    args = parser.parse_args()
    if args.epochs < 1 or args.size < 1 or args.workers < 0:
        parser.error("epochs and size must be positive and workers must be nonnegative")

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
    size = checkpoint["size"] if checkpoint else args.size
    val_loader = DataLoader(KAMRNetDicomDataset(val, args.data_root, size), batch_size=1, num_workers=args.workers)
    model = KAMRNetResNetCBAM(pretrained=checkpoint is None and not args.no_pretrained, **config).to(device)
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
        print(json.dumps(run_epoch(model, val_loader, loss_fn, device)))
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
    if args.resume and (args.output / "metrics.jsonl").exists():
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
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("-inf")
    with (args.output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(completed + 1, args.epochs + 1):
            metrics = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": run_epoch(model, train_loader, loss_fn, device, optimizer),
                "val": run_epoch(model, val_loader, loss_fn, device),
            }
            scheduler.step()
            line = json.dumps(metrics)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            state = {"model": model.state_dict(), "model_config": config, "label_codes": LABEL_CODES, "size": size, "epoch": epoch}
            state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            torch.save(state, args.output / "last.pt")
            auc = metrics["val"]["macro_auc"]
            if auc is not None and auc > best:
                best = auc
                torch.save(state, args.output / "best.pt")


if __name__ == "__main__":
    main()
