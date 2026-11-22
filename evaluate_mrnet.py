"""Evaluate an MRNet baseline or SIP checkpoint on the workbook's test split."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from data.mrnet_multilabel import DEFAULT_DATA_ROOT, LABEL_NAMES, MRNetDataset, read_records, split_records
from data.packed_views import collate_packed_views
from model import ResNetCBAM, SIPResNetCBAM


def binary_metrics(y, p, threshold):
    pred = p >= threshold
    positive = y == 1
    tp = int((positive & pred).sum())
    tn = int((~positive & ~pred).sum())
    fp = int((~positive & pred).sum())
    fn = int((positive & ~pred).sum())
    return {
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if tp + fn else None,
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def summarize(y, p, threshold, label_names=LABEL_NAMES):
    per_label = {name: binary_metrics(y[:, i], p[:, i], threshold) for i, name in enumerate(label_names)}
    macro = {}
    for key in ("roc_auc", "average_precision", "accuracy", "precision", "sensitivity", "specificity", "f1"):
        values = [metrics[key] for metrics in per_label.values() if metrics[key] is not None]
        macro[key] = sum(values) / len(values) if values else None
    errors = (p >= threshold) != y
    return {
        "num_exams": len(y),
        "threshold": threshold,
        "per_label": per_label,
        "macro": macro,
        "micro": binary_metrics(y.ravel(), p.ravel(), threshold),
        "subset_accuracy": float((~errors.any(axis=1)).mean()),
        "hamming_loss": float(errors.mean()),
    }


def compare_logged_auc(metrics, log_path):
    if not log_path.is_file():
        return {"status": "log_not_found", "log": str(log_path)}
    with log_path.open(encoding="utf-8") as log:
        reference = next(
            (row["val"] for row in map(json.loads, filter(str.strip, log))
             if row["epoch"] == metrics["epoch"]),
            None,
        )
    if reference is None:
        return {"status": "epoch_not_found", "log": str(log_path)}
    tolerance = 1e-6
    pairs = {"macro": (metrics["macro"]["roc_auc"], reference["macro_auc"])}
    pairs.update({
        name: (metrics["per_label"][name]["roc_auc"], reference["auc"].get(name))
        for name in metrics["per_label"]
    })
    comparisons = {}
    for name, (computed, logged) in pairs.items():
        difference = computed - logged if computed is not None and logged is not None else None
        comparisons[name] = {
            "recomputed": computed,
            "logged": logged,
            "delta": difference,
        }
    if any(comp is None or logged is None for comp, logged in pairs.values()):
        status = "incomplete"
    else:
        status = "match" if all(abs(c["delta"]) <= tolerance for c in comparisons.values()) else "mismatch"
    return {"status": status, "log": str(log_path), "tolerance": tolerance, "auc": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--reference-log", type=Path, help="Defaults to metrics.jsonl beside the checkpoint")
    parser.add_argument("--output", type=Path, help="Optional directory for metrics.json and predictions.csv")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or not 0 < args.threshold < 1:
        parser.error("Use a positive batch size, nonnegative workers and a threshold between 0 and 1")
    if args.output and any((args.output / name).exists() for name in ("metrics.json", "predictions.csv")):
        parser.error("--output already contains an evaluation report")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    mode = checkpoint.get("mode", "baseline")
    config = checkpoint["model_config"]
    if "label_codes" in checkpoint or mode not in ("baseline", "sip"):
        parser.error("Expected an MRNet baseline or SIP checkpoint")
    model_type = SIPResNetCBAM if mode == "sip" else ResNetCBAM
    model = model_type(pretrained=False, **config)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()

    metadata = args.metadata or args.data_root / "MRNet_Multilabel_Metadata.xlsx"
    _, val = split_records(read_records(metadata))
    dataset = MRNetDataset(val, args.data_root, size=checkpoint["size"])
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        collate_fn=collate_packed_views)
    ids, targets, probabilities = [], [], []
    with torch.inference_mode():
        for slices, counts, target, exam_ids in loader:
            logits = model(slices.to(args.device), counts)
            ids.extend(exam_ids)
            targets.append(target)
            probabilities.append(logits.float().sigmoid().cpu())
    y = torch.cat(targets).numpy().astype(int)
    p = torch.cat(probabilities).numpy()
    metrics = {"epoch": checkpoint["epoch"], "mode": mode, **summarize(y, p, args.threshold)}
    log_path = args.reference_log or args.checkpoint.with_name("metrics.jsonl")
    metrics["auc_verification"] = compare_logged_auc(metrics, log_path)
    report = json.dumps(metrics, indent=2)
    print(report)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "metrics.json").write_text(report + "\n", encoding="utf-8")
        with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id"] + [
                f"{name}_{suffix}" for name in LABEL_NAMES for suffix in ("target", "probability", "prediction")
            ])
            for patient, target, score in zip(ids, y, p):
                writer.writerow([patient] + [
                    value for truth, prob in zip(target, score)
                    for value in (truth, prob, int(prob >= args.threshold))
                ])


if __name__ == "__main__":
    main()
