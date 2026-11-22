"""Evaluate a KAMRNet DICOM Baseline or SIP checkpoint on the workbook validation split."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.kamrnet_dicom import DEFAULT_DATA_ROOT, LABEL_CODES, LABEL_NAMES, KAMRNetDicomDataset, read_splits
from data.packed_views import collate_packed_views
from evaluate_mrnet import compare_logged_auc, summarize
from model import SIPResNetCBAM
from train_kamrnet_dicom import KAMRNetResNetCBAM


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
    config = dict(checkpoint["model_config"])
    if tuple(checkpoint.get("label_codes", ())) != LABEL_CODES or mode not in ("baseline", "sip"):
        parser.error("Expected a KAMRNet DICOM Baseline or SIP checkpoint with matching label codes")
    model_type = KAMRNetResNetCBAM
    if mode == "sip":
        model_type = SIPResNetCBAM
        config.setdefault("num_labels", len(LABEL_CODES))
    model = model_type(pretrained=False, **config)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()

    metadata = args.metadata or args.data_root / "data_split.xlsx"
    _, val = read_splits(metadata)
    dataset = KAMRNetDicomDataset(val, args.data_root, size=checkpoint["size"])
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
    metrics = {"epoch": checkpoint["epoch"], "mode": mode, **summarize(y, p, args.threshold, LABEL_NAMES)}
    log_path = args.reference_log or args.checkpoint.with_name("metrics.jsonl")
    metrics["auc_verification"] = compare_logged_auc(metrics, log_path)
    report = json.dumps(metrics, indent=2)
    print(report)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "metrics.json").write_text(report + "\n", encoding="utf-8")
        with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["exam_id"] + [
                f"{name}_{suffix}" for name in LABEL_NAMES for suffix in ("target", "probability", "prediction")
            ])
            for patient, target, score in zip(ids, y, p):
                writer.writerow([patient] + [
                    value for truth, prob in zip(target, score)
                    for value in (truth, prob, int(prob >= args.threshold))
                ])


if __name__ == "__main__":
    main()
