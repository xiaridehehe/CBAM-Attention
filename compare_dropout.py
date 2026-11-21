"""Compare validation metrics from baseline and dropout training logs."""
import argparse
import json
from pathlib import Path


def read_metrics(path):
    with path.open(encoding="utf-8") as log:
        rows = [json.loads(line) for line in log if line.strip()]
    return {row["epoch"]: row["val"] for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("runs/resnet_cbam/metrics.jsonl"))
    parser.add_argument("--dropout", type=Path, default=Path("runs/resnet_cbam_dropout/metrics.jsonl"))
    args = parser.parse_args()
    try:
        baseline = read_metrics(args.baseline)
        experiment = read_metrics(args.dropout)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    epochs = sorted(baseline.keys() & experiment.keys())
    if not epochs:
        parser.error("The logs have no completed epochs in common.")
    auc_epochs = [
        epoch for epoch in epochs
        if all(run[epoch]["macro_auc"] is not None for run in (baseline, experiment))
    ]
    print(f"Common epochs: {len(epochs)} (latest: {epochs[-1]})")
    print("| Selection | Run | Epoch | Validation macro AUC | Validation weighted BCE |")
    print("|---|---|---:|---:|---:|")
    for name, run in (("Baseline", baseline), ("Dropout", experiment)):
        selections = [
            ("Lowest BCE", min(epochs, key=lambda epoch: run[epoch]["loss"])),
            ("Latest common epoch", epochs[-1]),
        ]
        if auc_epochs:
            selections.insert(0, ("Highest AUC", max(auc_epochs, key=lambda epoch: run[epoch]["macro_auc"])))
        for selection, epoch in selections:
            metrics = run[epoch]
            auc = metrics["macro_auc"]
            formatted_auc = f"{auc:.6f}" if auc is not None else "N/A"
            print(f"| {selection} | {name} | {epoch} | {formatted_auc} | {metrics['loss']:.6f} |")


if __name__ == "__main__":
    main()
