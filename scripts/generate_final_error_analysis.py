#!/usr/bin/env python3
"""Generate final confusion matrices and compact error examples.

This script reads final prediction CSVs from the selected final runs and writes
small artifacts for README/docs. It does not train or modify model outputs.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

RUNS: dict[str, dict[str, Any]] = {
    "twitter": {
        "label": "Twitter",
        "path": ROOT / "results" / "optimization" / "twitter-xlmr-large-lr2e-5-len128" / "predictions.csv",
        "strategy": "default",
        "pred_col": "pred_default",
        "correct_col": "default_correct",
        "model": "XLM-R Large lr=2e-5",
    },
    "reddit": {
        "label": "Reddit",
        "path": ROOT / "results" / "optimization" / "reddit-xlmr-large-threshold" / "predictions.csv",
        "strategy": "tuned",
        "pred_col": "pred_tuned",
        "correct_col": "tuned_correct",
        "model": "XLM-R Large threshold tuning",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_text(value: str, limit: int = 220) -> str:
    text = " ".join(html.unescape(value).split())
    return text[: limit - 1] + "…" if len(text) > limit else text


def metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def build_artifacts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []

    for dataset, config in RUNS.items():
        rows = [row for row in read_csv(config["path"]) if row["split"] == "test"]
        tn = fp = fn = tp = 0
        false_positives: list[dict[str, str]] = []
        false_negatives: list[dict[str, str]] = []

        for row in rows:
            true_label = int(row["true_label"])
            pred = int(row[config["pred_col"]])
            if true_label == 0 and pred == 0:
                tn += 1
            elif true_label == 0 and pred == 1:
                fp += 1
                false_positives.append(row)
            elif true_label == 1 and pred == 0:
                fn += 1
                false_negatives.append(row)
            elif true_label == 1 and pred == 1:
                tp += 1

        metrics = metrics_from_counts(tn, fp, fn, tp)
        matrix_rows.append(
            {
                "dataset": dataset,
                "dataset_label": config["label"],
                "model": config["model"],
                "strategy": config["strategy"],
                "test_rows": len(rows),
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
                **metrics,
            }
        )

        # Pick interpretable examples: high-confidence FP and low-probability FN.
        ranked_fp = sorted(false_positives, key=lambda row: float(row["prob_sarcastic"]), reverse=True)[:5]
        ranked_fn = sorted(false_negatives, key=lambda row: float(row["prob_sarcastic"]))[:5]
        for error_type, selected in [("false_positive", ranked_fp), ("false_negative", ranked_fn)]:
            for rank, row in enumerate(selected, 1):
                example_rows.append(
                    {
                        "dataset": dataset,
                        "error_type": error_type,
                        "rank": rank,
                        "sample_idx": row["sample_idx"],
                        "true_label": "sarcastic" if int(row["true_label"]) == 1 else "non_sarcastic",
                        "predicted_label": "sarcastic" if int(row[config["pred_col"]]) == 1 else "non_sarcastic",
                        "prob_sarcastic": round(float(row["prob_sarcastic"]), 4),
                        "strategy": config["strategy"],
                        "text": safe_text(row["text"]),
                    }
                )

    return matrix_rows, example_rows


def plot_confusion_matrices(matrix_rows: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    for ax, row in zip(axes, matrix_rows):
        matrix = [
            [row["true_negative"], row["false_positive"]],
            [row["false_negative"], row["true_positive"]],
        ]
        total = row["test_rows"]
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(max(r) for r in matrix))
        ax.set_title(f"{row['dataset_label']} — {row['strategy']} strategy\nF1={row['f1']:.4f} · n={total:,}")
        ax.set_xticks([0, 1], labels=["Predicted\nnon-sarcastic", "Predicted\nsarcastic"])
        ax.set_yticks([0, 1], labels=["Actual\nnon-sarcastic", "Actual\nsarcastic"])
        for i in range(2):
            for j in range(2):
                value = matrix[i][j]
                pct = value / total * 100 if total else 0.0
                color = "white" if value > max(max(r) for r in matrix) * 0.55 else "#0F172A"
                ax.text(
                    j,
                    i,
                    f"{value:,}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color=color,
                )
        ax.set_xlabel("Model prediction")
        ax.set_ylabel("Ground truth")
    fig.suptitle("Final Confusion Matrices for Selected IdSarcasm Runs", fontsize=14, fontweight="bold")
    plt.savefig(FIGURES / "final_confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    matrix_rows, example_rows = build_artifacts()
    write_csv(TABLES / "final_confusion_matrices.csv", matrix_rows)
    write_csv(TABLES / "final_error_examples.csv", example_rows)
    plot_confusion_matrices(matrix_rows)
    print("Generated final confusion matrix and error-analysis artifacts")


if __name__ == "__main__":
    main()
