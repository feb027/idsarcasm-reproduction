#!/usr/bin/env python3
"""Generate Progress 5 analysis tables and figures.

Outputs are intentionally small and committed to the repo so the academic report
can be verified without rerunning Colab/LM Studio experiments.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
OPT_DIR = ROOT / "results" / "optimization"
MODERN_DIR = ROOT / "results" / "modern_llm"
ERROR_DIR = ROOT / "results" / "progress5_error_analysis"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_pipeline_architecture() -> None:
    """Draw the two-lane Progress 5 workflow: transformer optimization and modern local LLM experiments."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.8, 6.2))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(
        6.9,
        5.72,
        "Arsitektur Pipeline Progress 5: Optimasi XLM-R dan Eksperimen LLM Lokal",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
    )

    lanes = [
        (
            "Jalur A: Optimasi Transformer XLM-R",
            3.62,
            [
                (0.35, "Model\nTerbaik", "XLM-R Large\nProgress 3", "#dbeafe", "#1d4ed8"),
                (2.35, "Prediksi\nProbabilitas", "validation + test\npredictions.csv", "#e0f2fe", "#0369a1"),
                (4.35, "Pencarian\nThreshold", "dipilih dari\nF1 validasi", "#dcfce7", "#15803d"),
                (6.35, "Evaluasi\nData Uji", "threshold 0,5 vs\nthreshold terpilih", "#fef3c7", "#b45309"),
                (8.35, "Transisi\nError", "membaik, memburuk,\ntetap salah", "#fee2e2", "#b91c1c"),
                (10.35, "Artefak\nFinal", "metrik, tabel,\ngambar, analisis", "#ede9fe", "#6d28d9"),
            ],
        ),
        (
            "Jalur B: Eksperimen Modern Local LLM",
            1.36,
            [
                (0.35, "Twitter\nTest Set", "538 data\nevaluasi ringan", "#f1f5f9", "#475569"),
                (2.35, "LM Studio\nLocal API", "endpoint localhost\nOpenAI-compatible", "#ffedd5", "#c2410c"),
                (4.35, "Model\nGGUF", "Qwen3.5-4B +\nGemma 4 E4B", "#fef3c7", "#b45309"),
                (6.35, "Mode\nPrompting", "zero-shot +\nfew-shot", "#dcfce7", "#15803d"),
                (8.35, "Parsing\nLabel", "sarcastic vs\nnot sarcastic", "#e0e7ff", "#4338ca"),
                (10.35, "Pembanding\nPraktis", "dibandingkan dengan\nBLOOMZ/mT0 + XLM-R", "#ede9fe", "#6d28d9"),
            ],
        ),
    ]

    arrow = dict(arrowstyle="->", color="#64748b", lw=1.75, shrinkA=5, shrinkB=5)
    for lane_title, y, stages in lanes:
        ax.text(0.38, y + 1.24, lane_title, ha="left", va="center", fontsize=10.2, fontweight="bold", color="#0f172a")
        for i, (x, title, desc, fc, ec) in enumerate(stages):
            rect = mpatches.FancyBboxPatch(
                (x, y),
                1.55,
                1.02,
                boxstyle="round,pad=0.12,rounding_size=0.11",
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.45,
            )
            ax.add_patch(rect)
            ax.text(x + 0.775, y + 0.66, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color=ec)
            ax.text(x + 0.775, y - 0.27, desc, ha="center", va="center", fontsize=7.8, color="#475569")
            if i < len(stages) - 1:
                ax.annotate("", xy=(stages[i + 1][0], y + 0.51), xytext=(x + 1.58, y + 0.51), arrowprops=arrow)

    ax.text(
        6.9,
        0.22,
        "Progress 5 memisahkan optimasi utama transformer dari eksperimen pembanding LLM lokal agar hasilnya tidak dicampur sebagai klaim yang sama.",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#64748b",
    )
    plt.tight_layout()
    plt.savefig(FIGURES / "progress5_pipeline_architecture.png", dpi=180)
    plt.close()



def generate_transformer_summary() -> None:
    rows = read_csv(TABLES / "optimization_runs.csv")
    summary: list[dict[str, Any]] = []
    for row in rows:
        summary.append(
            {
                "run_id": row["run_id"],
                "dataset": row["dataset"],
                "model_alias": row["model_alias"],
                "learning_rate": row["learning_rate"],
                "max_length": row["max_length"],
                "weight_decay": row["weight_decay"],
                "label_smoothing_factor": row["label_smoothing_factor"],
                "selected_threshold": row["selected_threshold"],
                "test_default_precision": row["test_default_precision"],
                "test_default_recall": row["test_default_recall"],
                "test_default_f1": row["test_default_f1"],
                "test_tuned_precision": row["test_tuned_precision"],
                "test_tuned_recall": row["test_tuned_recall"],
                "test_tuned_f1": row["test_tuned_f1"],
                "delta_f1": row["delta_f1"],
            }
        )
    write_csv(TABLES / "progress5_transformer_optimization_summary.csv", summary)

    xlmr_large = [r for r in rows if r["run_id"] in {"twitter-xlmr-large-threshold", "reddit-xlmr-large-threshold"}]
    labels = ["Twitter\nXLM-R Large", "Reddit\nXLM-R Large"]
    xlmr_large = sorted(xlmr_large, key=lambda r: 0 if r["dataset"] == "twitter" else 1)
    default = [f(r, "test_default_f1") for r in xlmr_large]
    tuned = [f(r, "test_tuned_f1") for r in xlmr_large]

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    x = range(len(labels))
    width = 0.34
    plt.bar([i - width / 2 for i in x], default, width, label="Default threshold", color="#9CA3AF")
    plt.bar([i + width / 2 for i in x], tuned, width, label="Tuned threshold", color="#2563EB")
    for i, (d, t) in enumerate(zip(default, tuned)):
        plt.text(i - width / 2, d + 0.01, f"{d:.4f}", ha="center", fontsize=9)
        plt.text(i + width / 2, t + 0.01, f"{t:.4f}", ha="center", fontsize=9)
    plt.xticks(list(x), labels)
    plt.ylim(0, max(tuned + default) + 0.1)
    plt.ylabel("F1-score")
    plt.title("Progress 5: XLM-R Large Sebelum dan Sesudah Threshold Tuning")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "progress5_threshold_tuning_f1.png", dpi=180)
    plt.close()

    twitter_screen = [r for r in rows if r["dataset"] == "twitter" and r["model_alias"] == "xlmr-base"]
    twitter_screen = sorted(twitter_screen, key=lambda r: f(r, "test_tuned_f1"), reverse=True)
    labels = [r["run_id"].replace("twitter-xlmr-base-", "") for r in twitter_screen]
    values = [f(r, "test_tuned_f1") for r in twitter_screen]
    plt.figure(figsize=(9.5, 4.8))
    bars = plt.bar(labels, values, color="#10B981")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, max(values) + 0.12)
    plt.ylabel("F1-score setelah threshold tuning")
    plt.title("Screening Hyperparameter XLM-R Base pada Twitter")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "progress5_xlmr_base_screening.png", dpi=180)
    plt.close()


def generate_modern_llm_summary() -> None:
    rows = read_csv(TABLES / "modern_llm_experiments.csv")
    summary: list[dict[str, Any]] = []
    for row in rows:
        summary.append(
            {
                "dataset": row["dataset"],
                "model_alias": row["model_alias"],
                "mode": row["mode"],
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "invalid_outputs": row["invalid_outputs"],
                "num_examples": row["num_examples"],
                "runtime_minutes": round(f(row, "runtime_seconds") / 60, 2),
                "avg_latency_seconds": row["avg_latency_seconds"],
            }
        )
    write_csv(TABLES / "progress5_modern_llm_summary.csv", summary)

    labels = [f"{r['model_alias'].replace('-gguf', '')}\n{r['mode']}" for r in rows]
    f1_values = [f(r, "f1") for r in rows]
    colors = ["#7C3AED" if "qwen" in r["model_alias"] else "#F97316" for r in rows]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, f1_values, color=colors)
    for bar, value in zip(bars, f1_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center", fontsize=9)
    plt.axhline(0.3989, color="#6B7280", linestyle="--", linewidth=1, label="Best zero-shot paper reproduction Twitter (mT0 Large)")
    plt.axhline(0.7649, color="#2563EB", linestyle="--", linewidth=1, label="Optimized XLM-R Large Twitter")
    plt.ylim(0, 0.85)
    plt.ylabel("F1-score")
    plt.title("Progress 5: Modern Local LLM vs Baseline Utama pada Twitter")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "progress5_modern_llm_f1_comparison.png", dpi=180)
    plt.close()

    labels = [f"{r['model_alias'].replace('-gguf', '')}\n{r['mode']}" for r in rows]
    precision = [f(r, "precision") for r in rows]
    recall = [f(r, "recall") for r in rows]
    x = range(len(labels))
    width = 0.36
    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], precision, width, color="#0EA5E9", label="Precision")
    plt.bar([i + width / 2 for i in x], recall, width, color="#EF4444", label="Recall")
    plt.xticks(list(x), labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Skor")
    plt.title("Precision dan Recall Modern Local LLM pada Twitter")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "progress5_modern_llm_precision_recall.png", dpi=180)
    plt.close()


def generate_error_analysis() -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    for run_id in ["twitter-xlmr-large-threshold", "reddit-xlmr-large-threshold"]:
        rows = read_csv(OPT_DIR / run_id / "predictions.csv")
        test_rows = [r for r in rows if r["split"] == "test"]
        counter = Counter()
        for r in test_rows:
            if r["default_correct"] == "0" and r["tuned_correct"] == "1":
                counter["improved"] += 1
            elif r["default_correct"] == "1" and r["tuned_correct"] == "0":
                counter["worsened"] += 1
            elif r["default_correct"] == "1" and r["tuned_correct"] == "1":
                counter["both_correct"] += 1
            else:
                counter["both_wrong"] += 1
        summary = [
            {"run_id": run_id, "category": key, "count": counter[key]}
            for key in ["improved", "worsened", "both_correct", "both_wrong"]
        ]
        write_csv(ERROR_DIR / f"{run_id}_transition_summary.csv", summary)
        examples: list[dict[str, Any]] = []
        for category, predicate in [
            ("improved", lambda r: r["default_correct"] == "0" and r["tuned_correct"] == "1"),
            ("worsened", lambda r: r["default_correct"] == "1" and r["tuned_correct"] == "0"),
            ("both_wrong", lambda r: r["default_correct"] == "0" and r["tuned_correct"] == "0"),
        ]:
            selected = [r for r in test_rows if predicate(r)][:12]
            for r in selected:
                examples.append(
                    {
                        "run_id": run_id,
                        "category": category,
                        "sample_idx": r["sample_idx"],
                        "true_label": r["true_label"],
                        "prob_sarcastic": r["prob_sarcastic"],
                        "pred_default": r["pred_default"],
                        "pred_tuned": r["pred_tuned"],
                        "text": r["text"],
                    }
                )
        write_csv(ERROR_DIR / f"{run_id}_examples.csv", examples)

    # Combined transition plot
    summaries = []
    for run_id in ["twitter-xlmr-large-threshold", "reddit-xlmr-large-threshold"]:
        rows = read_csv(ERROR_DIR / f"{run_id}_transition_summary.csv")
        summaries.append({r["category"]: int(r["count"]) for r in rows})
    labels = ["Twitter", "Reddit"]
    categories = ["improved", "worsened", "both_wrong"]
    colors = ["#22C55E", "#EF4444", "#6B7280"]
    x = range(len(labels))
    bottom = [0, 0]
    plt.figure(figsize=(8.5, 5))
    for cat, color in zip(categories, colors):
        values = [s.get(cat, 0) for s in summaries]
        plt.bar(list(x), values, bottom=bottom, color=color, label=cat)
        bottom = [b + v for b, v in zip(bottom, values)]
    plt.xticks(list(x), labels)
    plt.ylabel("Jumlah contoh test")
    plt.title("Dampak Threshold Tuning pada Contoh Salah/Benar XLM-R Large")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "progress5_threshold_error_transitions.png", dpi=180)
    plt.close()


def main() -> None:
    generate_pipeline_architecture()
    generate_transformer_summary()
    generate_modern_llm_summary()
    generate_error_analysis()
    print("Generated Progress 5 analysis artifacts")


if __name__ == "__main__":
    main()
