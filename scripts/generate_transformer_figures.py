#!/usr/bin/env python3
"""Generate Progress 3 transformer figures for the IdSarcasm report.

This script intentionally uses only Python stdlib + matplotlib so it can run on
minimal VPS environments without pandas/numpy.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "figures"
TABLE_PATH = ROOT / "results" / "tables" / "transformer_baselines.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

MODEL_LABELS = {
    "indobert-base": "IndoBERT\nBase",
    "indobert-large": "IndoBERT\nLarge",
    "indobert-indolem-base": "IndoLEM\nBase",
    "mbert-base": "mBERT",
    "xlmr-base": "XLM-R\nBase",
    "xlmr-large": "XLM-R\nLarge",
}
MODEL_ORDER = [
    "indobert-base",
    "indobert-large",
    "indobert-indolem-base",
    "mbert-base",
    "xlmr-base",
    "xlmr-large",
]
PAPER_F1 = {
    ("reddit", "indobert-base"): 0.6100,
    ("reddit", "indobert-large"): 0.6184,
    ("reddit", "indobert-indolem-base"): 0.5671,
    ("reddit", "mbert-base"): 0.5338,
    ("reddit", "xlmr-base"): 0.5690,
    ("reddit", "xlmr-large"): 0.6274,
    ("twitter", "indobert-base"): 0.7273,
    ("twitter", "indobert-large"): 0.7160,
    ("twitter", "indobert-indolem-base"): 0.6462,
    ("twitter", "mbert-base"): 0.6467,
    ("twitter", "xlmr-base"): 0.7386,
    ("twitter", "xlmr-large"): 0.7692,
}
CLASSICAL_BEST = {
    "reddit": ("TF-IDF Logistic Regression", 0.4959),
    "twitter": ("BoW Logistic Regression", 0.7206),
}


def read_rows() -> list[dict[str, str]]:
    with TABLE_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_map(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(r["dataset"], r["model_alias"]): r for r in rows if r["sample_limited"] == "False"}


def save(fig: plt.Figure, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"✅ {path.relative_to(ROOT)}")


def generate_pipeline_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    stages = [
        (0.35, "Dataset\nIdSarcasm", "CSV train/valid/test\nTwitter + Reddit", "#dbeafe", "#1d4ed8"),
        (2.45, "Tokenizer\nTransformer", "AutoTokenizer\npad/truncate 128", "#fef3c7", "#b45309"),
        (4.55, "Pre-trained\nEncoder", "IndoBERT, mBERT,\nXLM-R", "#dcfce7", "#15803d"),
        (6.65, "Classifier\nHead", "linear head\n2 kelas", "#ede9fe", "#6d28d9"),
        (8.75, "Fine-tuning", "100 epoch max, cosine LR,\nearly stopping", "#fee2e2", "#b91c1c"),
        (10.85, "Evaluasi", "accuracy, precision,\nrecall, F1", "#e0f2fe", "#0369a1"),
    ]
    arrow = dict(arrowstyle="->", color="#64748b", lw=1.8, shrinkA=5, shrinkB=5)
    for i, (x, title, desc, fc, ec) in enumerate(stages):
        rect = mpatches.FancyBboxPatch(
            (x, 1.65),
            1.65,
            1.55,
            boxstyle="round,pad=0.13,rounding_size=0.12",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.6,
        )
        ax.add_patch(rect)
        ax.text(x + 0.825, 2.62, title, ha="center", va="center", fontsize=9.6, fontweight="bold", color=ec)
        ax.text(x + 0.825, 1.02, desc, ha="center", va="center", fontsize=8.1, color="#475569")
        if i < len(stages) - 1:
            ax.annotate("", xy=(stages[i + 1][0], 2.42), xytext=(x + 1.67, 2.42), arrowprops=arrow)

    ax.text(
        6.5,
        4.25,
        "Arsitektur Pipeline Fine-tuning Transformer Progress 3",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        6.5,
        0.25,
        "Output disimpan ke results/tables/transformer_baselines.csv, results/transformer/*/metrics.json, dan results/logs/progress-3-*.log",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#64748b",
    )
    save(fig, "transformer_pipeline_architecture.png")


def generate_f1_vs_paper(rows_by_key: dict[tuple[str, str], dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    x = list(range(len(MODEL_ORDER)))
    width = 0.34

    for ax, dataset, title in zip(axes, ["reddit", "twitter"], ["Dataset Reddit", "Dataset Twitter"]):
        paper = [PAPER_F1[(dataset, m)] for m in MODEL_ORDER]
        repro = [float(rows_by_key[(dataset, m)]["f1"]) for m in MODEL_ORDER]
        ax.bar([i - width / 2 for i in x], paper, width, label="Paper", color="#a78bfa", edgecolor="#6d28d9", linewidth=0.8)
        ax.bar([i + width / 2 for i in x], repro, width, label="Reproduksi", color="#34d399", edgecolor="#047857", linewidth=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=8)
        ax.set_ylim(0.48, 0.82)
        ax.set_ylabel("F1-score")
        ax.grid(axis="y", alpha=0.25)
        for i, (p, r) in enumerate(zip(paper, repro)):
            gap = r - p
            color = "#047857" if gap >= 0 else "#dc2626"
            ax.text(i, max(p, r) + 0.010, f"{gap:+.4f}", ha="center", va="bottom", fontsize=7.5, color=color)
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Perbandingan F1-score Transformer: Paper vs Reproduksi", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "transformer_f1_vs_paper.png")


def generate_gap_heatmap(rows_by_key: dict[tuple[str, str], dict[str, str]]) -> None:
    datasets = ["reddit", "twitter"]
    gaps = [[float(rows_by_key[(d, m)]["f1"]) - PAPER_F1[(d, m)] for d in datasets] for m in MODEL_ORDER]
    vmax = max(abs(v) for row in gaps for v in row)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    im = ax.imshow(gaps, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(["Reddit", "Twitter"])
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABELS[m].replace("\n", " ") for m in MODEL_ORDER])
    ax.set_title("Heatmap Selisih F1 Reproduksi terhadap Paper", fontsize=13, fontweight="bold", pad=12)
    for i, row in enumerate(gaps):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:+.4f}", ha="center", va="center", fontsize=9, color="#0f172a")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("F1 reproduksi - F1 paper")
    save(fig, "transformer_gap_heatmap.png")


def generate_best_classical_vs_transformer(rows_by_key: dict[tuple[str, str], dict[str, str]]) -> None:
    datasets = ["reddit", "twitter"]
    x = list(range(len(datasets)))
    width = 0.34
    classical = [CLASSICAL_BEST[d][1] for d in datasets]
    transformer = [float(rows_by_key[(d, "xlmr-large")]["f1"]) for d in datasets]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    b1 = ax.bar([i - width / 2 for i in x], classical, width, label="Best classical ML", color="#60a5fa", edgecolor="#1d4ed8", linewidth=0.8)
    b2 = ax.bar([i + width / 2 for i in x], transformer, width, label="Best transformer", color="#f97316", edgecolor="#c2410c", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Reddit", "Twitter"])
    ax.set_ylabel("F1-score")
    ax.set_ylim(0.45, 0.76)
    ax.set_title("Model Terbaik Classical ML vs Transformer", fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{bar.get_height():.4f}", ha="center", fontsize=8)
    for i, dataset in enumerate(datasets):
        gain = transformer[i] - classical[i]
        ax.text(i, max(classical[i], transformer[i]) + 0.035, f"Δ {gain:+.4f}", ha="center", fontsize=9, color="#047857" if gain >= 0 else "#dc2626")
        ax.text(i - width / 2, classical[i] - 0.025, CLASSICAL_BEST[dataset][0], ha="center", va="top", fontsize=7, color="#334155", rotation=0)
        ax.text(i + width / 2, transformer[i] - 0.025, "XLM-R Large", ha="center", va="top", fontsize=7, color="#334155")
    save(fig, "best_classical_vs_transformer.png")


def generate_xlmr_large_metrics(rows_by_key: dict[tuple[str, str], dict[str, str]]) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    reddit = [float(rows_by_key[("reddit", "xlmr-large")][m]) for m in metrics]
    twitter = [float(rows_by_key[("twitter", "xlmr-large")][m]) for m in metrics]
    x = list(range(len(metrics)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    b1 = ax.bar([i - width / 2 for i in x], reddit, width, label="Reddit", color="#22c55e", edgecolor="#15803d", linewidth=0.8)
    b2 = ax.bar([i + width / 2 for i in x], twitter, width, label="Twitter", color="#38bdf8", edgecolor="#0369a1", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Skor")
    ax.set_ylim(0.50, 0.90)
    ax.set_title("Metrik Evaluasi XLM-R Large sebagai Model Transformer Terbaik", fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{bar.get_height():.4f}", ha="center", fontsize=8)
    save(fig, "xlmr_large_metrics.png")


def main() -> None:
    rows = read_rows()
    rows_by_key = row_map(rows)
    generate_pipeline_architecture()
    generate_f1_vs_paper(rows_by_key)
    generate_gap_heatmap(rows_by_key)
    generate_best_classical_vs_transformer(rows_by_key)
    generate_xlmr_large_metrics(rows_by_key)
    print(f"\n📁 Transformer figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
