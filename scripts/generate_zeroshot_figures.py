#!/usr/bin/env python3
"""Generate Progress 4 zero-shot LLM figures for the IdSarcasm report.

Uses only Python stdlib + matplotlib so it can run on the VPS without pandas.
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
TABLE_PATH = ROOT / "results" / "tables" / "zeroshot_baselines.csv"
LOG_DIR = ROOT / "results" / "logs"
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

MODEL_ORDER = [
    "bloomz-560m",
    "bloomz-1b1",
    "bloomz-1b7",
    "bloomz-3b",
    "bloomz-7b1",
    "mt0-small",
    "mt0-base",
    "mt0-large",
    "mt0-xl",
]
MODEL_LABELS = {
    "bloomz-560m": "BLOOMZ\n560M",
    "bloomz-1b1": "BLOOMZ\n1.1B",
    "bloomz-1b7": "BLOOMZ\n1.7B",
    "bloomz-3b": "BLOOMZ\n3B",
    "bloomz-7b1": "BLOOMZ\n7.1B",
    "mt0-small": "mT0\nSmall",
    "mt0-base": "mT0\nBase",
    "mt0-large": "mT0\nLarge",
    "mt0-xl": "mT0\nXL",
}
PAPER_F1 = {
    ("reddit", "bloomz-560m"): 0.3870,
    ("reddit", "bloomz-1b1"): 0.3944,
    ("reddit", "bloomz-1b7"): 0.3758,
    ("reddit", "bloomz-3b"): 0.4000,
    ("reddit", "bloomz-7b1"): 0.4036,
    ("reddit", "mt0-small"): 0.4000,
    ("reddit", "mt0-base"): 0.3990,
    ("reddit", "mt0-large"): 0.3998,
    ("reddit", "mt0-xl"): 0.4001,
    ("twitter", "bloomz-560m"): 0.3916,
    ("twitter", "bloomz-1b1"): 0.3987,
    ("twitter", "bloomz-1b7"): 0.3885,
    ("twitter", "bloomz-3b"): 0.3847,
    ("twitter", "bloomz-7b1"): 0.3968,
    ("twitter", "mt0-small"): 0.3988,
    ("twitter", "mt0-base"): 0.3985,
    ("twitter", "mt0-large"): 0.3989,
    ("twitter", "mt0-xl"): 0.3988,
}


def read_rows() -> list[dict[str, str]]:
    with TABLE_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(r["dataset"], r["model_alias"]): r for r in rows if r.get("sample_limited") == "False"}


def save(fig: plt.Figure, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"✅ {path.relative_to(ROOT)}")


def generate_pipeline_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 4.9))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    stages = [
        (0.35, "Dataset\nIdSarcasm", "test split\nTwitter + Reddit", "#dbeafe", "#1d4ed8"),
        (2.35, "5 Prompt\nPaper", "prompt templates\nfrom source code", "#fef3c7", "#b45309"),
        (4.35, "LLM\nZero-shot", "BLOOMZ / mT0\nno fine-tuning", "#dcfce7", "#15803d"),
        (6.35, "Label\nScoring", "log probability:\nsarcastic vs not", "#ede9fe", "#6d28d9"),
        (8.35, "Mean\nMetrics", "average across\n5 prompts", "#fee2e2", "#b91c1c"),
        (10.35, "Result\nArtifacts", "CSV, metrics.json,\npredictions, logs", "#e0f2fe", "#0369a1"),
    ]
    arrow = dict(arrowstyle="->", color="#64748b", lw=1.8, shrinkA=5, shrinkB=5)
    for i, (x, title, desc, fc, ec) in enumerate(stages):
        rect = mpatches.FancyBboxPatch(
            (x, 1.72),
            1.55,
            1.5,
            boxstyle="round,pad=0.13,rounding_size=0.12",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.6,
        )
        ax.add_patch(rect)
        ax.text(x + 0.775, 2.64, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=ec)
        ax.text(x + 0.775, 1.03, desc, ha="center", va="center", fontsize=8.0, color="#475569")
        if i < len(stages) - 1:
            ax.annotate("", xy=(stages[i + 1][0], 2.47), xytext=(x + 1.57, 2.47), arrowprops=arrow)

    ax.text(6.6, 4.32, "Arsitektur Pipeline Zero-shot LLM Progress 4", ha="center", va="center", fontsize=13, fontweight="bold", color="#0f172a")
    ax.text(6.6, 0.25, "Runner memakai hf-logprobs agar label dipilih dari skor kandidat, bukan jawaban bebas generatif", ha="center", va="center", fontsize=8.2, color="#64748b")
    save(fig, "zeroshot_pipeline_architecture.png")


def generate_f1_vs_paper(by_key: dict[tuple[str, str], dict[str, str]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=True)
    x = list(range(len(MODEL_ORDER)))
    width = 0.34
    for ax, dataset, title in zip(axes, ["twitter", "reddit"], ["Dataset Twitter (9/9 run selesai)", "Dataset Reddit (5/9 run selesai, 4 run tercatat sebagai keterbatasan runtime)"]):
        paper = [PAPER_F1[(dataset, m)] for m in MODEL_ORDER]
        repro = [float(by_key[(dataset, m)]["f1"]) if (dataset, m) in by_key else None for m in MODEL_ORDER]
        ax.bar([i - width / 2 for i in x], paper, width, label="Paper", color="#a78bfa", edgecolor="#6d28d9", linewidth=0.8)
        done_x = [i + width / 2 for i, v in enumerate(repro) if v is not None]
        done_y = [v for v in repro if v is not None]
        ax.bar(done_x, done_y, width, label="Reproduksi", color="#34d399", edgecolor="#047857", linewidth=0.8)
        missing_x = [i + width / 2 for i, v in enumerate(repro) if v is None]
        if missing_x:
            ax.bar(missing_x, [0.018] * len(missing_x), width, bottom=0.36, label="Belum selesai", color="#e5e7eb", edgecolor="#9ca3af", hatch="//", linewidth=0.8)
            for mx in missing_x:
                ax.text(mx, 0.405, "sesi\nterputus", ha="center", va="bottom", fontsize=7, color="#6b7280")
        for i, (p, r) in enumerate(zip(paper, repro)):
            if r is None:
                continue
            gap = r - p
            ax.text(i, max(p, r) + 0.004, f"{gap:+.4f}", ha="center", va="bottom", fontsize=7.5, color="#047857" if gap >= 0 else "#dc2626")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0.35, 0.415)
        ax.set_ylabel("F1-score")
        ax.grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=8)
    legend_handles = [
        mpatches.Patch(facecolor="#a78bfa", edgecolor="#6d28d9", label="Paper"),
        mpatches.Patch(facecolor="#34d399", edgecolor="#047857", label="Reproduksi"),
        mpatches.Patch(facecolor="#e5e7eb", edgecolor="#9ca3af", hatch="//", label="Dicoba, sesi terputus"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper left", frameon=False, ncol=3)
    fig.suptitle("Perbandingan F1 Zero-shot LLM: Paper vs Hasil Progress 4", fontsize=14, fontweight="bold", y=1.01)
    save(fig, "zeroshot_f1_vs_paper.png")


def generate_completion_matrix(by_key: dict[tuple[str, str], dict[str, str]]) -> None:
    status_values = []
    for model in MODEL_ORDER:
        row = []
        for dataset in ["twitter", "reddit"]:
            if (dataset, model) in by_key:
                row.append(2)
            else:
                log = LOG_DIR / f"progress-4-zeroshot-{dataset}-hf-logprobs-{model}-full.log"
                row.append(1 if log.exists() else 0)
        status_values.append(row)

    cmap = matplotlib.colors.ListedColormap(["#f1f5f9", "#fde68a", "#86efac"])
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.imshow(status_values, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Twitter", "Reddit"])
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABELS[m].replace("\n", " ") for m in MODEL_ORDER])
    ax.set_title("Status Run Zero-shot Progress 4", fontsize=13, fontweight="bold", pad=12)
    labels = {0: "belum", 1: "sesi\nterputus", 2: "selesai"}
    colors = {0: "#64748b", 1: "#92400e", 2: "#166534"}
    for i, row in enumerate(status_values):
        for j, value in enumerate(row):
            ax.text(j, i, labels[value], ha="center", va="center", fontsize=8, color=colors[value], fontweight="bold" if value == 2 else "normal")
    ax.set_xticks([x - 0.5 for x in range(3)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(MODEL_ORDER) + 1)], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    legend = [
        mpatches.Patch(color="#86efac", label="Selesai dan masuk CSV"),
        mpatches.Patch(color="#fde68a", label="Dicoba, sesi terputus"),
        mpatches.Patch(color="#f1f5f9", label="Belum ada log"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, frameon=False)
    save(fig, "zeroshot_run_completion_matrix.png")


def generate_runtime_chart(by_key: dict[tuple[str, str], dict[str, str]]) -> None:
    completed = [(d, m, float(r["runtime_seconds"]) / 60.0) for (d, m), r in by_key.items() if d in {"twitter", "reddit"}]
    completed.sort(key=lambda x: (x[0], MODEL_ORDER.index(x[1])))
    labels = [f"{d.title()}\n{MODEL_LABELS[m].replace(chr(10), ' ')}" for d, m, _ in completed]
    values = [v for _, _, v in completed]
    colors = ["#38bdf8" if d == "twitter" else "#22c55e" for d, _, _ in completed]

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="#334155", linewidth=0.6)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Runtime (menit)")
    ax.set_title("Runtime Full Run Zero-shot yang Berhasil Selesai", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35, f"{bar.get_height():.1f}", ha="center", fontsize=7.5)
    ax.text(0.01, 0.96, "Reddit lebih lama karena 2.824 test data × 5 prompt = 14.120 scoring per model", transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#475569")
    save(fig, "zeroshot_runtime_minutes.png")


def generate_metrics_profile(by_key: dict[tuple[str, str], dict[str, str]]) -> None:
    selected = [("twitter", "mt0-large"), ("twitter", "bloomz-1b1"), ("reddit", "mt0-small"), ("reddit", "bloomz-3b")]
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    width = 0.18
    x = list(range(len(metrics)))

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    palette = ["#38bdf8", "#60a5fa", "#22c55e", "#84cc16"]
    for idx, key in enumerate(selected):
        if key not in by_key:
            continue
        vals = [float(by_key[key][m]) for m in metrics]
        shift = (idx - 1.5) * width
        ax.bar([i + shift for i in x], vals, width, label=f"{key[0].title()} {MODEL_LABELS[key[1]].replace(chr(10), ' ')}", color=palette[idx], edgecolor="#334155", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor")
    ax.set_title("Profil Metrik Zero-shot: Recall Tinggi, Precision Rendah", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.96, "Banyak model cenderung memilih label sarkastik, sehingga recall tinggi tetapi precision dan accuracy rendah.", transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#475569")
    save(fig, "zeroshot_metrics_profile.png")


def main() -> None:
    rows = read_rows()
    by_key = rows_by_key(rows)
    generate_pipeline_architecture()
    generate_f1_vs_paper(by_key)
    generate_completion_matrix(by_key)
    generate_runtime_chart(by_key)
    generate_metrics_profile(by_key)
    print(f"\n📁 Zero-shot figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
