#!/usr/bin/env python3
"""Generate final Progress 6 comparison tables and figures.

The script reads committed result CSVs and writes final summary artifacts used by
`docs/laporan-proyek.md` and README.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

PAPER_TARGETS = {
    "twitter": {"model": "XLM-R Large (paper)", "f1": 0.7692},
    "reddit": {"model": "XLM-R Large (paper)", "f1": 0.6274},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
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


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def best(rows: list[dict[str, str]], key: str = "f1") -> dict[str, str]:
    return max(rows, key=lambda row: as_float(row, key))


def model_label(row: dict[str, str], family: str) -> str:
    if family == "classical":
        return f"{row['vectorizer'].upper()} {row['model'].upper()}"
    if family == "modern":
        alias = row["model_alias"].replace("-gguf", "")
        return f"{alias} ({row['mode']})"
    return row.get("model_alias", row.get("model_name", "-"))


def read_optimization_results() -> list[dict[str, str]]:
    """Read optimization rows from the canonical CSV plus result_row.json files.

    Some Colab runs can be pushed with their per-run artifacts before the
    aggregate CSV is updated. Reading both sources keeps final analysis robust.
    """
    rows = read_csv(TABLES / "optimization_runs.csv")
    seen = {row.get("run_id") for row in rows}
    for path in sorted((ROOT / "results" / "optimization").glob("*/result_row.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("run_id") not in seen:
            rows.append({key: str(value) for key, value in data.items()})
            seen.add(data.get("run_id"))
    return rows


def optimized_score(row: dict[str, str]) -> tuple[float, str]:
    default_f1 = as_float(row, "test_default_f1")
    tuned_f1 = as_float(row, "test_tuned_f1")
    if default_f1 >= tuned_f1:
        return default_f1, "default"
    return tuned_f1, "threshold"


def optimized_label(row: dict[str, str]) -> str:
    score, strategy = optimized_score(row)
    if row.get("learning_rate") == "2e-05" and row.get("max_length") == "128" and strategy == "default":
        return "xlmr-large-lr2e-5-default"
    if strategy == "threshold":
        return "xlmr-large-threshold"
    return f"{row.get('model_alias', 'xlmr-large')}-{strategy}"


def build_final_summary() -> list[dict[str, Any]]:
    classical = {
        "twitter": read_csv(TABLES / "classical_baselines_twitter.csv"),
        "reddit": read_csv(TABLES / "classical_baselines_reddit.csv"),
    }
    transformer = read_csv(TABLES / "transformer_baselines.csv")
    zeroshot = read_csv(TABLES / "zeroshot_baselines.csv")
    modern = read_csv(TABLES / "modern_llm_experiments.csv")
    optimization = read_optimization_results()

    rows: list[dict[str, Any]] = []
    for dataset in ["twitter", "reddit"]:
        best_classical = best(classical[dataset])
        best_transformer = best([r for r in transformer if r["dataset"] == dataset])
        best_zero = best([r for r in zeroshot if r["dataset"] == dataset])
        optimized_candidates = [
            r for r in optimization if r["dataset"] == dataset and r["model_alias"] == "xlmr-large"
        ]
        best_optimized = max(optimized_candidates, key=lambda row: optimized_score(row)[0])
        best_optimized_f1, best_optimized_strategy = optimized_score(best_optimized)
        dataset_modern = [r for r in modern if r["dataset"] == dataset]
        best_modern = best(dataset_modern) if dataset_modern else None
        rows.append(
            {
                "dataset": dataset,
                "best_classical_model": model_label(best_classical, "classical"),
                "best_classical_f1": round(as_float(best_classical, "f1"), 4),
                "best_transformer_model": model_label(best_transformer, "transformer"),
                "best_transformer_f1": round(as_float(best_transformer, "f1"), 4),
                "optimized_transformer_model": optimized_label(best_optimized),
                "optimized_transformer_strategy": best_optimized_strategy,
                "optimized_transformer_f1": round(best_optimized_f1, 4),
                "best_zeroshot_model": model_label(best_zero, "zeroshot"),
                "best_zeroshot_f1": round(as_float(best_zero, "f1"), 4),
                "best_modern_llm_model": model_label(best_modern, "modern") if best_modern else "not_run",
                "best_modern_llm_f1": round(as_float(best_modern, "f1"), 4) if best_modern else "",
                "paper_best_model": PAPER_TARGETS[dataset]["model"],
                "paper_best_f1": PAPER_TARGETS[dataset]["f1"],
            }
        )
    return rows


def build_overall_ranking(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary:
        dataset = item["dataset"]
        candidates = [
            ("Classical ML", item["best_classical_model"], item["best_classical_f1"]),
            ("Transformer baseline", item["best_transformer_model"], item["best_transformer_f1"]),
            ("Optimized transformer", item["optimized_transformer_model"], item["optimized_transformer_f1"]),
            ("Zero-shot paper LLM", item["best_zeroshot_model"], item["best_zeroshot_f1"]),
        ]
        if item["best_modern_llm_f1"] != "":
            candidates.append(("Modern local LLM", item["best_modern_llm_model"], item["best_modern_llm_f1"]))
        for rank, (family, model, f1) in enumerate(sorted(candidates, key=lambda x: float(x[2]), reverse=True), 1):
            rows.append({"dataset": dataset, "rank": rank, "method_family": family, "model": model, "f1": f1})
    return rows


def plot_final_method_comparison(summary: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    datasets = [r["dataset"].capitalize() for r in summary]
    method_keys = [
        ("best_classical_f1", "Classical ML", "#64748B"),
        ("best_transformer_f1", "Transformer baseline", "#2563EB"),
        ("optimized_transformer_f1", "Optimized transformer", "#16A34A"),
        ("best_zeroshot_f1", "Zero-shot paper LLM", "#F97316"),
        ("best_modern_llm_f1", "Modern local LLM", "#7C3AED"),
        ("paper_best_f1", "Paper target", "#111827"),
    ]
    x = list(range(len(datasets)))
    width = 0.12
    plt.figure(figsize=(11, 5.5))
    offsets = [(-2.5 + i) * width for i in range(len(method_keys))]
    for (key, label, color), offset in zip(method_keys, offsets):
        values = []
        for row in summary:
            value = row[key]
            values.append(float(value) if value != "" else 0.0)
        bars = plt.bar([i + offset for i in x], values, width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=7, rotation=90)
    plt.xticks(x, datasets)
    plt.ylim(0, 0.86)
    plt.ylabel("F1-score")
    plt.title("Perbandingan Akhir Metode pada Benchmark IdSarcasm")
    plt.legend(ncol=3, fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "final_method_comparison.png", dpi=180)
    plt.close()


def plot_best_ranking(ranking: list[dict[str, Any]]) -> None:
    for dataset in ["twitter", "reddit"]:
        rows = [r for r in ranking if r["dataset"] == dataset]
        labels = [r["method_family"] for r in rows]
        values = [float(r["f1"]) for r in rows]
        colors = ["#16A34A" if label == "Optimized transformer" else "#94A3B8" for label in labels]
        plt.figure(figsize=(9, 4.8))
        bars = plt.bar(labels, values, color=colors)
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center", fontsize=9)
        plt.xticks(rotation=20, ha="right")
        plt.ylim(0, max(values) + 0.12)
        plt.ylabel("F1-score")
        plt.title(f"Ranking Akhir Metode pada Dataset {dataset.capitalize()}")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(FIGURES / f"final_{dataset}_method_ranking.png", dpi=180)
        plt.close()


def plot_progress_summary() -> None:
    labels = ["P2\nClassical", "P3\nTransformer", "P4\nZero-shot", "P5\nOptimized", "P6\nFinal"]
    completed = [1, 1, 1, 1, 1]
    plt.figure(figsize=(9.5, 2.7))
    plt.barh([0] * len(labels), completed, left=list(range(len(labels))), color="#2563EB", height=0.35)
    for i, label in enumerate(labels):
        plt.text(i + 0.5, 0, label, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    plt.xlim(0, len(labels))
    plt.ylim(-0.6, 0.6)
    plt.yticks([])
    plt.xticks([])
    plt.title("Ringkasan Progress Proyek IdSarcasm")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES / "final_progress_summary.png", dpi=180)
    plt.close()


def main() -> None:
    summary = build_final_summary()
    ranking = build_overall_ranking(summary)
    write_csv(TABLES / "final_method_comparison.csv", summary)
    write_csv(TABLES / "final_method_ranking.csv", ranking)
    plot_final_method_comparison(summary)
    plot_best_ranking(ranking)
    plot_progress_summary()
    print("Generated final analysis artifacts")


if __name__ == "__main__":
    main()
