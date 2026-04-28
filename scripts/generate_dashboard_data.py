from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "dashboard" / "data" / "dashboard-data.json"

FINAL_COMPARISON_PATH = ROOT / "results" / "tables" / "final_method_comparison.csv"
FINAL_RANKING_PATH = ROOT / "results" / "tables" / "final_method_ranking.csv"

PREDICTION_SOURCES: dict[str, dict[str, Any]] = {
    "twitter": {
        "path": ROOT
        / "results"
        / "optimization"
        / "twitter-xlmr-large-lr2e-5-len128"
        / "predictions.csv",
        "result_path": ROOT
        / "results"
        / "optimization"
        / "twitter-xlmr-large-lr2e-5-len128"
        / "result_row.json",
        "selected_strategy": "default",
        "label": "Twitter",
    },
    "reddit": {
        "path": ROOT
        / "results"
        / "optimization"
        / "reddit-xlmr-large-threshold"
        / "predictions.csv",
        "result_path": ROOT
        / "results"
        / "optimization"
        / "reddit-xlmr-large-threshold"
        / "result_row.json",
        "selected_strategy": "tuned",
        "label": "Reddit",
    },
}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def to_float(value: str | float | int | None) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def to_int(value: str | int) -> int:
    return int(value)


def label_name(label: int) -> str:
    return "sarcastic" if label == 1 else "non-sarcastic"


def format_model_name(raw_name: str) -> str:
    return raw_name.replace("-default", " (default)")


def build_comparison(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    comparison: list[dict[str, Any]] = []
    for row in rows:
        dataset = row["dataset"]
        comparison.append(
            {
                "dataset": dataset,
                "dataset_label": dataset.title(),
                "paper_best_model": row["paper_best_model"],
                "paper_best_f1": to_float(row["paper_best_f1"]),
                "methods": [
                    {
                        "key": "classical",
                        "label": "Classical ML",
                        "model": row["best_classical_model"],
                        "f1": to_float(row["best_classical_f1"]),
                    },
                    {
                        "key": "transformer",
                        "label": "Transformer baseline",
                        "model": row["best_transformer_model"],
                        "f1": to_float(row["best_transformer_f1"]),
                    },
                    {
                        "key": "optimized",
                        "label": "Optimized transformer",
                        "model": format_model_name(row["optimized_transformer_model"]),
                        "f1": to_float(row["optimized_transformer_f1"]),
                        "strategy": row["optimized_transformer_strategy"],
                    },
                    {
                        "key": "zero_shot",
                        "label": "Zero-shot paper LLM",
                        "model": row["best_zeroshot_model"],
                        "f1": to_float(row["best_zeroshot_f1"]),
                    },
                    {
                        "key": "modern_llm",
                        "label": "Modern local LLM",
                        "model": row["best_modern_llm_model"],
                        "f1": to_float(row["best_modern_llm_f1"]),
                    },
                ],
            }
        )
    return comparison


def build_ranking(rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    ranking: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        dataset = row["dataset"]
        ranking.setdefault(dataset, []).append(
            {
                "rank": to_int(row["rank"]),
                "method_family": row["method_family"],
                "model": row["model"],
                "f1": to_float(row["f1"]),
            }
        )
    for dataset_rows in ranking.values():
        dataset_rows.sort(key=lambda item: item["rank"])
    return ranking


def build_gap_cards(comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for row in comparison_rows:
        optimized = next(method for method in row["methods"] if method["key"] == "optimized")
        gap = round(optimized["f1"] - row["paper_best_f1"], 4)
        selected_strategy = PREDICTION_SOURCES[row["dataset"]]["selected_strategy"]
        cards.append(
            {
                "dataset": row["dataset"],
                "dataset_label": row["dataset_label"],
                "optimized_model": optimized["model"],
                "optimized_f1": optimized["f1"],
                "paper_f1": row["paper_best_f1"],
                "gap_f1": gap,
                "status": "above" if gap >= 0 else "below",
                "selected_strategy": selected_strategy,
                "optimization_strategy": optimized.get("strategy", "default"),
            }
        )
    return cards


def build_explorer_dataset(dataset: str, source: dict[str, Any]) -> dict[str, Any]:
    prediction_rows = load_csv_rows(source["path"])
    result_row = load_json(source["result_path"])
    selected_strategy = source["selected_strategy"]
    test_rows = [row for row in prediction_rows if row["split"] == "test"]

    output_rows: list[dict[str, Any]] = []
    sarcastic_count = 0
    non_sarcastic_count = 0
    correctness_counts = {
        "default": {"correct": 0, "error": 0},
        "tuned": {"correct": 0, "error": 0},
        "selected": {"correct": 0, "error": 0},
    }

    for index, row in enumerate(test_rows, start=1):
        true_label = to_int(row["true_label"])
        pred_default = to_int(row["pred_default"])
        pred_tuned = to_int(row["pred_tuned"])
        default_correct = bool(to_int(row["default_correct"]))
        tuned_correct = bool(to_int(row["tuned_correct"]))
        selected_pred = pred_default if selected_strategy == "default" else pred_tuned
        selected_correct = default_correct if selected_strategy == "default" else tuned_correct

        if true_label == 1:
            sarcastic_count += 1
        else:
            non_sarcastic_count += 1

        correctness_counts["default"]["correct" if default_correct else "error"] += 1
        correctness_counts["tuned"]["correct" if tuned_correct else "error"] += 1
        correctness_counts["selected"]["correct" if selected_correct else "error"] += 1

        output_rows.append(
            {
                "id": f"{dataset}-{index}",
                "dataset": dataset,
                "dataset_label": source["label"],
                "split": row["split"],
                "sample_idx": to_int(row["sample_idx"]),
                "text": html.unescape(row["text"]).strip(),
                "true_label": true_label,
                "true_label_name": label_name(true_label),
                "prob_sarcastic": to_float(row["prob_sarcastic"]),
                "threshold_tuned": to_float(row["threshold_tuned"]),
                "predictions": {
                    "default": pred_default,
                    "tuned": pred_tuned,
                    "selected": selected_pred,
                },
                "prediction_names": {
                    "default": label_name(pred_default),
                    "tuned": label_name(pred_tuned),
                    "selected": label_name(selected_pred),
                },
                "correctness": {
                    "default": default_correct,
                    "tuned": tuned_correct,
                    "selected": selected_correct,
                },
            }
        )

    return {
        "dataset": dataset,
        "dataset_label": source["label"],
        "run_id": result_row["run_id"],
        "model_name": result_row["model_name"],
        "selected_strategy": selected_strategy,
        "selected_threshold": result_row["selected_threshold"],
        "strategies": ["selected", "default", "tuned"],
        "test_rows": len(output_rows),
        "label_distribution": {
            "sarcastic": sarcastic_count,
            "non_sarcastic": non_sarcastic_count,
        },
        "metrics": {
            "default": {
                "accuracy": result_row["test_default_accuracy"],
                "precision": result_row["test_default_precision"],
                "recall": result_row["test_default_recall"],
                "f1": result_row["test_default_f1"],
            },
            "tuned": {
                "accuracy": result_row["test_tuned_accuracy"],
                "precision": result_row["test_tuned_precision"],
                "recall": result_row["test_tuned_recall"],
                "f1": result_row["test_tuned_f1"],
            },
            "selected": {
                "accuracy": result_row[f"test_{selected_strategy}_accuracy"],
                "precision": result_row[f"test_{selected_strategy}_precision"],
                "recall": result_row[f"test_{selected_strategy}_recall"],
                "f1": result_row[f"test_{selected_strategy}_f1"],
            },
        },
        "correctness_counts": correctness_counts,
        "rows": output_rows,
    }


def build_payload() -> dict[str, Any]:
    comparison_rows = build_comparison(load_csv_rows(FINAL_COMPARISON_PATH))
    ranking_rows = build_ranking(load_csv_rows(FINAL_RANKING_PATH))
    explorer_datasets = [
        build_explorer_dataset(dataset, source)
        for dataset, source in PREDICTION_SOURCES.items()
    ]

    total_test_rows = sum(dataset["test_rows"] for dataset in explorer_datasets)
    twitter_card = next(card for card in build_gap_cards(comparison_rows) if card["dataset"] == "twitter")
    reddit_card = next(card for card in build_gap_cards(comparison_rows) if card["dataset"] == "reddit")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": {
            "name": "idsarcasm-reproduction",
            "url": "https://github.com/feb027/idsarcasm-reproduction",
            "readme": "./README.md",
            "report": "./docs/laporan-proyek.md",
            "reproducibility": "./docs/reproducibility.md",
        },
        "overview": {
            "dataset_count": len(comparison_rows),
            "test_rows": total_test_rows,
            "best_overall_f1": max(card["optimized_f1"] for card in build_gap_cards(comparison_rows)),
            "twitter_gap_f1": twitter_card["gap_f1"],
            "reddit_gap_f1": reddit_card["gap_f1"],
        },
        "comparison": comparison_rows,
        "ranking": ranking_rows,
        "gap_cards": build_gap_cards(comparison_rows),
        "explorer": {"datasets": explorer_datasets},
        "notes": [
            "Dashboard ini bersifat statis: semua visual membaca satu file JSON hasil generate, tanpa backend atau build pipeline.",
            "Perbandingan final mengambil tabel `final_method_comparison.csv` dan `final_method_ranking.csv`.",
            "Error explorer menampilkan baris split `test` dari run final terpilih: Twitter memakai strategi default, Reddit memakai threshold-tuned.",
            "Filter strategi tersedia agar pengguna bisa membandingkan prediksi `selected`, `default`, dan `tuned` pada dataset yang sama.",
        ],
    }


def main() -> None:
    payload = build_payload()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
