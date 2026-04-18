#!/usr/bin/env python3
"""
Run classical ML baselines for the IdSarcasm benchmark.

This script is designed for local PC / WSL2 / Colab execution, not the current VPS.
It reproduces the paper's classical setup at a practical level using:
- Logistic Regression
- Multinomial Naive Bayes
- SVM / SVC
- BoW + TF-IDF
- GridSearchCV + PredefinedSplit

Examples:
    python scripts/run_classical_baselines.py --dataset twitter
    python scripts/run_classical_baselines.py --dataset reddit
    python scripts/run_classical_baselines.py --dataset all
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import nltk
import numpy as np
import pandas as pd
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


@dataclass(frozen=True)
class DatasetConfig:
    hf_id: str
    text_column: str
    label_column: str = "label"


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "twitter": DatasetConfig(
        hf_id="w11wo/twitter_indonesia_sarcastic",
        text_column="tweet",
    ),
    "reddit": DatasetConfig(
        hf_id="w11wo/reddit_indonesia_sarcastic",
        text_column="text",
    ),
}

VECTORIZER_NAMES = ("bow", "tfidf")
MODEL_NAMES = ("lr", "nb", "svm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IdSarcasm classical ML baselines.")
    parser.add_argument(
        "--dataset",
        choices=["twitter", "reddit", "all"],
        default="all",
        help="Which dataset to run.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables",
        help="Directory for CSV and JSON outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="Subset of models to run.",
    )
    parser.add_argument(
        "--vectorizers",
        nargs="+",
        choices=VECTORIZER_NAMES,
        default=list(VECTORIZER_NAMES),
        help="Subset of vectorizers to run.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for GridSearchCV. Use 1 if your PC becomes unstable.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="GridSearchCV verbosity.",
    )
    return parser.parse_args()


def ensure_nltk_resources() -> None:
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
    }
    for package_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource: {package_name}")
            nltk.download(package_name, quiet=False)


def load_split_frames(dataset_name: str) -> Dict[str, pd.DataFrame]:
    config = DATASET_CONFIGS[dataset_name]
    print(f"Loading dataset: {config.hf_id}")
    dataset_dict = load_dataset(config.hf_id)
    frames: Dict[str, pd.DataFrame] = {}

    for split_name in ("train", "validation", "test"):
        frame = pd.DataFrame(dataset_dict[split_name])
        frame[config.text_column] = frame[config.text_column].fillna("").astype(str)
        frame[config.label_column] = frame[config.label_column].astype(int)
        frames[split_name] = frame
        print(f"  {split_name}: {len(frame):,} rows")

    return frames


def build_vectorizer(name: str):
    common_kwargs = {
        "tokenizer": word_tokenize,
        "token_pattern": None,
        "lowercase": True,
    }
    if name == "bow":
        return CountVectorizer(**common_kwargs)
    if name == "tfidf":
        return TfidfVectorizer(**common_kwargs)
    raise ValueError(f"Unsupported vectorizer: {name}")


def build_classifier(name: str):
    if name == "lr":
        model = LogisticRegression(max_iter=2000, solver="liblinear")
        grid = {
            "clf__C": [0.01, 0.1, 1, 10, 100],
        }
        return model, grid

    if name == "nb":
        model = MultinomialNB()
        grid = {
            "clf__alpha": np.linspace(0.001, 1.0, 50),
        }
        return model, grid

    if name == "svm":
        model = SVC()
        grid = {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__kernel": ["rbf", "linear"],
        }
        return model, grid

    raise ValueError(f"Unsupported model: {name}")


def build_trainval_data(
    frames: Dict[str, pd.DataFrame],
    text_column: str,
    label_column: str,
) -> tuple[pd.Series, pd.Series, PredefinedSplit]:
    train_df = frames["train"]
    val_df = frames["validation"]

    trainval_texts = pd.concat(
        [train_df[text_column], val_df[text_column]],
        ignore_index=True,
    )
    trainval_labels = pd.concat(
        [train_df[label_column], val_df[label_column]],
        ignore_index=True,
    )
    split_index = [-1] * len(train_df) + [0] * len(val_df)
    predefined_split = PredefinedSplit(test_fold=split_index)
    return trainval_texts, trainval_labels, predefined_split


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_single_experiment(
    dataset_name: str,
    vectorizer_name: str,
    model_name: str,
    frames: Dict[str, pd.DataFrame],
    output_dir: Path,
    n_jobs: int,
    verbose: int,
) -> Dict[str, object]:
    config = DATASET_CONFIGS[dataset_name]
    print(f"\n=== {dataset_name.upper()} | {vectorizer_name.upper()} | {model_name.upper()} ===")

    trainval_texts, trainval_labels, predefined_split = build_trainval_data(
        frames=frames,
        text_column=config.text_column,
        label_column=config.label_column,
    )
    test_texts = frames["test"][config.text_column]
    test_labels = frames["test"][config.label_column]

    vectorizer = build_vectorizer(vectorizer_name)
    classifier, param_grid = build_classifier(model_name)
    pipeline = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("clf", classifier),
        ]
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=predefined_split,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    search.fit(trainval_texts, trainval_labels)

    best_model = search.best_estimator_
    predictions = best_model.predict(test_texts)
    metrics = evaluate_predictions(test_labels, predictions)

    result: Dict[str, object] = {
        "dataset": dataset_name,
        "text_column": config.text_column,
        "vectorizer": vectorizer_name,
        "model": model_name,
        "best_validation_f1": float(search.best_score_),
        "best_params": search.best_params_,
        **metrics,
    }

    params_path = output_dir / f"{dataset_name}_{vectorizer_name}_{model_name}_best_params.json"
    with params_path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False, default=str)
    print(
        "Saved params:",
        params_path,
        f"| test_f1={result['f1']:.4f}",
    )
    return result


def run_dataset(
    dataset_name: str,
    output_dir: Path,
    models: List[str],
    vectorizers: List[str],
    n_jobs: int,
    verbose: int,
) -> None:
    frames = load_split_frames(dataset_name)
    rows: List[Dict[str, object]] = []

    for vectorizer_name in vectorizers:
        for model_name in models:
            rows.append(
                run_single_experiment(
                    dataset_name=dataset_name,
                    vectorizer_name=vectorizer_name,
                    model_name=model_name,
                    frames=frames,
                    output_dir=output_dir,
                    n_jobs=n_jobs,
                    verbose=verbose,
                )
            )

    results_df = pd.DataFrame(rows).sort_values(by=["vectorizer", "model"])
    csv_path = output_dir / f"classical_baselines_{dataset_name}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results table: {csv_path}")
    print(results_df[["dataset", "vectorizer", "model", "f1", "accuracy", "precision", "recall"]])


def main() -> None:
    args = parse_args()
    ensure_nltk_resources()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_datasets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]

    for dataset_name in selected_datasets:
        run_dataset(
            dataset_name=dataset_name,
            output_dir=output_dir,
            models=args.models,
            vectorizers=args.vectorizers,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
        )

    print("\nDone. Classical baseline outputs are ready.")


if __name__ == "__main__":
    main()
