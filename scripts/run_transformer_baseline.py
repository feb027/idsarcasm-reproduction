#!/usr/bin/env python3
"""Run a reproducible transformer baseline for Progress 3.

This runner is intentionally separate from the upstream snapshot. It keeps the
important IdSarcasm settings, but writes compact project-level artifacts under
results/tables/ so the experiment can be reported and compared with Progress 2.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataset_name: str
    text_column: str
    label_column: str = "label"


DATASET_CONFIGS: Mapping[str, DatasetConfig] = {
    "twitter": DatasetConfig(
        name="twitter",
        dataset_name="w11wo/twitter_indonesia_sarcastic",
        text_column="tweet",
    ),
    "reddit": DatasetConfig(
        name="reddit",
        dataset_name="w11wo/reddit_indonesia_sarcastic",
        text_column="text",
    ),
}

MODEL_ALIASES: Mapping[str, str] = {
    "indobert-base": "indobenchmark/indobert-base-p1",
    "xlmr-base": "xlm-roberta-base",
    "mbert-base": "bert-base-multilingual-cased",
}

METRIC_KEYS = ("accuracy", "precision", "recall", "f1")
PAPER_BASELINE_MODELS = ("indobert-base", "xlmr-base")
DEFAULT_BASELINE_TABLE = "results/tables/transformer_baselines.csv"
DEFAULT_SMOKE_TABLE = "results/tables/transformer_smoke.csv"
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_EVAL_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_LR_SCHEDULER_TYPE = "cosine"
DEFAULT_WEIGHT_DECAY = 0.03
DEFAULT_LABEL_SMOOTHING = 0.0
DEFAULT_PAD_TO_MAX_LENGTH = True
DEFAULT_EARLY_STOPPING_THRESHOLD = 0.01
DEFAULT_MAX_LENGTH = 128
DEFAULT_SEED = 42


def get_dataset_config(dataset: str) -> DatasetConfig:
    key = dataset.lower().strip()
    if key not in DATASET_CONFIGS:
        options = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset '{dataset}'. Choose one of: {options}")
    return DATASET_CONFIGS[key]


def resolve_model_name(model_alias_or_name: str) -> str:
    return MODEL_ALIASES.get(model_alias_or_name, model_alias_or_name)


def parse_best_metric(metrics: Mapping[str, Any]) -> Optional[float]:
    for key in ("eval_f1", "predict_f1", "test_f1", "f1"):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def training_strategy_kwargs(training_args_cls: Any) -> Dict[str, str]:
    """Return strategy keyword names compatible with installed Transformers.

    Transformers 4.x commonly uses `evaluation_strategy`; newer releases use
    `eval_strategy`. Save/logging names stay stable in most releases. Keeping
    this helper small makes the Colab runner less sensitive to whatever version
    Colab installs.
    """
    params = inspect.signature(training_args_cls).parameters
    eval_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    return {eval_key: "epoch", "save_strategy": "epoch", "logging_strategy": "epoch"}


def filter_training_arguments_kwargs(training_args_cls: Any, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop TrainingArguments kwargs unsupported by the installed Transformers.

    Some Colab images can have a Transformers build whose `TrainingArguments`
    signature differs from the usual pip release. Filtering here keeps the
    runner usable while preserving every supported paper-faithful setting.
    """
    params = inspect.signature(training_args_cls).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return dict(kwargs)

    filtered = {key: value for key, value in kwargs.items() if key in params}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"[compat] ignoring unsupported TrainingArguments kwargs: {', '.join(dropped)}")
    return filtered


def is_sample_limited(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, attr) is not None
        for attr in ("max_train_samples", "max_eval_samples", "max_predict_samples")
    )


def effective_table_path(args: argparse.Namespace) -> str:
    if is_sample_limited(args) and args.table_path == DEFAULT_BASELINE_TABLE:
        return DEFAULT_SMOKE_TABLE
    return args.table_path


def build_progress3_commands() -> Dict[str, str]:
    commands: Dict[str, str] = {}
    for alias in PAPER_BASELINE_MODELS:
        output_suffix = "twitter-indobert-base" if alias == "indobert-base" else "twitter-xlmr-base"
        commands[alias] = (
            "python scripts/run_transformer_baseline.py "
            f"--dataset twitter --model {alias} "
            f"--epochs {DEFAULT_EPOCHS} "
            f"--batch-size {DEFAULT_BATCH_SIZE} "
            f"--eval-batch-size {DEFAULT_EVAL_BATCH_SIZE} "
            f"--learning-rate {DEFAULT_LEARNING_RATE:g} "
            f"--lr-scheduler-type {DEFAULT_LR_SCHEDULER_TYPE} "
            f"--weight-decay {DEFAULT_WEIGHT_DECAY:g} "
            f"--label-smoothing-factor {DEFAULT_LABEL_SMOOTHING:g} "
            f"--max-length {DEFAULT_MAX_LENGTH} "
            f"--early-stopping-threshold {DEFAULT_EARLY_STOPPING_THRESHOLD:g} "
            f"--seed {DEFAULT_SEED} "
            "--pad-to-max-length --shuffle-train-dataset --fp16 "
            f"--output-dir results/transformer/{output_suffix} "
            f"--model-output-dir models/transformer/{output_suffix}"
        )
    return commands


def _rounded_metric(metrics: Mapping[str, Any], name: str) -> Optional[float]:
    for prefix in ("predict", "eval", "test", ""):
        key = f"{prefix}_{name}" if prefix else name
        if key in metrics and metrics[key] is not None:
            return round(float(metrics[key]), 4)
    return None


def build_result_row(
    *,
    dataset: str,
    model_alias: str,
    model_name: str,
    metrics: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset": dataset,
        "model_alias": model_alias,
        "model_name": model_name,
        "accuracy": _rounded_metric(metrics, "accuracy"),
        "precision": _rounded_metric(metrics, "precision"),
        "recall": _rounded_metric(metrics, "recall"),
        "f1": _rounded_metric(metrics, "f1"),
    }
    row.update(training_config)
    return row


def local_split_files(dataset: str, data_dir: Path) -> Dict[str, str]:
    return {
        "train": str(data_dir / f"{dataset}_train.csv"),
        "validation": str(data_dir / f"{dataset}_validation.csv"),
        "test": str(data_dir / f"{dataset}_test.csv"),
    }


def load_id_sarcasm_dataset(config: DatasetConfig, data_dir: Path):
    from datasets import load_dataset

    files = local_split_files(config.name, data_dir)
    if all(Path(path).exists() for path in files.values()):
        print(f"[data] using cached CSV files from {data_dir}")
        return load_dataset("csv", data_files=files)

    print(f"[data] cached CSV files not found, loading {config.dataset_name} from HuggingFace")
    return load_dataset(config.dataset_name)


def write_result_artifacts(
    *,
    output_dir: Path,
    table_path: Path,
    metrics: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    row_path = output_dir / "result_row.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    row_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

    write_header = not table_path.exists()
    fieldnames = list(row.keys())
    with table_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_and_evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    dataset_config = get_dataset_config(args.dataset)
    model_name = resolve_model_name(args.model)
    set_seed(args.seed)

    raw = load_id_sarcasm_dataset(dataset_config, Path(args.data_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def preprocess(batch: Mapping[str, Any]) -> Dict[str, Any]:
        padding = "max_length" if args.pad_to_max_length else False
        tokenized = tokenizer(
            batch[dataset_config.text_column],
            padding=padding,
            truncation=True,
            max_length=args.max_length,
        )
        tokenized["labels"] = batch[dataset_config.label_column]
        return tokenized

    tokenized = raw.map(preprocess, batched=True)
    keep_columns = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in tokenized["train"].column_names:
        keep_columns.add("token_type_ids")
    remove_columns = [col for col in tokenized["train"].column_names if col not in keep_columns]
    tokenized = tokenized.remove_columns(remove_columns)

    if args.shuffle_train_dataset:
        tokenized["train"] = tokenized["train"].shuffle(seed=args.seed)

    if args.max_train_samples:
        tokenized["train"] = tokenized["train"].select(range(min(args.max_train_samples, len(tokenized["train"]))))
    if args.max_eval_samples:
        tokenized["validation"] = tokenized["validation"].select(
            range(min(args.max_eval_samples, len(tokenized["validation"])))
        )
    if args.max_predict_samples:
        tokenized["test"] = tokenized["test"].select(range(min(args.max_predict_samples, len(tokenized["test"]))))

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "f1": f1_score(labels, predictions, zero_division=0),
        }

    fp16 = bool(args.fp16 and torch.cuda.is_available())
    training_args_kwargs = {
        "output_dir": str(args.model_output_dir),
        "overwrite_output_dir": True,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "lr_scheduler_type": args.lr_scheduler_type,
        "weight_decay": args.weight_decay,
        "label_smoothing_factor": args.label_smoothing_factor,
        **training_strategy_kwargs(TrainingArguments),
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "report_to": "none",
        "seed": args.seed,
        "fp16": fp16,
    }
    training_args = TrainingArguments(
        **filter_training_arguments_kwargs(TrainingArguments, training_args_kwargs)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
    )

    trainer.train()
    eval_metrics = trainer.evaluate(tokenized["validation"], metric_key_prefix="eval")
    predict_output = trainer.predict(tokenized["test"], metric_key_prefix="predict")
    metrics: Dict[str, Any] = {**eval_metrics, **predict_output.metrics}

    training_config = {
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "weight_decay": args.weight_decay,
        "label_smoothing_factor": args.label_smoothing_factor,
        "pad_to_max_length": args.pad_to_max_length,
        "shuffle_train_dataset": args.shuffle_train_dataset,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_predict_samples": args.max_predict_samples,
        "sample_limited": is_sample_limited(args),
        "seed": args.seed,
        "fp16": fp16,
    }
    row = build_result_row(
        dataset=dataset_config.name,
        model_alias=args.model,
        model_name=model_name,
        metrics=metrics,
        training_config=training_config,
    )
    write_result_artifacts(output_dir=Path(args.output_dir), table_path=Path(effective_table_path(args)), metrics=metrics, row=row)
    print(json.dumps(row, indent=2, ensure_ascii=False))
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run IdSarcasm transformer baseline for Progress 3")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="twitter")
    parser.add_argument("--model", default="indobert-base", help="Alias or HuggingFace model name")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="results/transformer/twitter-indobert-base")
    parser.add_argument("--table-path", default=DEFAULT_BASELINE_TABLE)
    parser.add_argument("--model-output-dir", default="models/transformer/twitter-indobert-base")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--lr-scheduler-type", default=DEFAULT_LR_SCHEDULER_TYPE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--label-smoothing-factor", type=float, default=DEFAULT_LABEL_SMOOTHING)
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=DEFAULT_EARLY_STOPPING_THRESHOLD)
    parser.add_argument(
        "--pad-to-max-length",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PAD_TO_MAX_LENGTH,
        help="Pad every sample to max length, matching the paper script default",
    )
    parser.add_argument(
        "--shuffle-train-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle the train split before training, matching the paper recipes by default",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fp16 when CUDA is available, matching the paper recipes by default",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Smoke-test limit")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Smoke-test limit")
    parser.add_argument("--max-predict-samples", type=int, default=None, help="Smoke-test limit")
    return parser


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    train_and_evaluate(parse_args())
