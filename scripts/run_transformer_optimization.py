#!/usr/bin/env python3
"""Run Progress 5 transformer optimization experiments for IdSarcasm.

This runner extends the Progress 3 transformer baseline workflow with:
- validation/test prediction artifacts,
- validation-set threshold tuning for the sarcasm class,
- before/after test metrics at default argmax vs tuned threshold,
- compact result rows for optimization reporting.

Important: threshold is selected on validation predictions only. The selected
threshold is then applied once to test predictions.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


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
    "indobert-large": "indobenchmark/indobert-large-p1",
    "indobert-indolem-base": "indolem/indobert-base-uncased",
    "mbert-base": "bert-base-multilingual-cased",
    "xlmr-base": "xlm-roberta-base",
    "xlmr-large": "xlm-roberta-large",
}

DEFAULT_TABLE = "results/tables/optimization_runs.csv"
DEFAULT_SMOKE_TABLE = "results/tables/optimization_smoke.csv"
DEFAULT_OUTPUT_ROOT = "results/optimization"
DEFAULT_MODEL_ROOT = "models/optimization"
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_EVAL_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_LR_SCHEDULER_TYPE = "cosine"
DEFAULT_WEIGHT_DECAY = 0.03
DEFAULT_LABEL_SMOOTHING = 0.0
DEFAULT_MAX_LENGTH = 128
DEFAULT_EARLY_STOPPING_THRESHOLD = 0.01
DEFAULT_SEED = 42


def get_dataset_config(dataset: str) -> DatasetConfig:
    key = dataset.lower().strip()
    if key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose one of: {', '.join(sorted(DATASET_CONFIGS))}")
    return DATASET_CONFIGS[key]


def resolve_model_name(model_alias_or_name: str) -> str:
    return MODEL_ALIASES.get(model_alias_or_name, model_alias_or_name)


def sanitize_for_path(value: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-._")
    return cleaned or "run"


def run_id(args: argparse.Namespace) -> str:
    if args.run_name:
        return sanitize_for_path(args.run_name)
    parts = [
        args.dataset,
        sanitize_for_path(args.model),
        f"lr{args.learning_rate:g}",
        f"len{args.max_length}",
        f"wd{args.weight_decay:g}",
        f"ls{args.label_smoothing_factor:g}",
        f"seed{args.seed}",
    ]
    if args.max_train_samples or args.max_eval_samples or args.max_predict_samples:
        parts.append("smoke")
    return "-".join(parts)


def default_output_dir(args: argparse.Namespace) -> str:
    return f"{DEFAULT_OUTPUT_ROOT}/{run_id(args)}"


def default_model_output_dir(args: argparse.Namespace) -> str:
    return f"{DEFAULT_MODEL_ROOT}/{run_id(args)}"


def is_sample_limited(args: argparse.Namespace) -> bool:
    return any(getattr(args, attr) is not None for attr in ("max_train_samples", "max_eval_samples", "max_predict_samples"))


def effective_table_path(args: argparse.Namespace) -> str:
    if is_sample_limited(args) and args.table_path == DEFAULT_TABLE:
        return DEFAULT_SMOKE_TABLE
    return args.table_path


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


def training_strategy_kwargs(training_args_cls: Any) -> Dict[str, str]:
    params = inspect.signature(training_args_cls).parameters
    eval_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    return {eval_key: "epoch", "save_strategy": "epoch", "logging_strategy": "epoch"}


def filter_supported_kwargs(callable_obj: Any, kwargs: Mapping[str, Any], label: str) -> Dict[str, Any]:
    params = inspect.signature(callable_obj).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return dict(kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in params}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"[compat] ignoring unsupported {label} kwargs: {', '.join(dropped)}")
    return filtered


def trainer_tokenizer_kwargs(trainer_cls: Any, tokenizer: Any) -> Dict[str, Any]:
    params = inspect.signature(trainer_cls).parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    if "tokenizer" in params:
        return {"tokenizer": tokenizer}
    return {}


def compute_binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def threshold_predictions(prob_sarcastic: Sequence[float], threshold: float) -> list[int]:
    return [1 if p >= threshold else 0 for p in prob_sarcastic]


def tune_threshold(
    *,
    y_true: Sequence[int],
    prob_sarcastic: Sequence[float],
    start: float,
    stop: float,
    step: float,
) -> tuple[float, Dict[str, float], list[Dict[str, Any]]]:
    thresholds: list[float] = []
    current = start
    while current <= stop + 1e-12:
        thresholds.append(round(current, 6))
        current += step

    sweep: list[Dict[str, Any]] = []
    best_threshold = thresholds[0]
    best_metrics: Dict[str, float] = {}
    best_key = (-1.0, -1.0, -1.0)
    for threshold in thresholds:
        preds = threshold_predictions(prob_sarcastic, threshold)
        metrics = compute_binary_metrics(y_true, preds)
        row = {"threshold": threshold, **metrics}
        sweep.append(row)
        # Primary: F1. Tie-break: precision, then accuracy. This avoids choosing
        # an overly aggressive threshold when F1 is identical.
        key = (float(metrics["f1"]), float(metrics["precision"]), float(metrics["accuracy"]))
        if key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics, sweep


def softmax_class1(logits: Any) -> list[float]:
    import numpy as np

    arr = np.asarray(logits)
    shifted = arr - arr.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return [float(x) for x in probs[:, 1]]


def labels_from_dataset(dataset: Any) -> list[int]:
    return [int(x) for x in dataset["labels"]]


def texts_from_raw(raw_split: Any, text_column: str, limit: Optional[int]) -> list[str]:
    split = raw_split
    if limit is not None:
        split = split.select(range(min(limit, len(split))))
    return [str(x) for x in split[text_column]]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_table_row(table_path: Path, row: Mapping[str, Any]) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if table_path.exists():
        with table_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        fieldnames += [key for key in row.keys() if key not in fieldnames]
        with table_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)
    else:
        with table_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)


def build_prediction_rows(
    *,
    split: str,
    texts: Sequence[str],
    y_true: Sequence[int],
    prob_sarcastic: Sequence[float],
    default_preds: Sequence[int],
    tuned_preds: Sequence[int],
    threshold: float,
    model_name: str,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for idx, (text, true, prob, pred_default, pred_tuned) in enumerate(
        zip(texts, y_true, prob_sarcastic, default_preds, tuned_preds)
    ):
        rows.append(
            {
                "split": split,
                "sample_idx": idx,
                "text": text,
                "true_label": true,
                "prob_sarcastic": round(float(prob), 6),
                "pred_default": int(pred_default),
                "pred_tuned": int(pred_tuned),
                "threshold_tuned": threshold,
                "default_correct": int(true == pred_default),
                "tuned_correct": int(true == pred_tuned),
                "model_name": model_name,
            }
        )
    return rows


def train_and_optimize(args: argparse.Namespace) -> Dict[str, Any]:
    if args.disable_tqdm:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TQDM_DISABLE", "1")

    import datasets
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

    if args.disable_tqdm and hasattr(datasets, "disable_progress_bar"):
        datasets.disable_progress_bar()

    dataset_config = get_dataset_config(args.dataset)
    model_name = resolve_model_name(args.model)
    output_dir = Path(args.output_dir or default_output_dir(args))
    model_output_dir = Path(args.model_output_dir or default_model_output_dir(args))
    table_path = Path(effective_table_path(args))
    output_dir.mkdir(parents=True, exist_ok=True)

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
        tokenized["validation"] = tokenized["validation"].select(range(min(args.max_eval_samples, len(tokenized["validation"]))))
    if args.max_predict_samples:
        tokenized["test"] = tokenized["test"].select(range(min(args.max_predict_samples, len(tokenized["test"]))))

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }

    fp16 = bool(args.fp16 and torch.cuda.is_available())
    training_args_kwargs = {
        "output_dir": str(model_output_dir),
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
        "disable_tqdm": args.disable_tqdm,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "auto_find_batch_size": args.auto_find_batch_size,
    }
    training_args = TrainingArguments(**filter_supported_kwargs(TrainingArguments, training_args_kwargs, "TrainingArguments"))

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        **trainer_tokenizer_kwargs(Trainer, tokenizer),
        "data_collator": DataCollatorWithPadding(tokenizer),
        "compute_metrics": compute_metrics,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
    }
    trainer = Trainer(**filter_supported_kwargs(Trainer, trainer_kwargs, "Trainer"))

    print(f"[run] output_dir={output_dir}")
    print(f"[run] model_output_dir={model_output_dir}")
    print(f"[run] table_path={table_path}")
    trainer.train()

    val_output = trainer.predict(tokenized["validation"], metric_key_prefix="validation")
    test_output = trainer.predict(tokenized["test"], metric_key_prefix="test")

    val_probs = softmax_class1(val_output.predictions)
    test_probs = softmax_class1(test_output.predictions)
    val_true = labels_from_dataset(tokenized["validation"])
    test_true = labels_from_dataset(tokenized["test"])
    val_default = [int(x) for x in np.argmax(val_output.predictions, axis=-1)]
    test_default = [int(x) for x in np.argmax(test_output.predictions, axis=-1)]

    threshold, validation_tuned_metrics, sweep = tune_threshold(
        y_true=val_true,
        prob_sarcastic=val_probs,
        start=args.threshold_start,
        stop=args.threshold_stop,
        step=args.threshold_step,
    )
    val_tuned = threshold_predictions(val_probs, threshold)
    test_tuned = threshold_predictions(test_probs, threshold)

    validation_default_metrics = compute_binary_metrics(val_true, val_default)
    test_default_metrics = compute_binary_metrics(test_true, test_default)
    test_tuned_metrics = compute_binary_metrics(test_true, test_tuned)

    val_texts = texts_from_raw(raw["validation"], dataset_config.text_column, args.max_eval_samples)
    test_texts = texts_from_raw(raw["test"], dataset_config.text_column, args.max_predict_samples)
    prediction_rows = build_prediction_rows(
        split="validation",
        texts=val_texts,
        y_true=val_true,
        prob_sarcastic=val_probs,
        default_preds=val_default,
        tuned_preds=val_tuned,
        threshold=threshold,
        model_name=model_name,
    ) + build_prediction_rows(
        split="test",
        texts=test_texts,
        y_true=test_true,
        prob_sarcastic=test_probs,
        default_preds=test_default,
        tuned_preds=test_tuned,
        threshold=threshold,
        model_name=model_name,
    )

    metrics: Dict[str, Any] = {
        "validation_default": validation_default_metrics,
        "validation_tuned": validation_tuned_metrics,
        "test_default": test_default_metrics,
        "test_tuned": test_tuned_metrics,
        "selected_threshold": threshold,
        "threshold_selection_split": "validation",
        "raw_validation_metrics": val_output.metrics,
        "raw_test_metrics": test_output.metrics,
    }

    config: Dict[str, Any] = {
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
        "seed": args.seed,
        "fp16": fp16,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "auto_find_batch_size": args.auto_find_batch_size,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_predict_samples": args.max_predict_samples,
        "sample_limited": is_sample_limited(args),
    }

    row: Dict[str, Any] = {
        "run_id": run_id(args),
        "dataset": dataset_config.name,
        "model_alias": args.model,
        "model_name": model_name,
        "selected_threshold": threshold,
        "validation_default_f1": validation_default_metrics["f1"],
        "validation_tuned_f1": validation_tuned_metrics["f1"],
        "test_default_accuracy": test_default_metrics["accuracy"],
        "test_default_precision": test_default_metrics["precision"],
        "test_default_recall": test_default_metrics["recall"],
        "test_default_f1": test_default_metrics["f1"],
        "test_tuned_accuracy": test_tuned_metrics["accuracy"],
        "test_tuned_precision": test_tuned_metrics["precision"],
        "test_tuned_recall": test_tuned_metrics["recall"],
        "test_tuned_f1": test_tuned_metrics["f1"],
        "delta_f1": round(test_tuned_metrics["f1"] - test_default_metrics["f1"], 4),
        **config,
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "result_row.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "threshold_sweep.csv", sweep)
    write_csv(output_dir / "predictions.csv", prediction_rows)
    append_table_row(table_path, row)
    print(json.dumps(row, indent=2, ensure_ascii=False))
    return row


def build_progress5_commands() -> Dict[str, str]:
    commands: Dict[str, str] = {}
    common = "--epochs 100 --batch-size 32 --eval-batch-size 64 --lr-scheduler-type cosine --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --disable-tqdm"
    configs = [
        ("twitter-xlmr-large-threshold", "twitter", "xlmr-large", "1e-5", "128", "0.03", "0.0"),
        ("twitter-xlmr-base-lr5e-6-len128", "twitter", "xlmr-base", "5e-6", "128", "0.03", "0.0"),
        ("twitter-xlmr-base-lr2e-5-len128", "twitter", "xlmr-base", "2e-5", "128", "0.03", "0.0"),
        ("twitter-xlmr-base-lr1e-5-len256", "twitter", "xlmr-base", "1e-5", "256", "0.03", "0.0"),
        ("twitter-xlmr-base-lr1e-5-wd001", "twitter", "xlmr-base", "1e-5", "128", "0.01", "0.0"),
        ("twitter-xlmr-base-label-smoothing005", "twitter", "xlmr-base", "1e-5", "128", "0.03", "0.05"),
        ("reddit-xlmr-large-threshold", "reddit", "xlmr-large", "1e-5", "128", "0.03", "0.0"),
    ]
    for name, dataset, model, lr, max_len, wd, ls in configs:
        commands[name] = (
            "python scripts/run_transformer_optimization.py "
            f"--dataset {dataset} --model {model} --run-name {name} "
            f"--learning-rate {lr} --max-length {max_len} --weight-decay {wd} --label-smoothing-factor {ls} "
            f"{common} 2>&1 | tee results/logs/progress-5-optimization-{name}.log"
        )
    return commands


def print_progress5_commands() -> None:
    print("# Progress 5 transformer optimization commands")
    print("# Run one command per Colab cell. Start with the smoke command in docs/progress-5-run-guide.md.")
    for key, command in build_progress5_commands().items():
        print(f"\n# {key}")
        print(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Progress 5 IdSarcasm transformer optimization")
    parser.add_argument("--print-progress5-commands", action="store_true")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="twitter")
    parser.add_argument("--model", default="xlmr-large", help="Model alias or HuggingFace model name")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-output-dir", default=None)
    parser.add_argument("--table-path", default=DEFAULT_TABLE)
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
    parser.add_argument("--pad-to-max-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-train-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-tqdm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto-find-batch-size", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-predict-samples", type=int, default=None)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-stop", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.print_progress5_commands:
        print_progress5_commands()
    else:
        train_and_optimize(parsed)
