#!/usr/bin/env python3
"""Run Progress 5 modern local LLM zero-shot/few-shot experiments.

This runner targets LM Studio or any OpenAI-compatible local server. It is meant
for GGUF/quantized models such as Gemma, Qwen, Cendol, Bahasa-4B, or SEA-LION
loaded locally. These experiments are reported as modern local LLM comparisons,
not exact IdSarcasm paper reproduction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataset_name: str
    text_column: str
    label_column: str = "label"


DATASET_CONFIGS: Mapping[str, DatasetConfig] = {
    "twitter": DatasetConfig("twitter", "w11wo/twitter_indonesia_sarcastic", "tweet"),
    "reddit": DatasetConfig("reddit", "w11wo/reddit_indonesia_sarcastic", "text"),
}

DEFAULT_TABLE = "results/tables/modern_llm_experiments.csv"
DEFAULT_SMOKE_TABLE = "results/tables/modern_llm_smoke.csv"
DEFAULT_RESULTS_DIR = "results/modern_llm"
DEFAULT_LOGS_DIR = "results/logs"
DEFAULT_SPLIT = "test"
DEFAULT_SEED = 42

MODEL_NOTES: Mapping[str, str] = {
    "gemma-3n-e4b": "Gemma 3n E4B / GGUF loaded in LM Studio",
    "qwen3.5-4b": "Qwen3.5 4B / GGUF or local server model identifier",
    "cendol": "Cendol Indonesian LLM variant loaded locally",
    "bahasa-4b": "Bahasalab Bahasa-4B loaded locally",
}

ZERO_SHOT_PROMPT = """Klasifikasikan teks Indonesia berikut sebagai sarkastik atau tidak sarkastik.
Jawab hanya satu label: sarcastic atau not sarcastic.

Teks: {text}
Label:"""

FEW_SHOT_PROMPT = """Klasifikasikan teks Indonesia sebagai sarkastik atau tidak sarkastik.
Jawab hanya satu label: sarcastic atau not sarcastic.

Contoh:
{examples}

Sekarang klasifikasikan teks ini.
Teks: {text}
Label:"""

LABEL_TEXTS = ("not sarcastic", "sarcastic")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_for_path(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-._")
    return cleaned or "model"


def get_dataset_config(dataset: str) -> DatasetConfig:
    key = dataset.lower().strip()
    if key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose one of: {', '.join(sorted(DATASET_CONFIGS))}")
    return DATASET_CONFIGS[key]


def local_split_file(dataset: str, split: str, data_dir: Path) -> Path:
    split_name = "validation" if split == "val" else split
    return data_dir / f"{dataset}_{split_name}.csv"


def load_dataset_split(config: DatasetConfig, *, split: str, data_dir: Path):
    from datasets import load_dataset

    local_file = local_split_file(config.name, split, data_dir)
    if local_file.exists():
        print(f"[data] using cached CSV file: {local_file}")
        return load_dataset("csv", data_files={split: str(local_file)})[split]
    print(f"[data] cached split not found, loading {config.dataset_name}:{split} from HuggingFace")
    return load_dataset(config.dataset_name, split=split)


def maybe_limit_dataset(dataset: Any, max_samples: Optional[int]) -> Any:
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def strip_thinking_blocks(text: str) -> str:
    """Remove common reasoning wrappers before label parsing.

    Some Qwen/Gemma local chat models emit <think>...</think> or preface text
    before the final label. Removing these wrappers makes parsing robust without
    changing the actual prompt/evaluation path.
    """
    without_xml_think = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    without_markdown_think = re.sub(r"```.*?```", " ", without_xml_think, flags=re.DOTALL)
    return without_markdown_think.strip()


def parse_generated_label(text: str) -> Optional[int]:
    cleaned = strip_thinking_blocks(text)
    normalized = re.sub(r"\s+", " ", cleaned.strip().lower())
    normalized = normalized.strip(" .,:;!?'\"`[](){}")
    if not normalized:
        return None

    # Prefer explicit label/final-answer clauses if present.
    explicit_match = re.search(
        r"(?:label|answer|jawaban|final)\s*[:=\-]?\s*(not[_\s-]?sarcastic|non[_\s-]?sarcastic|tidak\s+sarkas(?:tik|tis|me)?|bukan\s+sarkas(?:tik|tis|me)?|non\s+sarkas(?:tik|tis|me)?|0|sarcastic|sarcasm|sarkas(?:tik|tis|me)?|1)\b",
        normalized,
    )
    if explicit_match:
        token = explicit_match.group(1).replace("_", " ").replace("-", " ")
        if re.search(r"^(not|non|tidak|bukan)|^0$", token):
            return 0
        return 1

    negative_patterns = (
        r"not[_\s-]?sarcastic",
        r"non[_\s-]?sarcastic",
        r"not\s+sarcasm",
        r"tidak\s+sarkas(?:tik|tis|me)?",
        r"bukan\s+sarkas(?:tik|tis|me)?",
        r"non\s+sarkas(?:tik|tis|me)?",
        r"tidak\s+mengandung\s+sarkas(?:tik|tis|me)?",
    )
    if re.search(r"\b(" + "|".join(negative_patterns) + r")\b", normalized):
        return 0

    positive_patterns = (
        r"sarcastic",
        r"sarcasm",
        r"sarkas(?:tik|tis|me)?",
        r"mengandung\s+sarkas(?:tik|tis|me)?",
    )
    if re.search(r"\b(" + "|".join(positive_patterns) + r")\b", normalized):
        return 1

    if re.search(r"\blabel\s*[:=]?\s*0\b|^0$", normalized):
        return 0
    if re.search(r"\blabel\s*[:=]?\s*1\b|^1$", normalized):
        return 1
    if normalized in {"yes", "ya", "iya", "true"}:
        return 1
    if normalized in {"no", "tidak", "false"}:
        return 0
    return None


def compute_binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
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


def call_openai_compatible(
    *,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    request_timeout: int,
    seed: Optional[int],
    disable_reasoning: bool,
) -> str:
    endpoint = f"{api_base.rstrip('/')}/chat/completions"
    if disable_reasoning:
        system_prompt = (
            system_prompt
            + " Do not use reasoning mode. Do not think step by step. Return the final label immediately."
        )
        # Qwen thinking models and several llama.cpp/LM Studio templates honor
        # /no_think in the prompt even when OpenAI-compatible parameters are ignored.
        user_prompt = user_prompt.rstrip() + "\n/no_think"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_reasoning:
        # Different local servers use different names. Unknown fields are ignored
        # by LM Studio/llama.cpp-style servers, but these help on compatible builds.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["enable_thinking"] = False
        payload["reasoning"] = {"effort": "none"}
    if seed is not None:
        payload["seed"] = seed
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible request failed: HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
    return response_data["choices"][0]["message"]["content"].strip()


def build_few_shot_examples(train_dataset: Any, config: DatasetConfig, shots_per_class: int) -> str:
    if shots_per_class <= 0:
        return ""
    selected: list[str] = []
    counts = {0: 0, 1: 0}
    for datum in train_dataset:
        label = int(datum[config.label_column])
        if label not in counts or counts[label] >= shots_per_class:
            continue
        label_text = LABEL_TEXTS[label]
        selected.append(f"Teks: {datum[config.text_column]}\nLabel: {label_text}")
        counts[label] += 1
        if all(count >= shots_per_class for count in counts.values()):
            break
    return "\n\n".join(selected)


def output_dir_for(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    model_slug = sanitize_for_path(args.model_alias or args.model)
    mode = "fewshot" if args.few_shot else "zeroshot"
    suffix = "smoke" if args.max_samples else "full"
    return Path(DEFAULT_RESULTS_DIR) / f"{args.dataset}-{model_slug}-{mode}-{suffix}"


def effective_table_path(args: argparse.Namespace) -> Path:
    if args.max_samples and args.table_path == DEFAULT_TABLE:
        return Path(DEFAULT_SMOKE_TABLE)
    return Path(args.table_path)


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


def run_modern_llm(args: argparse.Namespace) -> Dict[str, Any]:
    config = get_dataset_config(args.dataset)
    output_dir = output_dir_for(args)
    table_path = effective_table_path(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dataset = load_dataset_split(config, split=args.split, data_dir=Path(args.data_dir))
    split_dataset = maybe_limit_dataset(split_dataset, args.max_samples)
    few_shot_examples = ""
    if args.few_shot:
        train_dataset = load_dataset_split(config, split="train", data_dir=Path(args.data_dir))
        few_shot_examples = build_few_shot_examples(train_dataset, config, args.shots_per_class)

    api_key = args.api_key or os.environ.get(args.api_key_env, "")
    system_prompt = args.system_prompt
    run_started_at = utc_now_iso()
    start = time.perf_counter()
    predictions: list[Dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    invalid_outputs = 0
    latency_total = 0.0

    for idx, datum in enumerate(split_dataset):
        text = str(datum[config.text_column])
        if args.few_shot:
            prompt = FEW_SHOT_PROMPT.format(examples=few_shot_examples, text=text)
        else:
            prompt = ZERO_SHOT_PROMPT.format(text=text)
        call_start = time.perf_counter()
        raw_output = call_openai_compatible(
            api_base=args.api_base,
            api_key=api_key,
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            seed=args.seed,
            disable_reasoning=args.disable_reasoning,
        )
        latency = time.perf_counter() - call_start
        latency_total += latency
        parsed = parse_generated_label(raw_output)
        invalid = parsed is None
        if args.print_raw_outputs or (invalid and args.print_invalid_outputs):
            preview = raw_output.replace("\n", "\\n")
            print(f"[raw] idx={idx} parsed={parsed} invalid={invalid} output={preview[:500]}")
        if invalid:
            invalid_outputs += 1
        pred_label = args.invalid_fallback_label if parsed is None else parsed
        true_label = int(datum[config.label_column])
        y_true.append(true_label)
        y_pred.append(pred_label)
        predictions.append(
            {
                "sample_idx": idx,
                "split": args.split,
                "text": text,
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_label_text": LABEL_TEXTS[pred_label],
                "raw_output": raw_output,
                "invalid_output": invalid,
                "latency_seconds": round(latency, 4),
                "mode": "few-shot" if args.few_shot else "zero-shot",
                "shots_per_class": args.shots_per_class if args.few_shot else 0,
                "model": args.model,
            }
        )
        if args.print_every and (idx + 1) % args.print_every == 0:
            print(f"[progress] {idx + 1}/{len(split_dataset)} done")

    metrics = compute_binary_metrics(y_true, y_pred)
    runtime_seconds = time.perf_counter() - start
    avg_latency_seconds = latency_total / len(predictions) if predictions else 0.0
    run_ended_at = utc_now_iso()
    mode = "few-shot" if args.few_shot else "zero-shot"
    row: Dict[str, Any] = {
        "dataset": config.name,
        "split": args.split,
        "mode": mode,
        "model_alias": args.model_alias or args.model,
        "model_name": args.model,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "num_examples": len(split_dataset),
        "invalid_outputs": invalid_outputs,
        "runtime_seconds": round(runtime_seconds, 2),
        "avg_latency_seconds": round(avg_latency_seconds, 4),
        "sample_limited": args.max_samples is not None,
        "shots_per_class": args.shots_per_class if args.few_shot else 0,
        "api_base": args.api_base,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "disable_reasoning": args.disable_reasoning,
        "run_started_at": run_started_at,
        "run_ended_at": run_ended_at,
    }
    metrics_blob = {
        "mean": metrics,
        "run_started_at": run_started_at,
        "run_ended_at": run_ended_at,
        "runtime_seconds": round(runtime_seconds, 2),
        "avg_latency_seconds": round(avg_latency_seconds, 4),
        "invalid_outputs": invalid_outputs,
        "mode": mode,
        "few_shot_examples": few_shot_examples if args.save_few_shot_examples else "[hidden; use --save-few-shot-examples to store]",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_blob, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "result_row.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "predictions.csv", predictions)
    append_table_row(table_path, row)
    print(json.dumps(row, indent=2, ensure_ascii=False))
    return row


def print_lmstudio_commands() -> None:
    print("# Progress 5 LM Studio / GGUF commands")
    print("# First start LM Studio Local Server at http://localhost:1234/v1 and load one model.")
    examples = [
        ("gemma-3n-e4b", "local-model", "twitter", False),
        ("gemma-3n-e4b", "local-model", "twitter", True),
        ("qwen3.5-4b", "local-model", "twitter", False),
        ("cendol-or-bahasa-4b", "local-model", "twitter", True),
    ]
    for alias, model, dataset, few in examples:
        mode = "--few-shot --shots-per-class 2" if few else ""
        print(f"\n# {alias} {dataset} {'few-shot' if few else 'zero-shot'}")
        print(
            "python scripts/run_modern_llm_experiments.py "
            f"--dataset {dataset} --model {model} --model-alias {alias} {mode} "
            "--api-base http://localhost:1234/v1 --max-samples 20 --print-every 5"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Progress 5 modern local LLM experiments via LM Studio/OpenAI-compatible API")
    parser.add_argument("--print-lmstudio-commands", action="store_true")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="twitter")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model", default="local-model", help="LM Studio model identifier shown by the local server")
    parser.add_argument("--model-alias", default="", help="Readable model alias for result table, e.g. gemma-3n-e4b-q4")
    parser.add_argument("--api-base", default="http://localhost:1234/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--table-path", default=DEFAULT_TABLE)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--few-shot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shots-per-class", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--invalid-fallback-label", type=int, choices=(0, 1), default=0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--print-raw-outputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-invalid-outputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--disable-reasoning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Try to disable Qwen/Gemma thinking mode via prompt and OpenAI-compatible extra fields",
    )
    parser.add_argument("--save-few-shot-examples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--system-prompt",
        default="You classify Indonesian text for sarcasm. Answer with exactly one label: sarcastic or not sarcastic. No explanation.",
    )
    return parser


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.print_lmstudio_commands:
        print_lmstudio_commands()
    else:
        run_modern_llm(args)
