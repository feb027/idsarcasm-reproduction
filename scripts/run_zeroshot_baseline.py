#!/usr/bin/env python3
"""Run Progress 4 zero-shot LLM baselines for IdSarcasm.

Two execution modes are supported:

1. ``hf-logprobs`` — paper-faithful HuggingFace scoring. The runner scores the
   two candidate labels (``not sarcastic`` and ``sarcastic``) for each prompt and
   selects the label with the larger log probability. This follows the upstream
   IdSarcasm zero-shot script.
2. ``openai-compatible`` — practical local/API inference for LM Studio or any
   OpenAI-compatible chat-completions endpoint. This is useful for local
   quantized models, but should be reported as a local zero-shot baseline when
   the model differs from BLOOMZ/mT0.

Every run writes metrics, per-prompt predictions, a result row, optional logs,
and runtime metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROMPTS: tuple[str, ...] = (
    "{text} => Sarcasm:",
    "Text: {text} => Sarcasm:",
    "{text}\nIs this text above sarcastic or not?",
    "Is the following text sarcastic?\nText: {text}\nAnswer:",
    "Text: {text}\nPlease classify the text above for sarcasm.",
)

LABEL_TEXTS: tuple[str, str] = ("not sarcastic", "sarcastic")
DEFAULT_BASELINE_TABLE = "results/tables/zeroshot_baselines.csv"
DEFAULT_SMOKE_TABLE = "results/tables/zeroshot_smoke.csv"
DEFAULT_RESULTS_DIR = "results/zeroshot"
DEFAULT_LOGS_DIR = "results/logs"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL_MAX_LENGTH = 1024
DEFAULT_SEED = 42


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


PAPER_ZERO_SHOT_MODEL_ALIASES: Mapping[str, str] = {
    "bloomz-560m": "bigscience/bloomz-560m",
    "bloomz-1b1": "bigscience/bloomz-1b1",
    "bloomz-1b7": "bigscience/bloomz-1b7",
    "bloomz-3b": "bigscience/bloomz-3b",
    "bloomz-7b1": "bigscience/bloomz-7b1",
    "mt0-small": "bigscience/mt0-small",
    "mt0-base": "bigscience/mt0-base",
    "mt0-large": "bigscience/mt0-large",
    "mt0-xl": "bigscience/mt0-xl",
}

SAFE_COLAB_MODELS: tuple[str, ...] = ("bloomz-560m", "mt0-small")


class Tee:
    """Write stdout/stderr both to terminal and to a log file."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class BasePredictor:
    def predict_label_scores(self, prompt_text: str) -> Dict[str, Any]:
        raise NotImplementedError


class HfLogprobPredictor(BasePredictor):
    def __init__(
        self,
        *,
        model_name: str,
        model_max_length: int,
        dtype: str,
        load_8bit: bool,
        device_map: str,
    ) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.model_max_length = model_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        model_class = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }
        if device_map != "none":
            model_kwargs["device_map"] = device_map
        if load_8bit:
            model_kwargs["load_in_8bit"] = True
        if dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = model_class.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        self.is_encoder_decoder = bool(self.model.config.is_encoder_decoder)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if device_map == "none":
            self.model.to(self.device)
        encoded_labels = self.tokenizer(list(LABEL_TEXTS), return_tensors="pt", padding=True, add_special_tokens=False)
        self.label_ids = encoded_labels["input_ids"]
        self.label_attn = encoded_labels["attention_mask"]

    def _score_seq2seq_label(self, prompt_text: str, label_idx: int) -> float:
        import torch.nn.functional as F

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_max_length,
        ).to(self.model.device)
        label_ids = self.label_ids[label_idx].view(1, -1).to(self.model.device)
        label_attn = self.label_attn[label_idx].view(1, -1).to(self.model.device)
        logits = self.model(**inputs, labels=label_ids).logits
        logprobs = F.log_softmax(logits, dim=2)
        gathered = self.torch.gather(logprobs, 2, label_ids.unsqueeze(2)) * label_attn.unsqueeze(2)
        return float(gathered.sum().detach().cpu())

    def _score_causal_label(self, prompt_text: str, label: str) -> float:
        import torch.nn.functional as F

        # Mirrors the upstream IdSarcasm zero-shot script: score the full
        # prompt+label sequence and select the candidate with the highest score.
        candidate_text = f"{prompt_text} {label}"
        inputs = self.tokenizer(
            candidate_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_max_length,
        ).to(self.model.device)
        input_ids = inputs["input_ids"]
        output_ids = input_ids[:, 1:]
        logits = self.model(**inputs, labels=input_ids).logits
        logprobs = F.log_softmax(logits, dim=2)
        gathered = self.torch.gather(logprobs[:, :-1, :], 2, output_ids.unsqueeze(2))
        return float(gathered.sum().detach().cpu())

    def predict_label_scores(self, prompt_text: str) -> Dict[str, Any]:
        with self.torch.no_grad():
            if self.is_encoder_decoder:
                scores = [self._score_seq2seq_label(prompt_text, idx) for idx in range(len(LABEL_TEXTS))]
            else:
                scores = [self._score_causal_label(prompt_text, label) for label in LABEL_TEXTS]
        pred_label = int(max(range(len(scores)), key=lambda idx: scores[idx]))
        return {
            "pred_label": pred_label,
            "score_not_sarcastic": scores[0],
            "score_sarcastic": scores[1],
            "raw_output": LABEL_TEXTS[pred_label],
            "invalid_output": False,
        }


class OpenAICompatiblePredictor(BasePredictor):
    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model_name: str,
        temperature: float,
        max_new_tokens: int,
        request_timeout: int,
        system_prompt: str,
        invalid_fallback: str,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.request_timeout = request_timeout
        self.system_prompt = system_prompt
        self.invalid_fallback = invalid_fallback

    def _fallback_label(self) -> int:
        return 1 if self.invalid_fallback == "sarcastic" else 0

    def predict_label_scores(self, prompt_text: str) -> Dict[str, Any]:
        endpoint = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_text},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible request failed: HTTP {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
        raw_output = response_data["choices"][0]["message"]["content"].strip()
        parsed = parse_generated_label(raw_output)
        invalid = parsed is None
        pred_label = self._fallback_label() if parsed is None else parsed
        return {
            "pred_label": pred_label,
            "score_not_sarcastic": None,
            "score_sarcastic": None,
            "raw_output": raw_output,
            "invalid_output": invalid,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_for_path(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-._")
    return sanitized or "model"


def get_dataset_config(dataset: str) -> DatasetConfig:
    key = dataset.lower().strip()
    if key not in DATASET_CONFIGS:
        options = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset '{dataset}'. Choose one of: {options}")
    return DATASET_CONFIGS[key]


def resolve_model_name(model_alias_or_name: str) -> str:
    return PAPER_ZERO_SHOT_MODEL_ALIASES.get(model_alias_or_name, model_alias_or_name)


def parse_generated_label(text: str) -> Optional[int]:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = normalized.strip(" .,:;!?'\"`[](){}")
    if not normalized:
        return None
    if re.search(r"\b(not\s+sarcastic|non[-\s]?sarcastic|non\s+sarkastik|tidak\s+sarkastik|bukan\s+sarkastik)\b", normalized):
        return 0
    if re.search(r"\b(sarcastic|sarkastik|sarcasm|sarkasme)\b", normalized):
        return 1
    if re.search(r"\blabel\s*[:=]?\s*0\b|^0$", normalized):
        return 0
    if re.search(r"\blabel\s*[:=]?\s*1\b|^1$", normalized):
        return 1
    return None


def compute_binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": round(correct / len(y_true), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def is_sample_limited(args: argparse.Namespace) -> bool:
    return getattr(args, "max_samples", None) is not None


def effective_table_path(args: argparse.Namespace) -> str:
    if is_sample_limited(args) and args.table_path == DEFAULT_BASELINE_TABLE:
        return DEFAULT_SMOKE_TABLE
    return args.table_path


def default_output_dir(args: argparse.Namespace) -> str:
    alias = sanitize_for_path(args.model)
    return f"{DEFAULT_RESULTS_DIR}/{args.dataset}-{args.backend}-{alias}"


def default_log_path(args: argparse.Namespace) -> str:
    alias = sanitize_for_path(args.model)
    suffix = "smoke" if is_sample_limited(args) else "full"
    return f"{DEFAULT_LOGS_DIR}/progress-4-zeroshot-{args.dataset}-{args.backend}-{alias}-{suffix}.log"


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


def build_result_row(
    *,
    dataset: str,
    backend: str,
    model_alias: str,
    model_name: str,
    split: str,
    metrics: Mapping[str, float],
    prompt_count: int,
    num_examples: int,
    invalid_outputs: int,
    runtime_seconds: float,
    avg_latency_seconds: float,
    sample_limited: bool,
    extra_config: Mapping[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset": dataset,
        "backend": backend,
        "model_alias": model_alias,
        "model_name": model_name,
        "split": split,
        "accuracy": round(float(metrics.get("accuracy", 0.0)), 4),
        "precision": round(float(metrics.get("precision", 0.0)), 4),
        "recall": round(float(metrics.get("recall", 0.0)), 4),
        "f1": round(float(metrics.get("f1", 0.0)), 4),
        "prompt_count": prompt_count,
        "num_examples": num_examples,
        "invalid_outputs": invalid_outputs,
        "runtime_seconds": round(runtime_seconds, 2),
        "avg_latency_seconds": round(avg_latency_seconds, 4),
        "sample_limited": sample_limited,
    }
    row.update(extra_config)
    return row


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_table_row(table_path: Path, row: Mapping[str, Any]) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if table_path.exists():
        with table_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            existing_fieldnames = list(reader.fieldnames or [])
        fieldnames = existing_fieldnames + [key for key in row.keys() if key not in existing_fieldnames]
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


def write_result_artifacts(
    *,
    output_dir: Path,
    table_path: Path,
    metrics: Mapping[str, Any],
    row: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "result_row.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv_rows(output_dir / "predictions.csv", predictions)
    _append_table_row(table_path, row)


def build_progress4_commands(
    *,
    models: Sequence[str] = SAFE_COLAB_MODELS,
    datasets: Sequence[str] = ("twitter", "reddit"),
) -> Dict[str, str]:
    commands: Dict[str, str] = {}
    for dataset in datasets:
        for model in models:
            output_dir = f"results/zeroshot/{dataset}-hf-logprobs-{sanitize_for_path(model)}"
            commands[f"{dataset}-{model}"] = (
                "python scripts/run_zeroshot_baseline.py "
                f"--dataset {dataset} --model {model} --backend hf-logprobs "
                f"--output-dir {output_dir} "
                "--dtype float16 --device-map auto --disable-tqdm --write-log"
            )
    return commands


def create_predictor(args: argparse.Namespace, model_name: str) -> BasePredictor:
    if args.backend == "hf-logprobs":
        return HfLogprobPredictor(
            model_name=model_name,
            model_max_length=args.model_max_length,
            dtype=args.dtype,
            load_8bit=args.load_8bit,
            device_map=args.device_map,
        )
    if args.backend == "openai-compatible":
        api_key = args.api_key or os.environ.get(args.api_key_env, "")
        return OpenAICompatiblePredictor(
            api_base=args.api_base,
            api_key=api_key,
            model_name=model_name,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            request_timeout=args.request_timeout,
            system_prompt=args.system_prompt,
            invalid_fallback=args.invalid_fallback,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def run_zeroshot(args: argparse.Namespace) -> Dict[str, Any]:
    if args.disable_tqdm:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TQDM_DISABLE", "1")

    if args.disable_tqdm:
        iterator_factory = lambda iterable, **_: iterable
    else:
        from tqdm.auto import tqdm

        iterator_factory = tqdm

    run_started_at = utc_now_iso()
    start = time.perf_counter()
    dataset_config = get_dataset_config(args.dataset)
    model_name = resolve_model_name(args.model)
    output_dir = Path(args.output_dir or default_output_dir(args))
    table_path = Path(effective_table_path(args))

    print(f"[run] started_at={run_started_at}")
    print(f"[run] backend={args.backend} dataset={dataset_config.name} split={args.split} model={model_name}")
    print(f"[run] output_dir={output_dir}")
    print(f"[run] table_path={table_path}")

    dataset = load_dataset_split(dataset_config, split=args.split, data_dir=Path(args.data_dir))
    dataset = maybe_limit_dataset(dataset, args.max_samples)
    print(f"[data] examples={len(dataset)} sample_limited={args.max_samples is not None}")

    predictor = create_predictor(args, model_name)

    all_predictions: List[Dict[str, Any]] = []
    metrics_by_prompt: Dict[str, Dict[str, float]] = {}
    latency_total = 0.0
    invalid_outputs = 0

    for prompt_id, prompt_template in enumerate(PROMPTS):
        y_true: List[int] = []
        y_pred: List[int] = []
        iterable = iterator_factory(dataset, desc=f"prompt-{prompt_id}")
        for sample_idx, datum in enumerate(iterable):
            text_value = str(datum[dataset_config.text_column])
            prompt_text = prompt_template.format(text=text_value)
            prediction_start = time.perf_counter()
            result = predictor.predict_label_scores(prompt_text)
            latency = time.perf_counter() - prediction_start
            latency_total += latency
            true_label = int(datum[dataset_config.label_column])
            pred_label = int(result["pred_label"])
            invalid = bool(result.get("invalid_output", False))
            invalid_outputs += int(invalid)
            y_true.append(true_label)
            y_pred.append(pred_label)
            all_predictions.append(
                {
                    "sample_idx": sample_idx,
                    "prompt_id": prompt_id,
                    "prompt_template": prompt_template,
                    "text": text_value,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "pred_label_text": LABEL_TEXTS[pred_label],
                    "raw_output": result.get("raw_output"),
                    "score_not_sarcastic": result.get("score_not_sarcastic"),
                    "score_sarcastic": result.get("score_sarcastic"),
                    "invalid_output": invalid,
                    "latency_seconds": round(latency, 4),
                    "backend": args.backend,
                    "model_name": model_name,
                }
            )
        prompt_metrics = compute_binary_metrics(y_true, y_pred)
        metrics_by_prompt[str(prompt_id)] = prompt_metrics
        print(f"[metrics] prompt_id={prompt_id} {prompt_metrics}")

    mean_metrics = {
        metric: round(sum(prompt_metrics[metric] for prompt_metrics in metrics_by_prompt.values()) / len(metrics_by_prompt), 4)
        for metric in ("accuracy", "precision", "recall", "f1")
    }
    runtime_seconds = time.perf_counter() - start
    run_ended_at = utc_now_iso()
    avg_latency = latency_total / len(all_predictions) if all_predictions else 0.0

    metrics: Dict[str, Any] = {
        "mean": mean_metrics,
        "by_prompt": metrics_by_prompt,
        "run_started_at": run_started_at,
        "run_ended_at": run_ended_at,
        "runtime_seconds": round(runtime_seconds, 2),
        "avg_latency_seconds": round(avg_latency, 4),
        "num_prediction_calls": len(all_predictions),
        "invalid_outputs": invalid_outputs,
    }
    row = build_result_row(
        dataset=dataset_config.name,
        backend=args.backend,
        model_alias=args.model,
        model_name=model_name,
        split=args.split,
        metrics=mean_metrics,
        prompt_count=len(PROMPTS),
        num_examples=len(dataset),
        invalid_outputs=invalid_outputs,
        runtime_seconds=runtime_seconds,
        avg_latency_seconds=avg_latency,
        sample_limited=is_sample_limited(args),
        extra_config={
            "model_max_length": args.model_max_length,
            "dtype": args.dtype,
            "load_8bit": args.load_8bit,
            "device_map": args.device_map,
            "temperature": args.temperature if args.backend == "openai-compatible" else None,
            "max_new_tokens": args.max_new_tokens if args.backend == "openai-compatible" else None,
            "run_started_at": run_started_at,
            "run_ended_at": run_ended_at,
        },
    )
    write_result_artifacts(
        output_dir=output_dir,
        table_path=table_path,
        metrics=metrics,
        row=row,
        predictions=all_predictions,
    )
    print(f"[run] ended_at={run_ended_at} runtime_seconds={runtime_seconds:.2f}")
    print(json.dumps(row, indent=2, ensure_ascii=False))
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run IdSarcasm Progress 4 zero-shot LLM baseline")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="twitter")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--backend", choices=("hf-logprobs", "openai-compatible"), default="hf-logprobs")
    parser.add_argument("--model", default="mt0-small", help="Model alias or model name")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--table-path", default=DEFAULT_BASELINE_TABLE)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test limit over the selected split")
    parser.add_argument("--model-max-length", type=int, default=DEFAULT_MODEL_MAX_LENGTH)
    parser.add_argument("--dtype", default="float16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--load-8bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device-map", default="auto", help="HuggingFace device_map; use 'none' to move model to cuda/cpu manually")
    parser.add_argument("--disable-tqdm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-log", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--api-base", default="http://localhost:1234/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument(
        "--system-prompt",
        default="You classify Indonesian text for sarcasm. Answer with exactly one label: sarcastic or not sarcastic.",
    )
    parser.add_argument("--invalid-fallback", choices=("not_sarcastic", "sarcastic"), default="not_sarcastic")
    return parser


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    if args.write_log:
        log_path = Path(args.log_path or default_log_path(args))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            tee_stdout = Tee(sys.stdout, log_handle)
            tee_stderr = Tee(sys.stderr, log_handle)
            with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                print(f"[log] writing to {log_path}")
                return run_zeroshot(args)
    return run_zeroshot(args)


if __name__ == "__main__":
    main()
