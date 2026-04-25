import csv
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts.run_zeroshot_baseline import (
    DATASET_CONFIGS,
    DEFAULT_BASELINE_TABLE,
    DEFAULT_SMOKE_TABLE,
    PAPER_ZERO_SHOT_MODEL_ALIASES,
    PAPER_ZERO_SHOT_MODEL_ORDER,
    PROMPTS,
    OpenAICompatiblePredictor,
    build_progress4_commands,
    build_result_row,
    compute_binary_metrics,
    effective_table_path,
    get_dataset_config,
    is_sample_limited,
    parse_generated_label,
    sanitize_for_path,
    write_result_artifacts,
)


class ZeroShotBaselineUtilsTest(unittest.TestCase):
    def test_prompts_match_upstream_paper_templates(self):
        self.assertEqual(len(PROMPTS), 5)
        self.assertEqual(PROMPTS[0], "{text} => Sarcasm:")
        self.assertEqual(PROMPTS[3], "Is the following text sarcastic?\nText: {text}\nAnswer:")
        self.assertTrue(all("{text}" in prompt for prompt in PROMPTS))

    def test_dataset_configs_cover_twitter_and_reddit(self):
        self.assertEqual(set(DATASET_CONFIGS), {"twitter", "reddit"})
        self.assertEqual(get_dataset_config("twitter").text_column, "tweet")
        self.assertEqual(get_dataset_config("reddit").text_column, "text")
        with self.assertRaises(ValueError):
            get_dataset_config("instagram")

    def test_paper_zero_shot_aliases_cover_bloomz_and_mt0_family(self):
        expected = {
            "bloomz-560m",
            "bloomz-1b1",
            "bloomz-1b7",
            "bloomz-3b",
            "bloomz-7b1",
            "mt0-small",
            "mt0-base",
            "mt0-large",
            "mt0-xl",
        }
        self.assertEqual(set(PAPER_ZERO_SHOT_MODEL_ALIASES), expected)
        self.assertEqual(tuple(PAPER_ZERO_SHOT_MODEL_ALIASES), PAPER_ZERO_SHOT_MODEL_ORDER)
        self.assertEqual(PAPER_ZERO_SHOT_MODEL_ALIASES["bloomz-560m"], "bigscience/bloomz-560m")
        self.assertEqual(PAPER_ZERO_SHOT_MODEL_ALIASES["mt0-small"], "bigscience/mt0-small")

    def test_parse_generated_label_prioritizes_not_sarcastic_phrase(self):
        self.assertEqual(parse_generated_label("not sarcastic"), 0)
        self.assertEqual(parse_generated_label("The answer is not sarcastic."), 0)
        self.assertEqual(parse_generated_label("sarcastic"), 1)
        self.assertEqual(parse_generated_label("Label: 1"), 1)
        self.assertEqual(parse_generated_label("Label: 0"), 0)
        self.assertIsNone(parse_generated_label("unclear / maybe"))

    def test_compute_binary_metrics_matches_expected_values(self):
        metrics = compute_binary_metrics(y_true=[0, 1, 1, 0], y_pred=[0, 1, 0, 0])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.6667, places=4)

    def test_sample_limited_runs_use_smoke_table_by_default(self):
        class Args:
            table_path = DEFAULT_BASELINE_TABLE
            max_samples = None

        full_args = Args()
        smoke_args = Args()
        smoke_args.max_samples = 8
        self.assertFalse(is_sample_limited(full_args))
        self.assertTrue(is_sample_limited(smoke_args))
        self.assertEqual(effective_table_path(full_args), DEFAULT_BASELINE_TABLE)
        self.assertEqual(effective_table_path(smoke_args), DEFAULT_SMOKE_TABLE)

    def test_build_result_row_records_runtime_and_prompt_metadata(self):
        row = build_result_row(
            dataset="twitter",
            backend="hf-logprobs",
            model_alias="mt0-small",
            model_name="bigscience/mt0-small",
            split="test",
            metrics={"accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.646153},
            prompt_count=5,
            num_examples=10,
            invalid_outputs=0,
            runtime_seconds=12.3456,
            avg_latency_seconds=1.23456,
            sample_limited=True,
            extra_config={"model_max_length": 512},
        )
        self.assertEqual(row["dataset"], "twitter")
        self.assertEqual(row["backend"], "hf-logprobs")
        self.assertEqual(row["f1"], 0.6462)
        self.assertEqual(row["runtime_seconds"], 12.35)
        self.assertEqual(row["avg_latency_seconds"], 1.2346)
        self.assertEqual(row["prompt_count"], 5)
        self.assertEqual(row["model_max_length"], 512)

    def test_write_result_artifacts_outputs_metrics_predictions_and_schema_migrated_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            table_path = Path(tmpdir) / "zeroshot.csv"
            row = {"dataset": "twitter", "model_alias": "mt0-small", "f1": 0.5}
            predictions = [
                {"sample_idx": 0, "prompt_id": 0, "true_label": 1, "pred_label": 1, "latency_seconds": 0.01}
            ]
            write_result_artifacts(
                output_dir=output_dir,
                table_path=table_path,
                metrics={"mean": {"f1": 0.5}},
                row=row,
                predictions=predictions,
            )
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "result_row.json").exists())
            self.assertTrue((output_dir / "predictions.csv").exists())
            with table_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model_alias"], "mt0-small")

    def test_progress4_commands_include_safe_colab_models_for_both_datasets(self):
        commands = build_progress4_commands(models=("bloomz-560m", "mt0-small"), datasets=("twitter", "reddit"))
        self.assertEqual(set(commands), {"twitter-bloomz-560m", "twitter-mt0-small", "reddit-bloomz-560m", "reddit-mt0-small"})
        for command in commands.values():
            self.assertIn("python scripts/run_zeroshot_baseline.py", command)
            self.assertIn("--backend hf-logprobs", command)
            self.assertIn("--write-log", command)
            self.assertIn("--disable-tqdm", command)

    def test_default_progress4_commands_cover_all_paper_models_on_both_datasets(self):
        commands = build_progress4_commands()
        self.assertEqual(len(commands), 18)
        expected_keys = {
            f"{dataset}-{model}"
            for dataset in ("twitter", "reddit")
            for model in PAPER_ZERO_SHOT_MODEL_ORDER
        }
        self.assertEqual(set(commands), expected_keys)
        ordered_keys = list(commands)
        self.assertEqual(ordered_keys[0], "twitter-bloomz-560m")
        self.assertEqual(ordered_keys[8], "twitter-mt0-xl")
        self.assertEqual(ordered_keys[9], "reddit-bloomz-560m")
        self.assertEqual(ordered_keys[-1], "reddit-mt0-xl")
        for model in PAPER_ZERO_SHOT_MODEL_ORDER:
            self.assertIn(f"--model {model}", commands[f"twitter-{model}"])
            self.assertIn(f"--model {model}", commands[f"reddit-{model}"])

    def test_colab_notebook_contains_all_paper_model_commands_one_by_one(self):
        notebook_path = REPO_ROOT / "notebooks" / "03_zeroshot_baseline_colab_or_lmstudio.ipynb"
        notebook_text = notebook_path.read_text(encoding="utf-8")
        for dataset in ("twitter", "reddit"):
            for model in PAPER_ZERO_SHOT_MODEL_ORDER:
                command = f"--dataset {dataset} --model {model} --backend hf-logprobs"
                self.assertIn(command, notebook_text)
        self.assertIn("18 full runs", notebook_text)
        self.assertIn("run one cell at a time", notebook_text.lower())

    def test_sanitize_for_path_removes_repo_and_api_unsafe_characters(self):
        self.assertEqual(sanitize_for_path("bigscience/mt0-small"), "bigscience-mt0-small")
        self.assertEqual(sanitize_for_path("local:model v1"), "local-model-v1")

    def test_openai_compatible_predictor_sends_chat_payload_and_parses_label(self):
        captured = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                captured["path"] = self.path
                captured["auth"] = self.headers.get("Authorization")
                captured["payload"] = json.loads(self.rfile.read(length).decode("utf-8"))
                response = {"choices": [{"message": {"content": "not sarcastic"}}]}
                encoded = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format, *args):
                return None

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            predictor = OpenAICompatiblePredictor(
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
                api_key="test-key",
                model_name="local-model",
                temperature=0.0,
                max_new_tokens=4,
                request_timeout=5,
                system_prompt="Answer with a label.",
                invalid_fallback="not_sarcastic",
            )
            result = predictor.predict_label_scores("Text: contoh => Sarcasm:")
        finally:
            server.shutdown()
            thread.join(timeout=5)

        self.assertEqual(captured["path"], "/v1/chat/completions")
        self.assertEqual(captured["auth"], "Bearer test-key")
        self.assertEqual(captured["payload"]["model"], "local-model")
        self.assertEqual(captured["payload"]["max_tokens"], 4)
        self.assertEqual(result["pred_label"], 0)
        self.assertFalse(result["invalid_output"])


if __name__ == "__main__":
    unittest.main()
