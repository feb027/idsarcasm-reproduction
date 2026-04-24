import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_transformer_baseline import (
    COMPLETE_PAPER_BASELINE_MODELS,
    DATASET_CONFIGS,
    MODEL_ALIASES,
    PAPER_BASELINE_MODELS,
    build_complete_paper_baseline_commands,
    build_progress3_commands,
    build_result_row,
    effective_table_path,
    filter_training_arguments_kwargs,
    get_dataset_config,
    is_sample_limited,
    parse_args,
    parse_best_metric,
    trainer_tokenizer_kwargs,
    training_strategy_kwargs,
    write_result_artifacts,
)


class TransformerBaselineUtilsTest(unittest.TestCase):
    def test_get_dataset_config_for_twitter(self):
        config = get_dataset_config("twitter")
        self.assertEqual(config.dataset_name, "w11wo/twitter_indonesia_sarcastic")
        self.assertEqual(config.text_column, "tweet")
        self.assertEqual(config.label_column, "label")

    def test_get_dataset_config_rejects_unknown_dataset(self):
        with self.assertRaises(ValueError) as ctx:
            get_dataset_config("instagram")
        self.assertIn("Unknown dataset", str(ctx.exception))
        self.assertIn("twitter", str(ctx.exception))
        self.assertIn("reddit", str(ctx.exception))

    def test_parse_best_metric_prefers_eval_f1_then_predict_f1(self):
        self.assertEqual(parse_best_metric({"eval_f1": 0.7, "predict_f1": 0.8}), 0.7)
        self.assertEqual(parse_best_metric({"predict_f1": 0.8}), 0.8)
        self.assertIsNone(parse_best_metric({"eval_accuracy": 0.9}))

    def test_training_strategy_kwargs_supports_old_and_new_transformers_names(self):
        def old_training_args(*, evaluation_strategy=None, save_strategy=None, logging_strategy=None):
            return None

        def new_training_args(*, eval_strategy=None, save_strategy=None, logging_strategy=None):
            return None

        self.assertEqual(
            training_strategy_kwargs(old_training_args),
            {"evaluation_strategy": "epoch", "save_strategy": "epoch", "logging_strategy": "epoch"},
        )
        self.assertEqual(
            training_strategy_kwargs(new_training_args),
            {"eval_strategy": "epoch", "save_strategy": "epoch", "logging_strategy": "epoch"},
        )

    def test_filter_training_arguments_kwargs_drops_unsupported_colab_keys(self):
        def colab_training_args(*, output_dir=None, learning_rate=None, eval_strategy=None):
            return None

        filtered = filter_training_arguments_kwargs(
            colab_training_args,
            {
                "output_dir": "models/transformer/test",
                "overwrite_output_dir": True,
                "learning_rate": 1e-5,
                "eval_strategy": "epoch",
            },
        )

        self.assertEqual(
            filtered,
            {
                "output_dir": "models/transformer/test",
                "learning_rate": 1e-5,
                "eval_strategy": "epoch",
            },
        )

    def test_trainer_tokenizer_kwargs_supports_old_and_new_trainer_names(self):
        tokenizer = object()

        def old_trainer(*, tokenizer=None):
            return None

        def new_trainer(*, processing_class=None):
            return None

        def minimal_trainer(*, model=None):
            return None

        self.assertEqual(trainer_tokenizer_kwargs(old_trainer, tokenizer), {"tokenizer": tokenizer})
        self.assertEqual(trainer_tokenizer_kwargs(new_trainer, tokenizer), {"processing_class": tokenizer})
        self.assertEqual(trainer_tokenizer_kwargs(minimal_trainer, tokenizer), {})

    def test_sample_limited_runs_use_smoke_table_by_default(self):
        full_args = SimpleNamespace(
            max_train_samples=None,
            max_eval_samples=None,
            max_predict_samples=None,
            table_path="results/tables/transformer_baselines.csv",
        )
        smoke_args = SimpleNamespace(
            max_train_samples=64,
            max_eval_samples=None,
            max_predict_samples=None,
            table_path="results/tables/transformer_baselines.csv",
        )
        explicit_args = SimpleNamespace(
            max_train_samples=64,
            max_eval_samples=None,
            max_predict_samples=None,
            table_path="results/tables/custom_smoke.csv",
        )

        self.assertFalse(is_sample_limited(full_args))
        self.assertTrue(is_sample_limited(smoke_args))
        self.assertEqual(effective_table_path(full_args), "results/tables/transformer_baselines.csv")
        self.assertEqual(effective_table_path(smoke_args), "results/tables/transformer_smoke.csv")
        self.assertEqual(effective_table_path(explicit_args), "results/tables/custom_smoke.csv")

    def test_default_arguments_match_paper_twitter_transformer_recipe(self):
        args = parse_args([])
        self.assertEqual(args.dataset, "twitter")
        self.assertEqual(args.max_length, 128)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.eval_batch_size, 64)
        self.assertEqual(args.learning_rate, 1e-5)
        self.assertEqual(args.lr_scheduler_type, "cosine")
        self.assertEqual(args.weight_decay, 0.03)
        self.assertEqual(args.label_smoothing_factor, 0.0)
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.seed, 42)
        self.assertTrue(args.pad_to_max_length)
        self.assertEqual(args.early_stopping_threshold, 0.01)
        self.assertTrue(args.shuffle_train_dataset)
        self.assertTrue(args.fp16)
        self.assertEqual(args.gradient_accumulation_steps, 1)
        self.assertFalse(args.gradient_checkpointing)
        self.assertFalse(args.auto_find_batch_size)

    def test_model_aliases_cover_complete_paper_baseline_models(self):
        self.assertEqual(
            COMPLETE_PAPER_BASELINE_MODELS,
            (
                "indobert-base",
                "indobert-large",
                "indobert-indolem-base",
                "mbert-base",
                "xlmr-base",
                "xlmr-large",
            ),
        )
        for alias in COMPLETE_PAPER_BASELINE_MODELS:
            self.assertIn(alias, MODEL_ALIASES)

    def test_complete_paper_baseline_commands_cover_six_models_two_datasets(self):
        commands = build_complete_paper_baseline_commands()
        self.assertEqual(len(commands), 12)
        for dataset in ("twitter", "reddit"):
            for alias in COMPLETE_PAPER_BASELINE_MODELS:
                key = f"{dataset}-{alias}"
                self.assertIn(key, commands)
                self.assertIn(f"--dataset {dataset}", commands[key])
                self.assertIn(f"--model {alias}", commands[key])
                self.assertIn("--batch-size 32", commands[key])
                self.assertIn("--eval-batch-size 64", commands[key])

    def test_progress3_official_commands_include_two_paper_baseline_models(self):
        commands = build_progress3_commands()
        self.assertEqual(PAPER_BASELINE_MODELS, ("indobert-base", "xlmr-base"))
        self.assertEqual(set(commands), {"indobert-base", "xlmr-base"})
        for command in commands.values():
            self.assertIn("--dataset twitter", command)
            self.assertIn("--epochs 100", command)
            self.assertIn("--batch-size 32", command)
            self.assertIn("--eval-batch-size 64", command)
            self.assertIn("--lr-scheduler-type cosine", command)
            self.assertIn("--pad-to-max-length", command)
            self.assertIn("--early-stopping-threshold 0.01", command)
            self.assertIn("--shuffle-train-dataset", command)
            self.assertIn("--fp16", command)

    def test_write_result_artifacts_migrates_table_schema_when_new_columns_are_added(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = f"{tmpdir}/table.csv"
            output_dir_1 = f"{tmpdir}/run1"
            output_dir_2 = f"{tmpdir}/run2"

            write_result_artifacts(
                output_dir=Path(output_dir_1),
                table_path=Path(table_path),
                metrics={"predict_f1": 0.1},
                row={"dataset": "twitter", "model_alias": "indobert-base", "f1": 0.1},
            )
            write_result_artifacts(
                output_dir=Path(output_dir_2),
                table_path=Path(table_path),
                metrics={"predict_f1": 0.2},
                row={
                    "dataset": "reddit",
                    "model_alias": "xlmr-large",
                    "f1": 0.2,
                    "gradient_checkpointing": True,
                },
            )

            with open(table_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertIn("gradient_checkpointing", rows[0])
            self.assertEqual(rows[0]["gradient_checkpointing"], "")
            self.assertEqual(rows[1]["gradient_checkpointing"], "True")

    def test_build_result_row_rounds_metrics_and_preserves_config(self):
        row = build_result_row(
            dataset="twitter",
            model_alias="indobert-base",
            model_name="indobenchmark/indobert-base-p1",
            metrics={
                "predict_accuracy": 0.8661710037,
                "predict_precision": 0.7627118644,
                "predict_recall": 0.6716417910,
                "predict_f1": 0.7142857142,
            },
            training_config={"learning_rate": 1e-5, "epochs": 5, "batch_size": 16, "max_length": 128},
        )
        self.assertEqual(row["dataset"], "twitter")
        self.assertEqual(row["model_alias"], "indobert-base")
        self.assertEqual(row["model_name"], "indobenchmark/indobert-base-p1")
        self.assertEqual(row["accuracy"], 0.8662)
        self.assertEqual(row["precision"], 0.7627)
        self.assertEqual(row["recall"], 0.6716)
        self.assertEqual(row["f1"], 0.7143)
        self.assertEqual(row["learning_rate"], 1e-5)
        self.assertEqual(row["epochs"], 5)
        self.assertEqual(row["batch_size"], 16)
        self.assertEqual(row["max_length"], 128)


if __name__ == "__main__":
    unittest.main()
