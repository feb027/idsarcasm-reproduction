# Model Card — Final IdSarcasm Reproduction Runs

## Model Summary

This model card summarizes the selected final runs in this repository. The project reproduces and optimizes the IdSarcasm benchmark for Indonesian sarcasm detection.

| Dataset | Selected run | Strategy | Test F1 | Paper F1 | Status |
|---|---|---|---:|---:|---|
| Twitter | XLM-R Large lr=2e-5 | Default threshold | 0.7905 | 0.7692 | Above paper |
| Reddit | XLM-R Large threshold tuning | Validation-tuned threshold | 0.6241 | 0.6274 | Slightly below paper |

## Intended Use

- Academic reproduction and analysis of Indonesian sarcasm detection.
- Benchmark comparison between classical ML, fine-tuned transformers, zero-shot LLMs, and local modern LLMs.
- Error analysis using committed prediction CSVs and the static dashboard.

## Out-of-Scope Use

- Production moderation without further validation.
- High-stakes decision-making.
- General Indonesian sentiment analysis outside the IdSarcasm dataset domain.

## Dataset

The benchmark uses two IdSarcasm subsets:

| Dataset | Train | Validation | Test | Label ratio |
|---|---:|---:|---:|---|
| Twitter | 1,878 | 268 | 538 | 25% sarcastic / 75% non-sarcastic |
| Reddit | 9,881 | 1,411 | 2,824 | 25% sarcastic / 75% non-sarcastic |

## Metrics

| Dataset | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Twitter | 0.8848 | 0.7222 | 0.8731 | 0.7905 | 359 | 45 | 17 | 117 |
| Reddit | 0.7875 | 0.5596 | 0.7054 | 0.6241 | 1726 | 392 | 208 | 498 |


## Limitations

- Most transformer experiments use a single seed, so seed variance is not measured.
- Checkpoints are not committed because model files are large; reproducibility relies on scripts, notebooks, logs, prediction CSVs, and result tables.
- Sarcasm can depend on social context, speaker intent, or prior conversation. A text-only model may miss those signals.
- Reddit remains slightly below the paper target, so the reproduction is strongest on Twitter.
- Local modern LLM experiments were only completed on Twitter due to inference time.

## Ethical Notes

Sarcasm detection can misread humor, dialect, political context, or informal Indonesian expressions. Predictions should be interpreted as benchmark outputs, not final judgments about user intent.

## Artifacts

- Final report: [`docs/laporan-proyek.md`](laporan-proyek.md)
- Error analysis: [`docs/error-analysis.md`](error-analysis.md)
- Dashboard: <https://feb027.github.io/idsarcasm-reproduction/>
- Final confusion matrix: [`results/figures/final_confusion_matrices.png`](../results/figures/final_confusion_matrices.png)
- Final comparison table: [`results/tables/final_method_comparison.csv`](../results/tables/final_method_comparison.csv)
