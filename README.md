# IdSarcasm Reproduction — UAS NLP

Reproduksi paper: **IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection**
(Suhartono, Wongso, Tri Handoyo — IEEE Access 2024)

## Project Structure

```
├── data/
│   ├── raw/              # Dataset asli dari HuggingFace
│   └── processed/        # Dataset setelah preprocessing
├── notebooks/            # Jupyter notebook untuk eksperimen
├── scripts/              # Python scripts untuk training & eval
├── results/
│   ├── tables/           # Hasil evaluasi (CSV/tabel)
│   └── figures/          # Grafik & visualisasi
├── docs/                 # Paper summary, rencana reproduksi
└── requirements.txt
```

## Dataset

| Dataset | Source | Size |
|---------|--------|------|
| Reddit Indonesia Sarcastic | [HuggingFace](https://huggingface.co/datasets/w11wo/reddit_indonesia_sarcastic) | 14,116 comments |
| Twitter Indonesia Sarcastic | [HuggingFace](https://huggingface.co/datasets/w11wo/twitter_indonesia_sarcastic) | 12,861 tweets |

## Models to Reproduce

- Classical ML: Logistic Regression, Naive Bayes, SVC
- Fine-tuned: IndoBERT (IndoNLU), XLM-R Base
- Zero-shot: BLOOMZ, mT0

## Progress

- [x] Progress 1: Topik & Paper Selection
- [ ] Progress 2: Dataset & Preprocessing
- [ ] Progress 3: Classical ML Baseline
- [ ] Progress 4: Transformer Fine-tuning
- [ ] Progress 5: Evaluation & Comparison
- [ ] Progress 6: Improvement Proposal & Final Report

## Reference

```bibtex
@article{10565877,
  author = {Suhartono, Derwin and Wongso, Wilson and Tri Handoyo, Alif},
  journal = {IEEE Access},
  title = {IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection},
  year = {2024},
  pages = {87323-87332},
  doi = {10.1109/ACCESS.2024.3416955}
}
```
