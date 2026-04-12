# IdSarcasm Reproduction — UAS NLP

Reproduksi paper: **IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection**
(Derwin Suhartono, Wilson Wongso, Alif Tri Handoyo — IEEE Access 2024)

## Paper Info
- **DOI:** [10.1109/ACCESS.2024.3416955](https://doi.org/10.1109/ACCESS.2024.3416955)
- **Original Repo:** https://github.com/w11wo/id_sarcasm
- **Published:** 20 June 2024

## Project Structure

```
├── data/
│   ├── raw/              # Dataset CSV dari HuggingFace
│   └── processed/        # Dataset setelah preprocessing
├── notebooks/
│   ├── 01_eda.ipynb      # EDA: label distribution, text length, data quality
│   └── ...
├── scripts/
│   ├── download_data.py  # Download dataset dari HuggingFace
│   └── ...
├── results/
│   ├── tables/           # Hasil evaluasi (CSV/tabel)
│   └── figures/          # Grafik & visualisasi (PNG)
├── docs/
│   ├── paper-summary.md  # Ringkasan paper + EDA results
│   ├── progress-plan.md  # Timeline & rencana 6 progress
│   ├── progress-1.md     # Dokumentasi Progress 1
│   └── progress-2.md     # Dokumentasi Progress 2
├── 10565877.pdf          # Paper asli
├── requirements.txt      # Python dependencies
└── README.md
```

## Dataset

| Dataset | Train | Val | Test | Total |
|---------|-------|-----|------|-------|
| Reddit Indonesia Sarcastic | 9,881 | 1,411 | 2,824 | 14,116 |
| Twitter Indonesia Sarcastic | 1,878 | 268 | 538 | 2,684 |

Source: [HuggingFace](https://huggingface.co/collections/w11wo/indonesian-sarcasm-detection-65840069489f3b53a0452c04)

**Catatan:** Twitter dataset di paper disebut 12,861, itu adalah total *cleaned unbalanced*. Yang di-publish ke HuggingFace adalah versi *balanced* (1:3 ratio sarcastic:non-sarcastic) = 2,684 total. Paper experiments menggunakan balanced version.

## Reproduction Scope

### Primary (wajib)
- Classical ML: Logistic Regression, Naive Bayes, SVM
- Feature: BoW (CountVectorizer) + TF-IDF
- Tokenizer: NLTK word_tokenize
- Hyperparameter tuning: GridSearchCV dengan PredefinedSplit
- Eval: F1-score (primary), accuracy, precision, recall

### Secondary
- Classical ML pada Reddit dataset

### Stretch
- Fine-tune IndoBERT Base atau XLM-R Base (via Google Colab)

## Methodology (Classical ML)

1. Load dataset dari HuggingFace
2. Tokenisasi dengan NLTK word_tokenize
3. Vectorisasi dengan BoW dan TF-IDF
4. GridSearchCV untuk hyperparameter tuning:
   - LR: C = [0.01, 0.1, 1, 10, 100]
   - SVM: C = [0.01, 0.1, 1, 10, 100], kernel = [rbf, linear]
   - NB: alpha = np.linspace(0.001, 1, 50)
5. PredefinedSplit: train+val digabung, val sebagai holdout
6. Best params dipilih berdasarkan validation
7. Evaluasi final di test set

## Progress

| # | Progress | Status | Detail |
|---|----------|--------|--------|
| 1 | Topik & Paper Selection | ✅ | Paper IdSarcasm dipilih, repo setup |
| 2 | Dataset & Preprocessing | ✅ | Download + EDA selesai |
| 3 | Classical ML Baseline | ⬜ | Reproduksi LR, NB, SVM |
| 4 | Transformer Fine-tuning | ⬜ | IndoBERT/XLM-R (stretch) |
| 5 | Evaluation & Comparison | ⬜ | Bandingkan hasil ke paper |
| 6 | Improvement & Final Report | ⬜ | Proposal improvement + laporan |

Detail dokumentasi tiap progress ada di `docs/progress-{n}.md`.

## Quick Start

```bash
git clone https://github.com/feb027/idsarcasm-reproduction.git
cd idsarcasm-reproduction
python -m venv .venv
.venv\Scripts\activate              # Windows
pip install -r requirements.txt
python scripts/download_data.py     # Download dataset
jupyter notebook                    # Buka notebook
```

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
