# IdSarcasm Reproduction — UAS NLP

Reproduksi paper: **IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection**
(Derwin Suhartono, Wilson Wongso, Alif Tri Handoyo — IEEE Access 2024)

## Paper Info
- **Paper acuan:** IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection
- **Judul proyek (Google Sheets):** Optimasi Performa Model Transformer dalam Klasifikasi Sarkasme Teks Berbahasa Indonesia Berdasarkan Benchmark IdSarcasm
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
├── source-code/
│   ├── README.md         # Penjelasan snapshot source code upstream
│   └── original-id-sarcasm/  # Snapshot repo asli paper (read-only reference)
├── docs/
│   ├── paper-summary.md  # Ringkasan paper + catatan EDA/progress
│   ├── progress-plan.md  # Timeline & rencana 6 progress (revisi)
│   ├── progress-1.md     # Dokumentasi Progress 1
│   ├── progress-2.md     # Progress 2 gabungan: dataset, EDA, baseline classical ML
│   ├── progress-3.md     # Rencana/konfigurasi Progress 3 transformer baseline
│   └── progress-3-local-run-guide.md  # Panduan run transformer di Colab
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

> Revisi struktur: Progress 2 lama (dataset + EDA) dan Progress 3 lama (classical ML baseline) sekarang digabung menjadi Progress 2 baru. Saat ini baseline classical ML sudah berhasil dijalankan, dan source snapshot repo asli sudah disimpan untuk fase berikutnya.

| # | Progress | Status | Detail |
|---|----------|--------|--------|
| 1 | Topik, Paper, dan Target Reproduksi | ✅ | Paper final, repo setup, scope reproduksi ditetapkan |
| 2 | Dataset, EDA, dan Baseline Classical ML | ✅ | EDA + baseline Twitter/Reddit sudah jalan, hasil tabel tersimpan |
| 3 | Reproduksi Transformer Baseline dan Benchmark Lanjutan | 🔄 | Runner, notebook Colab, dan panduan Progress 3 sudah siap; menunggu eksekusi GPU |
| 4 | Optimasi Transformer Terarah | ⬜ | Tuning konfigurasi transformer untuk meningkatkan performa |
| 5 | Analisis Komparatif dan Error Analysis | ⬜ | Komparasi penuh, confusion matrix, dan analisis error |
| 6 | Finalisasi Laporan, Repo, dan Narasi Hasil | ⬜ | Rapikan hasil akhir, README, laporan, dan kesimpulan |

Detail dokumentasi saat ini tersedia di `docs/progress-1.md`, `docs/progress-2.md`, `docs/progress-3.md`, dan `source-code/README.md`. Untuk menjalankan transformer baseline Progress 3, gunakan `notebooks/02_transformer_baseline_colab.ipynb` atau ikuti `docs/progress-3-local-run-guide.md`.

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

Untuk menjalankan Progress 2 di PC lokal / WSL2, ikuti panduan:
- `docs/progress-2-local-run-guide.md`
- script utama baseline: `scripts/run_classical_baselines.py`

Untuk mulai Progress 3 transformer baseline di Google Colab GPU:
- notebook siap jalan: `notebooks/02_transformer_baseline_colab.ipynb`
- panduan detail: `docs/progress-3-local-run-guide.md`
- script utama: `scripts/run_transformer_baseline.py`

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
