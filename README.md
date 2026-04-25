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
│   ├── 02_transformer_baseline_colab.ipynb  # Progress 3 Colab runner
│   └── 03_zeroshot_baseline_colab_or_lmstudio.ipynb  # Progress 4 zero-shot runner
├── scripts/
│   ├── download_data.py  # Download dataset dari HuggingFace
│   ├── run_classical_baselines.py  # Progress 2 classical ML
│   ├── run_transformer_baseline.py # Progress 3 transformer baseline
│   └── run_zeroshot_baseline.py    # Progress 4 zero-shot LLM baseline
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
│   ├── progress-3.md     # Progress 3: paper baseline complete transformer
│   ├── progress-3-local-run-guide.md  # Panduan run transformer di Colab
│   ├── progress-3-paper-baseline-complete-plan.md  # Plan 12 baseline transformer
│   ├── progress-4.md     # Progress 4 zero-shot LLM baseline
│   └── progress-4-zero-shot-run-guide.md  # Panduan run zero-shot
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

### Stretch / Progress 3
- Fine-tune baseline transformer paper-complete pada Twitter dan Reddit: 6 model × 2 dataset = 12 run.

### Progress 4
- Zero-shot LLM baseline menggunakan HuggingFace/Colab (`hf-logprobs`) atau LM Studio lokal sebagai OpenAI-compatible inference server.
- Setiap run menyimpan metrics, predictions, result row, log, dan runtime (`runtime_seconds`, `avg_latency_seconds`).

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
| 3 | Reproduksi Transformer Baseline dan Benchmark Lanjutan | ✅ | Paper baseline complete fine-tuned transformer selesai: 12/12 run pada Twitter + Reddit, hasil/log tersimpan |
| 4 | Zero-shot LLM Baseline | 🔄 | Script, notebook, dan panduan run siap; menunggu eksekusi Colab/lokal |
| 5 | Optimasi dan Eksperimen Lanjutan | ⬜ | Tuning/weighted loss/preprocessing/eksperimen lanjutan berdasarkan hasil baseline |
| 6 | Analisis Komparatif dan Finalisasi Laporan | ⬜ | Komparasi penuh, error analysis, README, laporan akhir, dan kesimpulan |

Detail dokumentasi saat ini tersedia di `docs/progress-1.md`, `docs/progress-2.md`, `docs/progress-3.md`, `docs/progress-4.md`, dan `source-code/README.md`. Untuk melihat workflow transformer baseline Progress 3, gunakan `notebooks/02_transformer_baseline_colab.ipynb` atau ikuti `docs/progress-3-paper-baseline-complete-plan.md`.

## Quick Start

```bash
git clone https://github.com/feb027/idsarcasm-reproduction.git
cd idsarcasm-reproduction
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\activate
# Linux / WSL / Colab-like shell:
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py     # Download dataset
jupyter notebook                    # Buka notebook
```

Untuk menjalankan Progress 2 di PC lokal / WSL2, ikuti panduan:
- `docs/progress-2-local-run-guide.md`
- script utama baseline: `scripts/run_classical_baselines.py`

Untuk Progress 3 transformer baseline yang sudah selesai:
- notebook Colab: `notebooks/02_transformer_baseline_colab.ipynb`
- panduan detail: `docs/progress-3-paper-baseline-complete-plan.md`
- script utama: `scripts/run_transformer_baseline.py`
- hasil utama: `results/tables/transformer_baselines.csv`
- scope selesai: 6 model transformer × 2 dataset = 12 run

Untuk Progress 4 zero-shot LLM baseline:
- script utama: `scripts/run_zeroshot_baseline.py`
- notebook Colab/LM Studio: `notebooks/03_zeroshot_baseline_colab_or_lmstudio.ipynb`
- panduan: `docs/progress-4-zero-shot-run-guide.md`
- rencana/detail: `docs/progress-4.md`
- opsi runtime: HuggingFace/Colab (`hf-logprobs`) atau LM Studio lokal via OpenAI-compatible API
- smoke test aman:
  ```bash
  python scripts/run_zeroshot_baseline.py --dataset twitter --model mt0-small --backend hf-logprobs --max-samples 8 --dtype float16 --device-map auto --disable-tqdm --write-log
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
