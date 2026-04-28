# IdSarcasm Reproduction and Transformer Optimization

Repository ini berisi reproduksi dan optimasi paper **“IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection”** (Suhartono, Wongso, Handoyo — IEEE Access 2024).

**Judul proyek:** Optimasi Performa Model Transformer dalam Klasifikasi Sarkasme Teks Berbahasa Indonesia Berdasarkan Benchmark IdSarcasm

- Paper DOI: [10.1109/ACCESS.2024.3416955](https://doi.org/10.1109/ACCESS.2024.3416955)
- Original repository: <https://github.com/w11wo/id_sarcasm>
- Dashboard: [https://feb027.github.io/idsarcasm-reproduction/](https://feb027.github.io/idsarcasm-reproduction/)
- Final report: [`docs/laporan-proyek.md`](docs/laporan-proyek.md)
- Reproducibility guide: [`docs/reproducibility.md`](docs/reproducibility.md)

## Ringkasan Hasil

| Dataset | Metode terbaik | F1 baseline | F1 setelah optimasi | Target paper |
|---|---|---:|---:|---:|
| Twitter | XLM-R Large + LR 2e-5 | 0.7226 | 0.7905 | 0.7692 |
| Reddit | XLM-R Large + threshold tuning | 0.6117 | 0.6241 | 0.6274 |

Temuan utama:

- XLM-R Large menjadi model terbaik pada Twitter dan Reddit; hasil Twitter setelah tuning learning rate sudah melampaui paper.
- Optimasi lanjutan XLM-R Large menaikkan F1 Twitter dari `0.7226` ke `0.7905`, sehingga melampaui target paper `0.7692`. Pada Reddit, threshold tuning menaikkan F1 dari `0.6117` ke `0.6241`, masih sedikit di bawah paper `0.6274`.
- Classical ML tetap kompetitif pada Twitter, terutama BoW Logistic Regression (`F1 = 0.7206`).
- Zero-shot LLM sesuai paper berhasil direproduksi dekat dengan target paper, tetapi performanya tetap rendah (`F1 ≈ 0.39–0.40`).
- Modern local LLM via LM Studio lebih baik dari zero-shot paper pada Twitter, tetapi belum mendekati fine-tuned transformer. Qwen3.5-4B few-shot memperoleh `F1 = 0.4755`.

## Struktur Repository

```text
├── data/                 # Dataset lokal, tidak dikomit jika besar
├── docs/
│   ├── laporan-proyek.md # Laporan akhir
│   ├── reproducibility.md
│   ├── paper-summary.md
│   └── progress/         # Catatan progress dan run guide pendukung
├── notebooks/            # Notebook EDA, Colab, dan eksperimen
├── results/
│   ├── figures/          # Figure laporan
│   ├── tables/           # Tabel hasil utama
│   ├── transformer/      # Output transformer baseline
│   ├── zeroshot/         # Output zero-shot LLM
│   ├── optimization/     # Output optimasi transformer
│   └── modern_llm/       # Output eksperimen local LLM
├── scripts/              # Runner eksperimen dan generator analisis
├── source-code/          # Snapshot repo paper asli sebagai referensi
└── tests/                # Unit tests untuk runner
```

## Dataset

| Dataset | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Reddit Indonesia Sarcastic | 9,881 | 1,411 | 2,824 | 14,116 |
| Twitter Indonesia Sarcastic | 1,878 | 268 | 538 | 2,684 |

Dataset berasal dari koleksi HuggingFace IdSarcasm. Kedua dataset memiliki proporsi kelas 25% sarkastik dan 75% non-sarkastik pada setiap split.

## Eksperimen

| Tahap | Isi | Status |
|---|---|---|
| Classical ML | Logistic Regression, Naive Bayes, SVM + BoW/TF-IDF | selesai |
| Transformer baseline | IndoBERT, mBERT, XLM-R pada Twitter dan Reddit | selesai |
| Zero-shot LLM | BLOOMZ dan mT0 sesuai paper | selesai sebagian penuh: Twitter 9/9, Reddit 5/9 |
| Optimasi transformer | Threshold tuning XLM-R Large + screening XLM-R Base | selesai |
| Modern local LLM | Qwen3.5-4B dan Gemma 4 E4B via LM Studio | selesai pada Twitter |
| Final analysis | Perbandingan akhir, figure final, laporan akhir | selesai |
| Static dashboard | GitHub Pages dashboard + error explorer dari hasil final | selesai |

## Quick Start

```bash
git clone https://github.com/feb027/idsarcasm-reproduction.git
cd idsarcasm-reproduction
python -m venv .venv
source .venv/bin/activate      # Linux/WSL
# .venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
python scripts/download_data.py
```


Catatan environment: baseline classical ML dapat dijalankan lokal, transformer membutuhkan Colab/GPU, sedangkan eksperimen local LLM membutuhkan LM Studio atau endpoint OpenAI-compatible lokal. Detail lengkap ada di [`docs/reproducibility.md`](docs/reproducibility.md).

Generate ulang tabel/figure final dan data dashboard dari hasil yang sudah ada:

```bash
python scripts/generate_progress5_analysis.py
python scripts/generate_final_analysis.py
python scripts/generate_dashboard_data.py
```

Jalankan test:

```bash
python -m pytest tests/ -q
```

## Output Utama

```text
results/tables/classical_baselines_*.csv
results/tables/transformer_baselines.csv
results/tables/zeroshot_baselines.csv
results/tables/optimization_runs.csv
results/tables/modern_llm_experiments.csv
results/tables/final_method_comparison.csv
results/figures/final_*.png
dashboard/data/dashboard-data.json
```

## Citation

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
