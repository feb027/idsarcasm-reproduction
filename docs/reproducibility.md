# Reproducibility Guide

Dokumen ini merangkum cara menjalankan ulang eksperimen utama tanpa membuka catatan progress lama satu per satu.

## Ringkasan Environment

| Jalur eksperimen | Environment yang disarankan | Catatan |
|---|---|---|
| Classical ML | PC lokal/WSL/Colab CPU | Relatif ringan |
| Transformer baseline | Google Colab GPU | Membutuhkan fine-tuning |
| Zero-shot LLM | Google Colab GPU | Reddit lebih lama dari Twitter |
| Transformer optimization | Google Colab GPU | Untuk XLM-R Large dan XLM-R Base |
| Modern local LLM | Windows + LM Studio | Memakai model GGUF dan endpoint lokal |
| Figure/tabel final | PC lokal/WSL | Membaca CSV hasil yang sudah tersedia |

## Setup

```bash
git clone https://github.com/feb027/idsarcasm-reproduction.git
cd idsarcasm-reproduction
python -m venv .venv
source .venv/bin/activate  # Linux/WSL
# .venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
python scripts/download_data.py
```

## 1. Classical ML Baseline

```bash
python scripts/run_classical_baselines.py --dataset twitter
python scripts/run_classical_baselines.py --dataset reddit
```

Output utama:

```text
results/tables/classical_baselines_twitter.csv
results/tables/classical_baselines_reddit.csv
```

## 2. Transformer Baseline

Transformer baseline dijalankan di Google Colab/GPU karena membutuhkan fine-tuning.

Notebook:

```text
notebooks/02_transformer_baseline_colab.ipynb
```

Output utama:

```text
results/tables/transformer_baselines.csv
results/transformer/
```

## 3. Zero-shot LLM Baseline

Notebook:

```text
notebooks/03_zeroshot_baseline_colab_or_lmstudio.ipynb
```

Cetak command paper-complete:

```bash
python scripts/run_zeroshot_baseline.py --print-paper-commands
```

Output utama:

```text
results/tables/zeroshot_baselines.csv
results/zeroshot/
```

## 4. Transformer Optimization

Cetak command Progress 5:

```bash
python scripts/run_transformer_optimization.py --print-progress5-commands
```

Output utama:

```text
results/tables/optimization_runs.csv
results/optimization/
```

## 5. Modern Local LLM via LM Studio

Model GGUF dijalankan dari LM Studio di Windows, lalu script memanggil endpoint OpenAI-compatible.

```powershell
$env:API_BASE = "http://localhost:1234/v1"
$env:MODEL_ID = "nama-model-di-lm-studio"
python scripts/run_modern_llm_experiments.py --print-lmstudio-commands
```

Output utama:

```text
results/tables/modern_llm_experiments.csv
results/modern_llm/
```

## 6. Optimasi Lanjutan Twitter

Run final yang melampaui paper pada Twitter:

```bash
python scripts/run_transformer_optimization.py \
  --dataset twitter \
  --model xlmr-large \
  --run-name twitter-xlmr-large-lr2e-5-len128 \
  --learning-rate 2e-5 \
  --max-length 128 \
  --batch-size 8 \
  --eval-batch-size 32 \
  --gradient-accumulation-steps 4 \
  --gradient-checkpointing \
  --auto-find-batch-size \
  --fp16
```

Output utama:

```text
results/optimization/twitter-xlmr-large-lr2e-5-len128/
```

## 7. Derived Analysis Figures

Setelah semua hasil utama tersedia, generate ulang figure dan tabel ringkasan:

```bash
python scripts/generate_progress5_analysis.py
python scripts/generate_final_analysis.py
```

Output utama:

```text
results/tables/progress5_*.csv
results/tables/final_*.csv
results/figures/progress5_*.png
results/figures/final_*.png
```

## Catatan

- File percobaan sementara tidak disimpan di repo final agar repository tetap bersih.
- Checkpoint model tidak disimpan karena ukurannya besar.
- Hasil yang dikomit adalah tabel, prediksi, log penting, script, notebook, dan figure yang diperlukan untuk laporan.
