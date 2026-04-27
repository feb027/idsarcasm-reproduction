# Progress 5 Run Guide — Optimasi dan Eksperimen Lanjutan

Panduan ini dipakai untuk menjalankan Progress 5 IdSarcasm. Ada dua jalur:

1. **Colab GPU:** transformer optimization, terutama XLM-R.
2. **Local PC + LM Studio:** modern LLM zero-shot/few-shot dengan model GGUF/quantized.

Untuk Windows 11 + LM Studio yang lebih rinci, baca juga:

```text
docs/progress-5-lmstudio-windows11-guide.md
```

---

## 1. Checklist Sebelum Mulai

Pastikan repo sudah terbaru:

```bash
git pull
```

Pastikan Progress 4 sudah ada:

```bash
ls results/tables/zeroshot_baselines.csv
ls results/figures/zeroshot_*.png
```

Aset Progress 5:

```bash
ls scripts/run_transformer_optimization.py
ls scripts/run_modern_llm_experiments.py
ls notebooks/04_progress5_optimization_and_modern_llm.ipynb
```

---

## 2. Jalur Colab — Transformer Optimization

### 2.1 Setup Colab

Di Colab, jalankan:

```bash
!git clone https://github.com/feb027/idsarcasm-reproduction.git
%cd idsarcasm-reproduction
!pip install -r requirements.txt
```

Kalau repo sudah ada di session Colab:

```bash
%cd idsarcasm-reproduction
!git pull
!pip install -r requirements.txt
```

### 2.2 Smoke test aman

Jalankan ini dulu untuk memastikan script tidak error:

```bash
!python scripts/run_transformer_optimization.py \
  --dataset twitter \
  --model xlmr-base \
  --run-name smoke-twitter-xlmr-base-progress5 \
  --epochs 1 \
  --batch-size 4 \
  --eval-batch-size 8 \
  --max-train-samples 24 \
  --max-eval-samples 12 \
  --max-predict-samples 12 \
  --disable-tqdm
```

Output smoke masuk ke:

```text
results/tables/optimization_smoke.csv
results/optimization/smoke-twitter-xlmr-base-progress5/
```

Kalau smoke test gagal, jangan lanjut full run. Kirim error/log ke agent.

---

## 3. Wajib 1 — Threshold Tuning XLM-R Large

Jalankan di Colab GPU:

```bash
!python scripts/run_transformer_optimization.py \
  --dataset twitter \
  --model xlmr-large \
  --run-name twitter-xlmr-large-threshold \
  --epochs 100 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --learning-rate 1e-5 \
  --lr-scheduler-type cosine \
  --weight-decay 0.03 \
  --label-smoothing-factor 0.0 \
  --max-length 128 \
  --early-stopping-threshold 0.01 \
  --seed 42 \
  --pad-to-max-length \
  --shuffle-train-dataset \
  --fp16 \
  --disable-tqdm \
  2>&1 | tee results/logs/progress-5-optimization-twitter-xlmr-large-threshold.log
```

Output penting:

```text
results/tables/optimization_runs.csv
results/optimization/twitter-xlmr-large-threshold/predictions.csv
results/optimization/twitter-xlmr-large-threshold/threshold_sweep.csv
results/optimization/twitter-xlmr-large-threshold/metrics.json
results/logs/progress-5-optimization-twitter-xlmr-large-threshold.log
```

Kirim balik ke agent setelah run ini selesai kalau ingin dicek dulu sebelum lanjut run mahal lain.

---

## 4. Wajib 2 — Small Hyperparameter Screening XLM-R Base

Jalankan satu per satu. Jangan sekaligus satu loop panjang, supaya kalau Colab putus hasil sebelumnya tetap aman.

### 4.1 Learning rate rendah

```bash
!python scripts/run_transformer_optimization.py --dataset twitter --model xlmr-base --run-name twitter-xlmr-base-lr5e-6-len128 --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 5e-6 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --disable-tqdm 2>&1 | tee results/logs/progress-5-optimization-twitter-xlmr-base-lr5e-6-len128.log
```

### 4.2 Learning rate tinggi

```bash
!python scripts/run_transformer_optimization.py --dataset twitter --model xlmr-base --run-name twitter-xlmr-base-lr2e-5-len128 --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 2e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --disable-tqdm 2>&1 | tee results/logs/progress-5-optimization-twitter-xlmr-base-lr2e-5-len128.log
```

### 4.3 Max length 256

```bash
!python scripts/run_transformer_optimization.py --dataset twitter --model xlmr-base --run-name twitter-xlmr-base-lr1e-5-len256 --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 256 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --disable-tqdm 2>&1 | tee results/logs/progress-5-optimization-twitter-xlmr-base-lr1e-5-len256.log
```

### 4.4 Weight decay 0.01

```bash
!python scripts/run_transformer_optimization.py --dataset twitter --model xlmr-base --run-name twitter-xlmr-base-lr1e-5-wd001 --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.01 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --disable-tqdm 2>&1 | tee results/logs/progress-5-optimization-twitter-xlmr-base-lr1e-5-wd001.log
```

### 4.5 Label smoothing 0.05

```bash
!python scripts/run_transformer_optimization.py --dataset twitter --model xlmr-base --run-name twitter-xlmr-base-label-smoothing005 --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.05 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --disable-tqdm 2>&1 | tee results/logs/progress-5-optimization-twitter-xlmr-base-label-smoothing005.log
```

Setelah 3–5 run XLM-R Base selesai, commit hasil dulu atau kirim tabel `optimization_runs.csv` ke agent untuk dipilihkan konfigurasi XLM-R Large final.

---

## 5. Optional Reddit Check

Kalau Twitter sudah selesai dan Colab masih cukup, jalankan Reddit XLM-R Large threshold tuning:

```bash
!python scripts/run_transformer_optimization.py --dataset reddit --model xlmr-large --run-name reddit-xlmr-large-threshold --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --disable-tqdm 2>&1 | tee results/logs/progress-5-optimization-reddit-xlmr-large-threshold.log
```

---

## 6. Cara Melihat Ringkasan Hasil Optimization

Di Colab/local:

```bash
python - <<'PY'
import csv
from pathlib import Path
p = Path('results/tables/optimization_runs.csv')
if not p.exists():
    print('Belum ada optimization_runs.csv')
else:
    rows = list(csv.DictReader(p.open(encoding='utf-8')))
    for r in rows:
        print(r['run_id'], 'default_f1=', r['test_default_f1'], 'tuned_f1=', r['test_tuned_f1'], 'delta=', r['delta_f1'], 'thr=', r['selected_threshold'])
PY
```

---

## 7. Jalur Windows 11 — LM Studio / GGUF Modern LLM

Jalur ini **Windows-first**. Karena LM Studio berjalan di Windows, jalankan script dari Windows PowerShell + Windows venv. Panduan lengkap ada di:

```text
docs/progress-5-lmstudio-windows11-guide.md
```

### 7.1 Start LM Studio server

Di LM Studio:

1. Load model GGUF.
2. Buka **Developer / Local Server**.
3. Klik **Start Server**.
4. Pastikan aktif di:

```text
http://localhost:1234/v1
```

Test dari PowerShell:

```powershell
Invoke-RestMethod http://localhost:1234/v1/models
```

Set variable PowerShell:

```powershell
$env:API_BASE = "http://localhost:1234/v1"
$env:MODEL_ID = "qwen3.5-4b"
```

Ganti `$env:MODEL_ID` sesuai `id` yang muncul dari LM Studio.

### 7.2 Setup Windows venv

```powershell
cd "G:\semester 6\idsarcasm-reproduction"
git pull
.\.venv\Scripts\Activate.ps1
```

Kalau venv belum ada:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 7.3 Smoke test

```powershell
python scripts/run_modern_llm_experiments.py `
  --dataset twitter `
  --model $env:MODEL_ID `
  --model-alias lmstudio-smoke `
  --api-base $env:API_BASE `
  --max-samples 5 `
  --print-every 1
```

Cek hasil:

```powershell
Import-Csv results	ables\modern_llm_smoke.csv | Format-Table dataset,mode,model_alias,f1,invalid_outputs,num_examples
```

Jika `invalid_outputs` tinggi, pakai strict few-shot.

### 7.4 Strict few-shot smoke — direkomendasikan

```powershell
python scripts/run_modern_llm_experiments.py `
  --dataset twitter `
  --model $env:MODEL_ID `
  --model-alias qwen3.5-4b-strict-smoke `
  --api-base $env:API_BASE `
  --few-shot `
  --shots-per-class 2 `
  --temperature 0.0 `
  --max-tokens 12 `
  --system-prompt "You are a strict binary classifier. Answer exactly one label only: sarcastic or not sarcastic. Do not explain." `
  --max-samples 10 `
  --print-every 1
```

### 7.5 Full run Twitter few-shot

```powershell
python scripts/run_modern_llm_experiments.py `
  --dataset twitter `
  --model $env:MODEL_ID `
  --model-alias qwen3.5-4b-gguf `
  --api-base $env:API_BASE `
  --few-shot `
  --shots-per-class 2 `
  --temperature 0.0 `
  --max-tokens 12 `
  --system-prompt "You are a strict binary classifier. Answer exactly one label only: sarcastic or not sarcastic. Do not explain." `
  --print-every 50
```

Untuk Gemma/Bahasa/Cendol, ganti `--model-alias` saja, misalnya:

```text
gemma-3n-e4b-gguf
bahasa-4b-gguf
cendol-gguf
```

### 7.6 Reddit modern LLM

Reddit lebih lama. Jalankan hanya untuk model/mode terbaik dari Twitter:

```powershell
python scripts/run_modern_llm_experiments.py `
  --dataset reddit `
  --model $env:MODEL_ID `
  --model-alias qwen3.5-4b-gguf `
  --api-base $env:API_BASE `
  --few-shot `
  --shots-per-class 2 `
  --temperature 0.0 `
  --max-tokens 12 `
  --system-prompt "You are a strict binary classifier. Answer exactly one label only: sarcastic or not sarcastic. Do not explain." `
  --print-every 100
```

---

## 8. Kapan Commit dan Kapan Balik ke Agent

### Setelah aset ini dibuat
Commit boleh dilakukan sekarang setelah validasi script/notebook:

```bash
git add scripts/run_transformer_optimization.py scripts/run_modern_llm_experiments.py notebooks/04_progress5_optimization_and_modern_llm.ipynb docs/progress-5.md docs/progress-5-run-guide.md docs/progress-plan.md README.md
git commit -m "feat: add Progress 5 optimization experiment assets"
git push
```

### Setelah XLM-R Large threshold selesai
Balik ke agent dengan:

```text
results/tables/optimization_runs.csv
results/optimization/twitter-xlmr-large-threshold/metrics.json
results/optimization/twitter-xlmr-large-threshold/predictions.csv
results/optimization/twitter-xlmr-large-threshold/threshold_sweep.csv
results/logs/progress-5-optimization-twitter-xlmr-large-threshold.log
```

Kalau hasilnya bagus, agent bisa bantu pilih konfigurasi berikutnya.

### Setelah XLM-R Base screening selesai
Commit:

```bash
git add results/tables/optimization_runs.csv results/optimization results/logs/progress-5-optimization-*.log
git commit -m "results: add Progress 5 transformer optimization runs"
git push
```

Lalu balik ke agent untuk:

- pilih 1–2 konfigurasi XLM-R Large final,
- buat figure awal,
- mulai error analysis.

### Setelah LM Studio modern LLM selesai
Commit:

```bash
git add results/tables/modern_llm_experiments.csv results/modern_llm
git commit -m "results: add Progress 5 modern local LLM experiments"
git push
```

Lalu balik ke agent untuk:

- komparasi modern LLM vs zero-shot Progress 4,
- figure final,
- update `docs/laporan-proyek.md`,
- review Codex final.

---

## 9. Troubleshooting

### XLM-R Large OOM di Colab
Coba fallback yang tetap dicatat sebagai fallback, bukan strict baseline:

```bash
--batch-size 16 --gradient-accumulation-steps 2 --gradient-checkpointing
```

atau:

```bash
--auto-find-batch-size
```

Catat di laporan bahwa konfigurasi fallback berbeda dari baseline paper.

### LM Studio connection refused
Cek:

1. LM Studio Local Server sudah Start.
2. URL benar: `http://localhost:1234/v1`.
3. Script dijalankan di mesin yang sama dengan LM Studio.
4. Kalau dari WSL dan LM Studio di Windows, coba pakai IP Windows host, bukan localhost.

### Output model tidak bisa diparse
Script akan menghitung `invalid_outputs`. Jika banyak invalid:

- turunkan temperature ke `0.0`,
- pastikan system prompt bilang jawab satu label saja,
- naikkan `--max-tokens 8` atau `12`,
- coba few-shot.
