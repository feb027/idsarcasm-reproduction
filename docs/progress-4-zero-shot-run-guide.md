# Progress 4 Zero-shot Run Guide

Panduan ini untuk menjalankan baseline zero-shot LLM IdSarcasm. Runner utama:

```text
scripts/run_zeroshot_baseline.py
```

Notebook Colab:

```text
notebooks/03_zeroshot_baseline_colab_or_lmstudio.ipynb
```

---

## 1. Rekomendasi Runtime

| Mode | Bisa? | Catatan |
|---|---|---|
| Colab GPU | ✅ paling disarankan | Cocok untuk `mt0-small` dan `bloomz-560m`; paper-like karena memakai HuggingFace logprobs |
| PC lokal CPU | ⚠️ bisa tapi lambat | Aman untuk smoke test, tidak realistis untuk full Reddit |
| PC lokal RX 6600 / WSL | ⚠️ tidak semudah CUDA | PyTorch ROCm di Windows/WSL tidak semulus NVIDIA CUDA |
| LM Studio lokal | ✅ untuk eksperimen tambahan | Cocok untuk model GGUF/quantized; bukan reproduksi exact jika model beda dari BLOOMZ/mT0 |

Jawaban praktis: **jalankan full Progress 4 di Colab**, lalu pakai lokal hanya untuk smoke test atau LM Studio tambahan.

---

## 2. Setup Colab

Di Colab, pakai runtime GPU.

```bash
!git clone https://github.com/feb027/idsarcasm-reproduction.git
%cd idsarcasm-reproduction
!pip install -q -r requirements.txt
```

Kalau Colab belum punya `sentencepiece`, requirements sudah menyiapkannya. Jika masih error tokenizer, jalankan:

```bash
!pip install -q sentencepiece protobuf
```

---

## 3. Smoke Test Wajib

Jalankan ini dulu sebelum full run:

```bash
!python scripts/run_zeroshot_baseline.py --dataset twitter --model mt0-small --backend hf-logprobs --max-samples 8 --dtype float16 --device-map auto --disable-tqdm --write-log
```

Cek output:

```bash
!ls -R results/zeroshot | head -80
!cat results/tables/zeroshot_smoke.csv
!ls results/logs/progress-4-zeroshot-*.log
```

Jika smoke test berhasil, lanjut full run.

---

## 4. Full Run Minimal untuk Progress 4

Minimal cukup satu model pada dua dataset. Saya sarankan mulai dari `mt0-small`.

```bash
!python scripts/run_zeroshot_baseline.py --dataset twitter --model mt0-small --backend hf-logprobs --dtype float16 --device-map auto --disable-tqdm --write-log
!python scripts/run_zeroshot_baseline.py --dataset reddit --model mt0-small --backend hf-logprobs --dtype float16 --device-map auto --disable-tqdm --write-log
```

Output utama:

```text
results/tables/zeroshot_baselines.csv
results/zeroshot/twitter-hf-logprobs-mt0-small/metrics.json
results/zeroshot/twitter-hf-logprobs-mt0-small/predictions.csv
results/zeroshot/reddit-hf-logprobs-mt0-small/metrics.json
results/zeroshot/reddit-hf-logprobs-mt0-small/predictions.csv
results/logs/progress-4-zeroshot-twitter-hf-logprobs-mt0-small-full.log
results/logs/progress-4-zeroshot-reddit-hf-logprobs-mt0-small-full.log
```

---

## 5. Full Run Tambahan jika Waktu Cukup

Tambahkan `bloomz-560m` agar ada dua keluarga model paper: mT0 dan BLOOMZ.

```bash
!python scripts/run_zeroshot_baseline.py --dataset twitter --model bloomz-560m --backend hf-logprobs --dtype float16 --device-map auto --disable-tqdm --write-log
!python scripts/run_zeroshot_baseline.py --dataset reddit --model bloomz-560m --backend hf-logprobs --dtype float16 --device-map auto --disable-tqdm --write-log
```

Kalau masih ada waktu dan VRAM, baru coba model lebih besar seperti `mt0-base` atau `bloomz-1b1`. Jangan langsung ke model besar sebelum smoke test.

---

## 6. LM Studio Lokal

Mode ini berguna kalau ingin baseline tambahan dari model quantized lokal.

Langkah:

1. Buka LM Studio.
2. Download/load model GGUF quantized.
3. Start local server.
4. Pastikan endpoint aktif di:

```text
http://localhost:1234/v1
```

Smoke test:

```bash
python scripts/run_zeroshot_baseline.py --dataset twitter --model local-model --backend openai-compatible --api-base http://localhost:1234/v1 --max-samples 20 --disable-tqdm --write-log
```

Full run:

```bash
python scripts/run_zeroshot_baseline.py --dataset twitter --model local-model --backend openai-compatible --api-base http://localhost:1234/v1 --disable-tqdm --write-log
python scripts/run_zeroshot_baseline.py --dataset reddit --model local-model --backend openai-compatible --api-base http://localhost:1234/v1 --disable-tqdm --write-log
```

Catatan laporan: tulis hasil ini sebagai **zero-shot local LLM baseline**, bukan reproduksi exact paper.

---

## 7. Cara Melihat Runtime

Setiap run menulis runtime ke:

1. CSV utama:

```bash
cat results/tables/zeroshot_baselines.csv
```

Kolom penting:

```text
runtime_seconds
avg_latency_seconds
run_started_at
run_ended_at
num_examples
prompt_count
```

2. Metrics JSON:

```bash
cat results/zeroshot/twitter-hf-logprobs-mt0-small/metrics.json
```

3. Log file:

```bash
cat results/logs/progress-4-zeroshot-twitter-hf-logprobs-mt0-small-full.log
```

---

## 8. Setelah Run Selesai

Commit file berikut:

```bash
git add results/tables/zeroshot_baselines.csv results/tables/zeroshot_smoke.csv

git add results/zeroshot/ results/logs/progress-4-zeroshot-*.log

git commit -m "results: add Progress 4 zero-shot baseline runs"
git push
```

Sebelum commit, scan log agar tidak ada token/API key:

```bash
grep -RniE "hf_[A-Za-z0-9]|api[_-]?key|password|secret|Bearer" results/logs/progress-4-zeroshot-*.log || true
```

String seperti `HF_TOKEN` sebagai warning biasa masih aman. Yang tidak boleh adalah token asli.

---

## 9. Troubleshooting

### CUDA out of memory

Coba model lebih kecil:

```bash
--model mt0-small
```

Atau pakai CPU untuk smoke test saja:

```bash
--dtype float32 --device-map none --max-samples 4
```

### Tokenizer error / sentencepiece missing

```bash
pip install sentencepiece protobuf
```

### Full Reddit terlalu lama

Jalankan Twitter full dulu. Untuk Reddit, boleh mulai subset yang lebih besar:

```bash
--max-samples 250
```

Tapi untuk laporan Progress 4 final, minimal satu full Reddit tetap lebih baik jika Colab memungkinkan.

### LM Studio output banyak invalid

Gunakan model instruction-tuned yang lebih patuh, tetap temperature 0, dan cek kolom `invalid_outputs` pada CSV/JSON. Jika invalid terlalu banyak, hasil perlu dibahas sebagai keterbatasan backend generatif.
