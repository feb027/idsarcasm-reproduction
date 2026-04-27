# Progress 5 — Optimasi Transformer dan Eksperimen Model Ringan Modern

**Status:** aset eksekusi disiapkan, belum menjalankan training/inference full.

**Tujuan utama:** melanjutkan proyek dari reproduksi baseline ke tahap optimasi yang terukur. Progress ini tetap menempatkan fine-tuned transformer sebagai jalur utama, lalu menambahkan eksperimen LLM ringan modern sebagai pembanding praktis/local.

---

## 1. Ringkasan Keputusan Scope

Progress 5 memakai empat target kerja:

1. **Wajib:** threshold tuning XLM-R Large.
2. **Wajib:** small hyperparameter experiment XLM-R Base → XLM-R Large.
3. **Wajib:** error analysis dari baseline vs optimized.
4. **Tambahan:** Gemma 3n E4B + Qwen3.5-4B + Cendol/Bahasa-4B untuk zero-shot/few-shot.

Pembagian ini penting karena model baru seperti Gemma/Qwen/Cendol tidak otomatis disebut optimasi transformer. Model-model tersebut lebih tepat ditulis sebagai **eksperimen model ringan modern**, sedangkan optimasi utama tetap dilakukan pada XLM-R yang sudah menjadi baseline terbaik Progress 3.

---

## 2. Dasar dari Progress Sebelumnya

Sampai Progress 4, proyek sudah punya tiga kelompok baseline:

| Kelompok | Hasil utama |
|---|---|
| Classical ML | Twitter sangat kuat; best Twitter BoW LR F1 = 0,7206, best Reddit TF-IDF LR F1 = 0,4959 |
| Fine-tuned transformer | Best: XLM-R Large, Twitter F1 = 0,7226, Reddit F1 = 0,6117 |
| Zero-shot LLM paper | Twitter 9/9 selesai, Reddit 5/9 selesai; F1 sekitar 0,39–0,40 |

Dari sini, target Progress 5 bukan mengulang baseline, tetapi menjawab:

- apakah XLM-R Large bisa diperbaiki setelah baseline paper,
- apakah trade-off precision/recall bisa dibuat lebih baik,
- contoh kesalahan apa yang masih terjadi,
- apakah LLM ringan modern yang bisa jalan lokal/Colab bisa mendekati zero-shot paper atau fine-tuned transformer.

---

## 3. Riset Singkat dan Alasan Metode

### 3.1 Threshold tuning

Dari hasil Progress 3, XLM-R Large Twitter memiliki recall tinggi tetapi precision lebih rendah. Ini menunjukkan model cukup agresif memprediksi kelas sarkastik. Praktik umum untuk binary classification adalah mengambil probabilitas/logit model, lalu mengubah threshold dari default 0,5 menjadi nilai lain. Namun threshold harus dipilih dari validation set, bukan test set.

Exa menemukan diskusi HuggingFace tentang pengubahan threshold klasifikasi: logits model sequence classification dapat dikonversi menjadi probabilitas, lalu threshold bisa diubah untuk mengatur prediksi kelas positif. Dokumentasi HuggingFace `Trainer.predict` juga menjelaskan bahwa output prediksi menyediakan `predictions`/logits dan metrik, sehingga cocok untuk menyimpan probabilitas dan melakukan analisis threshold.

### 3.2 Model modern ringan

Hasil pencarian Exa awal:

| Model | Catatan ringkas | Posisi di Progress 5 |
|---|---|---|
| `google/gemma-3n-E4B` | HuggingFace menyebut Gemma 3n E4B mendukung Transformers, effective 4B walaupun raw sekitar 8B, dirancang low-resource dan multilingual. | Kandidat utama local/Colab. |
| `Qwen/Qwen3.5-4B` | HuggingFace menyebut Qwen3.5-4B kompatibel dengan Transformers/vLLM/SGLang/KTransformers, rilis 2026. | Kandidat ringan terbaru. |
| `indonlp/cendol` | Koleksi Indonesian LLM 300M–13B berbasis mT5 dan LLaMA2. | Pembanding Indonesia-specific. |
| `Bahasalab/Bahasa-4b` | Continued training dari Qwen-4B pada data Indonesia. | Pembanding Indonesia 4B. |

Untuk LM Studio, Exa menemukan dokumentasi resmi bahwa LM Studio menyediakan endpoint OpenAI-compatible `POST /v1/chat/completions`, dengan contoh Python `OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")`. Jadi model GGUF yang dimuat di LM Studio bisa diuji melalui script OpenAI-compatible.

---

## 4. Desain Eksperimen Wajib

### 4.1 Threshold tuning XLM-R Large

Langkah:

1. Fine-tune ulang XLM-R Large dengan konfigurasi baseline Progress 3.
2. Simpan probabilitas kelas sarkastik untuk validation dan test.
3. Cari threshold terbaik di validation set berdasarkan F1.
4. Terapkan threshold tersebut ke test set.
5. Bandingkan:
   - default argmax / threshold 0,5,
   - tuned threshold dari validation.

Output:

```text
results/optimization/<run-id>/predictions.csv
results/optimization/<run-id>/threshold_sweep.csv
results/optimization/<run-id>/metrics.json
results/optimization/<run-id>/result_row.json
results/tables/optimization_runs.csv
```

### 4.2 Small hyperparameter experiment XLM-R Base → XLM-R Large

Agar biaya Colab tidak terlalu besar, screening dilakukan di XLM-R Base dulu. Setelah itu, 1–2 konfigurasi terbaik baru dibawa ke XLM-R Large.

Konfigurasi screening yang disarankan:

| Run | Model | Dataset | Learning rate | Max length | Weight decay | Label smoothing |
|---|---|---|---:|---:|---:|---:|
| baseline-threshold | XLM-R Large | Twitter | 1e-5 | 128 | 0,03 | 0,00 |
| lr-low | XLM-R Base | Twitter | 5e-6 | 128 | 0,03 | 0,00 |
| lr-high | XLM-R Base | Twitter | 2e-5 | 128 | 0,03 | 0,00 |
| length-256 | XLM-R Base | Twitter | 1e-5 | 256 | 0,03 | 0,00 |
| wd-001 | XLM-R Base | Twitter | 1e-5 | 128 | 0,01 | 0,00 |
| smoothing | XLM-R Base | Twitter | 1e-5 | 128 | 0,03 | 0,05 |
| reddit-check | XLM-R Large | Reddit | 1e-5 | 128 | 0,03 | 0,00 |

Setelah screening, pilih 1–2 konfigurasi dengan F1 validation/test terbaik dan trade-off precision/recall paling masuk akal. Jalankan ulang pada XLM-R Large bila resource masih cukup.

### 4.3 Error analysis baseline vs optimized

Error analysis dilakukan setelah minimal satu optimized run selesai. Ambil `predictions.csv` lalu bandingkan:

- false positive baseline yang menjadi benar setelah threshold tuning,
- false negative baseline yang menjadi benar setelah threshold tuning,
- contoh yang tetap salah pada baseline dan optimized,
- perubahan precision/recall.

Pola yang dicari:

- teks terlalu pendek,
- sarkasme implisit tanpa kata kunci,
- humor/slang Indonesia,
- ekspresi positif yang sebenarnya menyindir,
- konteks sosial/politik yang tidak muncul lengkap di teks.

---

## 5. Eksperimen Tambahan: Modern LLM Zero-shot/Few-shot

Eksperimen ini memakai script:

```text
scripts/run_modern_llm_experiments.py
```

Mode yang disiapkan:

1. **Zero-shot:** model langsung diberi teks dan diminta menjawab `sarcastic` atau `not sarcastic`.
2. **Few-shot:** prompt diberi contoh dari train split, default 2 contoh per kelas.

Target model:

- Gemma 3n E4B / GGUF di LM Studio,
- Qwen3.5-4B / GGUF di LM Studio,
- Cendol atau Bahasa-4B sebagai pembanding Indonesia-specific.

Output:

```text
results/modern_llm/<run-id>/predictions.csv
results/modern_llm/<run-id>/metrics.json
results/modern_llm/<run-id>/result_row.json
results/tables/modern_llm_experiments.csv
```

Hasil ini nanti dibandingkan dengan zero-shot Progress 4, bukan langsung dianggap optimasi transformer.

---

## 6. File Aset Progress 5

Aset yang disiapkan:

```text
scripts/run_transformer_optimization.py
scripts/run_modern_llm_experiments.py
notebooks/04_progress5_optimization_and_modern_llm.ipynb
docs/progress-5.md
docs/progress-5-run-guide.md
```

File hasil yang akan muncul setelah user menjalankan eksperimen:

```text
results/tables/optimization_runs.csv
results/tables/modern_llm_experiments.csv
results/optimization/*
results/modern_llm/*
results/logs/progress-5-*.log
```

---

## 7. Kapan Harus Commit

### Commit A — setelah aset Progress 5 siap
Commit ini boleh dilakukan setelah script, notebook, dan dokumentasi lolos validasi ringan.

Isi commit:

```text
scripts/run_transformer_optimization.py
scripts/run_modern_llm_experiments.py
notebooks/04_progress5_optimization_and_modern_llm.ipynb
docs/progress-5.md
docs/progress-5-run-guide.md
README.md atau docs/progress-plan.md jika ikut diupdate
```

Contoh pesan commit:

```bash
git add scripts/run_transformer_optimization.py scripts/run_modern_llm_experiments.py notebooks/04_progress5_optimization_and_modern_llm.ipynb docs/progress-5.md docs/progress-5-run-guide.md docs/progress-plan.md README.md
git commit -m "feat: add Progress 5 optimization experiment assets"
git push
```

### Commit B — setelah run transformer selesai
Setelah menjalankan threshold tuning dan hyperparameter screening, commit hasil:

```text
results/tables/optimization_runs.csv
results/optimization/*
results/logs/progress-5-optimization-*.log
```

Contoh pesan:

```bash
git add results/tables/optimization_runs.csv results/optimization results/logs/progress-5-optimization-*.log
git commit -m "results: add Progress 5 transformer optimization runs"
git push
```

Setelah Commit B, kembalikan ke agent untuk dibuatkan analisis, figure, dan narasi laporan.

### Commit C — setelah run LM Studio modern LLM selesai
Commit hasil modern LLM:

```text
results/tables/modern_llm_experiments.csv
results/modern_llm/*
```

Contoh pesan:

```bash
git add results/tables/modern_llm_experiments.csv results/modern_llm
git commit -m "results: add Progress 5 modern local LLM experiments"
git push
```

Setelah Commit C, kembalikan lagi ke agent. Agent bisa lanjut membuat grafik, error analysis final, dan update `docs/laporan-proyek.md`.

---

## 8. Kriteria Selesai Progress 5

Progress 5 dianggap cukup jika minimal punya:

1. satu run XLM-R Large dengan threshold tuning,
2. minimal 3 run screening XLM-R Base,
3. satu tabel before/after baseline vs optimized,
4. minimal 10 contoh error analysis dari `predictions.csv`,
5. minimal satu eksperimen modern LLM zero-shot atau few-shot,
6. log dan hasil tersimpan di repo.

Jika waktu terbatas, prioritasnya:

1. XLM-R Large threshold tuning,
2. XLM-R Base screening,
3. error analysis,
4. baru Gemma/Qwen/Cendol.
