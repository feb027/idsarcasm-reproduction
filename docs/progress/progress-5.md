# Progress 5 — Optimasi Transformer dan Eksperimen Model Ringan Modern

**Status:** ✅ selesai.

Progress 5 menyelesaikan tahap optimasi utama proyek. Fokusnya adalah memperbaiki performa XLM-R Large melalui *threshold tuning*, menjalankan screening hyperparameter kecil pada XLM-R Base, membuat error analysis baseline vs optimized, dan menambahkan pembanding modern local LLM via LM Studio.

---

## 1. Scope yang Dijalankan

Target Progress 5:

1. **Threshold tuning XLM-R Large** ✅
2. **Small hyperparameter experiment XLM-R Base → XLM-R Large** ✅
3. **Error analysis baseline vs optimized** ✅
4. **Modern local LLM comparison: Qwen3.5-4B dan Gemma 4 E4B** ✅

Cendol/Bahasa-4B tidak dijalankan karena waktu dan ketersediaan model lokal. Scope tetap cukup karena sudah ada dua model modern ringan yang berhasil dievaluasi penuh pada Twitter.

---

## 2. Hasil Utama Optimasi Transformer

File utama:

```text
results/tables/optimization_runs.csv
results/tables/progress5_transformer_optimization_summary.csv
results/optimization/*
results/progress5_error_analysis/*
```

Ringkasan threshold tuning XLM-R Large:

| Dataset | Threshold | F1 Default | F1 Tuned | Delta F1 | Precision Tuned | Recall Tuned |
|---|---:|---:|---:|---:|---:|---:|
| Twitter | 0.85 | 0.7226 | 0.7649 | +0.0423 | 0.7219 | 0.8134 |
| Reddit | 0.26 | 0.6117 | 0.6241 | +0.0124 | 0.5596 | 0.7054 |

Interpretasi:

- Twitter mendapat kenaikan paling jelas. Threshold 0.85 membuat model lebih selektif memprediksi sarkastik, sehingga precision naik tanpa kehilangan recall terlalu banyak.
- Reddit juga naik, tetapi lebih kecil. Threshold optimal 0.26 menunjukkan karakter probabilitas model berbeda dari Twitter.
- Hasil Twitter tuned F1 0.7649 sudah mendekati target paper XLM-R Large Twitter 0.7692.
- Hasil Reddit tuned F1 0.6241 juga mendekati target paper 0.6274.

Figure:

```text
results/figures/progress5_pipeline_architecture.png
results/figures/progress5_threshold_tuning_f1.png
results/figures/progress5_xlmr_base_screening.png
results/figures/progress5_threshold_error_transitions.png
```

---

## 3. Screening Hyperparameter XLM-R Base

Ringkasan Twitter XLM-R Base:

| Konfigurasi | F1 Default | F1 Tuned | Delta F1 | Threshold |
|---|---:|---:|---:|---:|
| lr=5e-6, len=128 | 0.0000 | 0.4800 | +0.4800 | 0.25 |
| lr=2e-5, len=128 | 0.7039 | 0.7317 | +0.0278 | 0.17 |
| lr=1e-5, len=256 | 0.6953 | 0.6953 | 0.0000 | 0.50 |
| lr=1e-5, wd=0.01 | 0.7154 | 0.7115 | -0.0039 | 0.41 |
| label smoothing=0.05 | 0.7260 | 0.6877 | -0.0383 | 0.67 |

Konfigurasi paling menjanjikan dari screening adalah `lr=2e-5, max_length=128`, karena menghasilkan F1 tuned 0.7317. Namun, hasil terbaik keseluruhan tetap XLM-R Large dengan threshold tuning.

---

## 4. Error Analysis

File:

```text
results/progress5_error_analysis/twitter-xlmr-large-threshold_transition_summary.csv
results/progress5_error_analysis/twitter-xlmr-large-threshold_examples.csv
results/progress5_error_analysis/reddit-xlmr-large-threshold_transition_summary.csv
results/progress5_error_analysis/reddit-xlmr-large-threshold_examples.csv
```

Ringkasan transisi prediksi:

| Dataset | Membaik | Memburuk | Tetap Benar | Tetap Salah |
|---|---:|---:|---:|---:|
| Twitter | 22 | 3 | 449 | 64 |
| Reddit | 71 | 129 | 2153 | 471 |

Interpretasi:

- Pada Twitter, threshold tuning jelas efektif karena lebih banyak contoh yang membaik daripada memburuk.
- Pada Reddit, jumlah contoh memburuk lebih banyak, tetapi perubahan precision/recall secara agregat masih menaikkan F1 sedikit.
- Dengan kata lain, threshold tuning lebih cocok untuk karakter output XLM-R Large di Twitter.

---

## 5. Modern Local LLM via LM Studio

File utama:

```text
results/tables/modern_llm_experiments.csv
results/tables/progress5_modern_llm_summary.csv
results/modern_llm/*
```

Hasil Twitter:

| Model | Mode | Accuracy | Precision | Recall | F1 | Invalid Output |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Zero-shot | 0.6896 | 0.4078 | 0.5448 | 0.4665 | 0 |
| Qwen3.5-4B | Few-shot | 0.7416 | 0.4809 | 0.4701 | 0.4755 | 0 |
| Gemma 4 E4B | Zero-shot | 0.3755 | 0.2833 | 0.9851 | 0.4400 | 0 |
| Gemma 4 E4B | Few-shot | 0.4721 | 0.3077 | 0.8955 | 0.4580 | 0 |

Interpretasi:

- Qwen3.5-4B few-shot menjadi modern LLM lokal terbaik dengan F1 0.4755.
- Gemma 4 E4B sangat agresif memprediksi sarkastik, terlihat dari recall tinggi tetapi precision rendah.
- Modern LLM lokal lebih baik dari zero-shot BLOOMZ/mT0 Progress 4 pada Twitter, tetapi masih jauh di bawah XLM-R Large tuned.
- Masalah awal Qwen reasoning-only berhasil diatasi dengan `{%- set enable_thinking = false %}`, `/no_think`, dan parser label Indonesia.

Figure:

```text
results/figures/progress5_modern_llm_f1_comparison.png
results/figures/progress5_modern_llm_precision_recall.png
```

---

## 6. Kesimpulan Progress 5

Progress 5 sudah memenuhi target. Optimasi yang paling berhasil adalah threshold tuning XLM-R Large. Hasilnya bukan hanya menaikkan F1, tetapi juga membuat hasil reproduksi sangat dekat dengan paper untuk model terbaik.

Kesimpulan utama:

1. XLM-R Large tetap model terkuat untuk IdSarcasm.
2. Threshold tuning sederhana bisa memberi peningkatan nyata, terutama di Twitter.
3. Hyperparameter screening memberi petunjuk tambahan, tetapi belum mengalahkan XLM-R Large tuned.
4. Modern LLM lokal berguna sebagai pembanding praktis, namun belum mendekati fine-tuned transformer.
5. Untuk deteksi sarkasme bahasa Indonesia, data berlabel dan fine-tuning masih lebih penting daripada sekadar memakai model generatif baru.

Progress berikutnya adalah Progress 6: finalisasi laporan, komparasi akhir, dan kesimpulan proyek.
