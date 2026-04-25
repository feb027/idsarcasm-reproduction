# Progress 3 — Paper Baseline Complete Transformer Reproduction

**Status:** ✅ selesai  
**Target utama:** menjalankan baseline fine-tuned transformer paper pada dua dataset IdSarcasm: Twitter dan Reddit.

---

## Tujuan

Progress 3 menaikkan proyek dari baseline classical ML ke baseline transformer. Scope awal hanya dua model Twitter, tetapi setelah eksekusi Colab diperluas menjadi **paper baseline complete** untuk baseline fine-tuned transformer: **6 model × 2 dataset = 12 run**.

Progress ini belum mencakup zero-shot LLM. Zero-shot dipindahkan ke **Progress 4** karena workflow-nya berbeda: inference/prompting LLM, bukan fine-tuning transformer.

---

## Scope Progress 3

- [x] Membaca paper dan source-code original sebagai acuan.
- [x] Menyamakan setting utama dengan recipe paper: epoch 100, batch 32/64, scheduler cosine, learning rate 1e-5, weight decay 0.03, max length 128, pad-to-max-length, shuffle train, early stopping threshold 0.01, seed 42, fp16.
- [x] Menyiapkan runner fine-tuning: `scripts/run_transformer_baseline.py`.
- [x] Menyiapkan notebook Colab lengkap: `notebooks/02_transformer_baseline_colab.ipynb`.
- [x] Menyiapkan panduan eksekusi: `docs/progress-3-local-run-guide.md` dan `docs/progress-3-paper-baseline-complete-plan.md`.
- [x] Menjalankan smoke test.
- [x] Menjalankan 12 baseline transformer paper pada Twitter dan Reddit.
- [x] Menyimpan metrik, konfigurasi, dan log Colab.
- [x] Membandingkan hasil dengan angka paper dan baseline classical Progress 2.

---

## Model dan Dataset

| Model | HF model | Reddit F1 paper | Twitter F1 paper |
|---|---|---:|---:|
| IndoBERT Base (IndoNLU) | `indobenchmark/indobert-base-p1` | 0.6100 | 0.7273 |
| IndoBERT Large (IndoNLU) | `indobenchmark/indobert-large-p1` | 0.6184 | 0.7160 |
| IndoBERT Base (IndoLEM) | `indolem/indobert-base-uncased` | 0.5671 | 0.6462 |
| mBERT | `bert-base-multilingual-cased` | 0.5338 | 0.6467 |
| XLM-R Base | `xlm-roberta-base` | 0.5690 | 0.7386 |
| XLM-R Large | `xlm-roberta-large` | 0.6274 | 0.7692 |

---

## Kesesuaian dengan Source Code Paper

Runner lokal dibuat agar setting default mengikuti recipe asli paper di `source-code/original-id-sarcasm/recipes/{twitter,reddit}/baseline/`.

| Komponen | Setting Progress 3 | Recipe Paper |
|---|---:|---:|
| Dataset | Twitter + Reddit | Twitter + Reddit |
| Text column | `tweet` / `text` | `tweet` / `text` |
| Label column | `label` | `label` |
| Max sequence length | 128 | 128 |
| Train batch size | 32 | 32 |
| Eval batch size | 64 | 64 |
| Learning rate | 1e-5 | 1e-5 |
| LR scheduler | cosine | cosine |
| Weight decay | 0.03 | 0.03 |
| Label smoothing | 0.0 | 0.0 |
| Epoch maksimum | 100 | 100 |
| Shuffle train split | yes | yes |
| Early stopping | patience 3, threshold 0.01 | patience 3, threshold 0.01 |
| Padding | pad to max length | pad to max length default |
| Metric best model | F1 | F1 |
| Seed | 42 | 42 |
| FP16 | yes jika CUDA tersedia | yes |

Perbedaan yang sengaja dipertahankan:

- Tidak memakai `--push_to_hub`, karena project UAS tidak perlu upload model ke HuggingFace.
- Output checkpoint disimpan lokal di `models/transformer/` dan tidak di-commit.
- Output metrik dirapikan ke `results/tables/transformer_baselines.csv` dan `results/transformer/...`.
- Log Colab disimpan di `results/logs/` sebagai bukti eksekusi.
- Ada compatibility patch untuk Transformers versi baru (`overwrite_output_dir`, `tokenizer`/`processing_class`, dan progress bar), karena Colab memakai library yang lebih baru dari source-code paper.

---

## Hasil Progress 3

| Model | Reddit F1 | Paper Reddit | Gap | Twitter F1 | Paper Twitter | Gap |
|---|---:|---:|---:|---:|---:|---:|
| IndoBERT Base (IndoNLU) | 0.5839 | 0.6100 | -0.0261 | 0.6812 | 0.7273 | -0.0461 |
| IndoBERT Large (IndoNLU) | 0.5825 | 0.6184 | -0.0359 | 0.6831 | 0.7160 | -0.0329 |
| IndoBERT Base (IndoLEM) | 0.5457 | 0.5671 | -0.0214 | 0.6835 | 0.6462 | +0.0373 |
| mBERT | 0.5413 | 0.5338 | +0.0075 | 0.7092 | 0.6467 | +0.0625 |
| XLM-R Base | 0.5819 | 0.5690 | +0.0129 | 0.7000 | 0.7386 | -0.0386 |
| XLM-R Large | 0.6117 | 0.6274 | -0.0157 | 0.7226 | 0.7692 | -0.0466 |

Model terbaik hasil reproduksi:

- **Reddit:** XLM-R Large, F1 = **0.6117**.
- **Twitter:** XLM-R Large, F1 = **0.7226**.

Hasil terbaik masih di bawah skor terbaik paper, tetapi urutan umum tetap masuk akal: XLM-R Large menjadi model terkuat di kedua dataset.

---

## Output yang Tersimpan

```text
results/tables/transformer_baselines.csv
results/tables/transformer_smoke.csv
results/transformer/*/metrics.json
results/transformer/*/result_row.json
results/logs/progress-3-*.log
```

Folder `models/` tidak di-commit karena berisi checkpoint besar.

---

## Catatan Penting

1. Beberapa model IndoBERT/IndoLEM menampilkan warning `LayerNorm.gamma/beta` vs `LayerNorm.weight/bias`. Training tetap selesai, tetapi warning ini dicatat sebagai isu kompatibilitas checkpoint lama dengan Transformers versi baru.
2. Warning `classifier.weight/bias MISSING` normal untuk fine-tuning sequence classification karena head klasifikasi dibuat baru.
3. XLM-R Large berhasil dijalankan di Colab dengan batch paper-faithful, sehingga Progress 3 dapat dianggap **paper baseline complete** untuk fine-tuned transformer.

---

## Gate Kelulusan Progress 3

Progress 3 dianggap selesai karena:

1. semua 12 baseline transformer fine-tuning paper sudah dijalankan,
2. semua hasil full run memiliki `sample_limited=false`,
3. metrik tersimpan di CSV/JSON,
4. log Colab tersimpan sebagai bukti,
5. hasil sudah dibandingkan dengan angka paper.

Langkah berikutnya adalah **Progress 4: zero-shot LLM baseline**, termasuk kemungkinan inference lokal via LM Studio.
