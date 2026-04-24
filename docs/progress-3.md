# Progress 3 — Reproduksi Transformer Baseline

**Status:** siap dieksekusi di Google Colab GPU  
**Target utama:** dua baseline transformer paper-faithful pada dataset Twitter IdSarcasm

---

## Tujuan

Progress 3 menaikkan proyek dari baseline classical ML ke baseline transformer. Pada Progress 2, Logistic Regression dan SVM sudah berhasil mendekati hasil paper untuk dataset Twitter dan Reddit. Progress 3 sekarang difokuskan untuk menjalankan model transformer dengan setting yang lebih dekat ke source code asli paper.

Dua model yang dijadikan baseline resmi:

1. **IndoBERT Base** (`indobenchmark/indobert-base-p1`)
2. **XLM-R Base** (`xlm-roberta-base`)

Keduanya dijalankan pada dataset **Twitter Indonesia Sarcastic**. XLM-R Large belum dijadikan target Progress 3 karena lebih berat dan lebih cocok dipertimbangkan sebagai batas atas hasil paper, bukan baseline awal yang realistis.

---

## Scope Progress 3

- [x] Menentukan dua model transformer baseline yang realistis: IndoBERT Base dan XLM-R Base.
- [x] Membandingkan setting dengan recipe asli paper di `source-code/original-id-sarcasm/recipes/twitter/baseline/`.
- [x] Menyiapkan script fine-tuning transformer: `scripts/run_transformer_baseline.py`.
- [x] Menyiapkan notebook Colab: `notebooks/02_transformer_baseline_colab.ipynb`.
- [x] Menyiapkan panduan eksekusi Colab/lokal: `docs/progress-3-local-run-guide.md`.
- [ ] Menjalankan smoke test di Colab.
- [ ] Menjalankan full baseline Twitter IndoBERT Base.
- [ ] Menjalankan full baseline Twitter XLM-R Base.
- [ ] Menyimpan hasil metrik di `results/tables/transformer_baselines.csv`.
- [ ] Membandingkan hasil transformer dengan baseline classical ML terbaik dari Progress 2.
- [ ] Mengisi laporan utama pada section 2.2.2, 3.2.2, dan 3.3.2.

---

## Kesesuaian dengan Source Code Paper

Runner lokal dibuat agar setting default mengikuti recipe asli paper:

| Komponen | Setting Progress 3 | Recipe Paper |
|---|---:|---:|
| Dataset | Twitter Indonesia Sarcastic | Twitter Indonesia Sarcastic |
| Text column | `tweet` | `tweet` |
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
| Early stopping | patience 3, threshold 0.01 | patience 3, threshold 0.01 di script upstream |
| Padding | pad to max length | pad to max length default |
| Metric best model | F1 | F1 |
| Seed | 42 | 42 |
| FP16 | yes jika CUDA tersedia | yes |

Perbedaan yang sengaja dipertahankan:

- Tidak memakai `--push_to_hub`, karena project UAS tidak perlu upload model ke HuggingFace.
- Output checkpoint disimpan lokal di `models/transformer/` dan tidak di-commit.
- Output metrik dirapikan ke `results/tables/transformer_baselines.csv` dan `results/transformer/...` supaya mudah masuk laporan.
- Smoke test dipisah ke `results/tables/transformer_smoke.csv`, agar tidak mencampur hasil sampel kecil dengan hasil final.

Dengan kata lain, konfigurasi training utamanya dibuat **paper-faithful**, tetapi output dan workflow disesuaikan untuk kebutuhan repo tugas.

---

## Konfigurasi Baseline Resmi

| Komponen | IndoBERT Base | XLM-R Base |
|---|---|---|
| Dataset | Twitter | Twitter |
| HF dataset | `w11wo/twitter_indonesia_sarcastic` | `w11wo/twitter_indonesia_sarcastic` |
| HF model | `indobenchmark/indobert-base-p1` | `xlm-roberta-base` |
| Max length | 128 | 128 |
| Learning rate | 1e-5 | 1e-5 |
| Scheduler | cosine | cosine |
| Weight decay | 0.03 | 0.03 |
| Epoch maksimum | 100 | 100 |
| Batch size | 32 | 32 |
| Eval batch size | 64 | 64 |
| Seed | 42 | 42 |
| Runtime disarankan | Colab GPU | Colab GPU |

---

## Command Utama

### IndoBERT Base

```bash
python scripts/run_transformer_baseline.py \
  --dataset twitter \
  --model indobert-base \
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
  --output-dir results/transformer/twitter-indobert-base \
  --model-output-dir models/transformer/twitter-indobert-base
```

### XLM-R Base

```bash
python scripts/run_transformer_baseline.py \
  --dataset twitter \
  --model xlmr-base \
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
  --output-dir results/transformer/twitter-xlmr-base \
  --model-output-dir models/transformer/twitter-xlmr-base
```

---

## Output yang Diharapkan

```text
results/tables/transformer_baselines.csv
results/transformer/twitter-indobert-base/metrics.json
results/transformer/twitter-indobert-base/result_row.json
results/transformer/twitter-xlmr-base/metrics.json
results/transformer/twitter-xlmr-base/result_row.json
```

File hasil ini harus disimpan di repo karena akan dipakai untuk laporan dan analisis komparatif. Folder checkpoint di `models/` tidak perlu di-commit karena ukurannya besar.

---

## Pembanding dari Progress 2

| Dataset | Model Classical Terbaik | Representasi | F1 |
|---|---|---|---:|
| Twitter | Logistic Regression | TF-IDF | 0,7143 |
| Twitter | SVM | BoW | 0,6850 |
| Twitter | SVM | TF-IDF | 0,6783 |
| Reddit | Logistic Regression | TF-IDF | 0,4959 |

Untuk Progress 3, hasil IndoBERT Base dan XLM-R Base perlu dibandingkan terutama dengan **Twitter TF-IDF Logistic Regression = 0,7143**. Jika transformer lebih tinggi, ada bukti awal bahwa model kontekstual memberi peningkatan. Jika lebih rendah, hasilnya tetap berguna karena Progress 4 dapat difokuskan pada tuning.

---

## Catatan untuk Laporan

Setelah hasil training tersedia, laporan utama `docs/laporan-proyek.md` harus diisi pada bagian:

1. `2.2.2 Model Transformer` — jelaskan IndoBERT Base, XLM-R Base, tokenisasi subword, fine-tuning, dan alasan pemilihan model base.
2. `3.2.2 Tahapan Eksperimen Transformer` — jelaskan alur load dataset, tokenisasi, shuffle train, training, early stopping, validasi, dan evaluasi test.
3. `3.3.2 Hasil Model Transformer` — masukkan tabel hasil dua model, bandingkan dengan baseline classical ML, dan jelaskan gap.

Jangan mengisi angka hasil sebelum training selesai.
