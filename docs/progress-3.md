# Progress 3 — Reproduksi Transformer Baseline

**Status:** disiapkan untuk eksekusi di Google Colab GPU  
**Target utama:** IndoBERT Base pada dataset Twitter IdSarcasm

---

## Tujuan

Progress 3 menaikkan proyek dari baseline classical ML ke baseline transformer. Pada progress sebelumnya, Logistic Regression dan SVM sudah berhasil mendekati hasil paper untuk dataset Twitter dan Reddit. Progress ini mulai menguji apakah model berbasis transformer dapat memberi performa yang lebih kuat pada dataset yang sama.

Karena resource proyek tetap perlu realistis, target awal tidak langsung memakai XLM-R Large seperti model terbaik paper. Target yang dipilih adalah **IndoBERT Base** (`indobenchmark/indobert-base-p1`) pada dataset **Twitter Indonesia Sarcastic**. Model ini lebih ringan, relevan untuk teks bahasa Indonesia, dan cocok dijalankan di Google Colab GPU.

---

## Scope Progress 3

- [x] Menentukan model transformer baseline yang realistis.
- [x] Menyiapkan script fine-tuning transformer: `scripts/run_transformer_baseline.py`.
- [x] Menyiapkan panduan eksekusi Colab/lokal: `docs/progress-3-local-run-guide.md`.
- [ ] Menjalankan smoke test di Colab.
- [ ] Menjalankan full baseline Twitter IndoBERT Base.
- [ ] Menyimpan hasil metrik di `results/tables/transformer_baselines.csv`.
- [ ] Membandingkan hasil transformer dengan baseline classical ML terbaik dari Progress 2.
- [ ] Mengisi laporan utama pada section 2.2.2, 3.2.2, dan 3.3.2.

---

## Konfigurasi Baseline Utama

| Komponen | Nilai |
|---|---|
| Dataset | Twitter Indonesia Sarcastic |
| HuggingFace ID | `w11wo/twitter_indonesia_sarcastic` |
| Model | IndoBERT Base |
| HuggingFace model | `indobenchmark/indobert-base-p1` |
| Max length | 128 |
| Learning rate | 1e-5 |
| Weight decay | 0.03 |
| Epoch | 5 |
| Batch size | 16 |
| Eval batch size | 32 |
| Seed | 42 |
| Metric utama | F1-score |
| Runtime disarankan | Google Colab GPU |

Konfigurasi ini diturunkan dari recipe transformer pada snapshot repo asli paper, tetapi dibuat lebih aman untuk Progress 3. Recipe upstream memakai 100 epoch dengan early stopping; untuk proyek ini, 5 epoch dipilih sebagai baseline awal agar lebih realistis di Colab dan tidak terlalu mahal.

---

## Command Utama

```bash
python scripts/run_transformer_baseline.py \
  --dataset twitter \
  --model indobert-base \
  --epochs 5 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --learning-rate 1e-5 \
  --weight-decay 0.03 \
  --max-length 128 \
  --fp16 \
  --output-dir results/transformer/twitter-indobert-base \
  --model-output-dir models/transformer/twitter-indobert-base
```

---

## Output yang Diharapkan

```text
results/tables/transformer_baselines.csv
results/transformer/twitter-indobert-base/metrics.json
results/transformer/twitter-indobert-base/result_row.json
```

Untuk smoke test dengan `--max-*-samples`, script otomatis memakai `results/tables/transformer_smoke.csv` jika `--table-path` tidak diubah. Ini sengaja dipisahkan agar metrik sampel kecil tidak tercampur dengan tabel baseline final.

File hasil ini harus disimpan di repo karena akan dipakai untuk laporan dan analisis komparatif. Folder model checkpoint di `models/` tidak perlu di-commit karena ukurannya besar.

---

## Pembanding dari Progress 2

| Dataset | Model Classical Terbaik | Representasi | F1 |
|---|---|---|---:|
| Twitter | Logistic Regression | TF-IDF | 0,7143 |
| Twitter | SVM | BoW | 0,6850 |
| Reddit | Logistic Regression | TF-IDF | 0,4959 |
| Reddit | SVM | TF-IDF | 0,4467 |

Untuk Progress 3, hasil IndoBERT Base perlu dibandingkan terutama dengan **Twitter TF-IDF Logistic Regression = 0,7143**. Jika transformer lebih tinggi, maka ada bukti awal bahwa model kontekstual memberi peningkatan. Jika lebih rendah, hasilnya tetap berguna karena Progress 4 dapat difokuskan pada tuning.

---

## Catatan untuk Laporan

Setelah hasil training tersedia, laporan utama `docs/laporan-proyek.md` harus diisi pada bagian:

1. `2.2.2 Model Transformer` — jelaskan IndoBERT Base, tokenisasi subword, fine-tuning, dan alasan pemilihan.
2. `3.2.2 Tahapan Eksperimen Transformer` — jelaskan alur load dataset, tokenisasi, training, validasi, dan evaluasi test.
3. `3.3.2 Hasil Model Transformer` — masukkan tabel hasil, bandingkan dengan baseline classical ML, dan jelaskan gap.

Jangan mengisi angka hasil sebelum training selesai.
