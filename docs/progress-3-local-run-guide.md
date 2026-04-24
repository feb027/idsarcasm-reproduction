# Progress 3 Local/Colab Run Guide — Transformer Baseline

Progress 3 bertujuan menjalankan **minimal satu baseline transformer** pada benchmark IdSarcasm, lalu membandingkannya dengan baseline classical ML dari Progress 2.

Target paling aman: **IndoBERT Base (`indobenchmark/indobert-base-p1`) pada dataset Twitter**.

Alasan:
- dataset Twitter kecil (2.684 data), jadi lebih realistis untuk Google Colab;
- IndoBERT Base lebih ringan daripada XLM-R Large;
- tetap relevan dengan judul proyek karena sudah masuk keluarga transformer.

---

## 1. Environment yang disarankan

Gunakan **Google Colab GPU**.

Di Colab:
1. buka `Runtime > Change runtime type`,
2. pilih `T4 GPU` atau GPU lain yang tersedia,
3. jalankan setup di bawah.

WSL2/lokal bisa dipakai untuk membaca dan mengedit repo, tetapi training transformer lebih aman di Colab karena GPU NVIDIA lebih siap daripada RX 6600 di Windows/WSL2.

---

## 2. Clone repo

```bash
git clone https://github.com/feb027/idsarcasm-reproduction.git
cd idsarcasm-reproduction
```

Kalau sudah punya folder repo di Colab/Drive:

```bash
cd idsarcasm-reproduction
git pull
```

---

## 3. Install dependency

```bash
pip install -r requirements.txt
pip install accelerate
```

Kalau Colab meminta restart runtime setelah install, restart lalu masuk lagi ke folder repo.

---

## 4. Download/cache dataset

Script transformer bisa langsung ambil dataset dari HuggingFace. Namun supaya konsisten dengan Progress 2, cache CSV dulu:

```bash
python scripts/download_data.py
```

Output yang diharapkan:

```text
data/raw/twitter_train.csv
data/raw/twitter_validation.csv
data/raw/twitter_test.csv
data/raw/reddit_train.csv
data/raw/reddit_validation.csv
data/raw/reddit_test.csv
```

---

## 5. Smoke test cepat

Jalankan sampel kecil dulu untuk memastikan dependency dan GPU aman:

```bash
python scripts/run_transformer_baseline.py \
  --dataset twitter \
  --model indobert-base \
  --epochs 1 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --max-train-samples 64 \
  --max-eval-samples 32 \
  --max-predict-samples 32 \
  --output-dir results/transformer/smoke-twitter-indobert-base \
  --model-output-dir models/transformer/smoke-twitter-indobert-base
```

Script otomatis menyimpan run sample-limited ke `results/tables/transformer_smoke.csv`, bukan ke tabel baseline utama. Jadi hasil smoke test tidak akan tercampur dengan hasil Progress 3 final.

Kalau ini selesai tanpa error, lanjut full baseline.

---

## 6. Full baseline Progress 3 — Twitter IndoBERT Base

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

Script akan menyimpan:

```text
results/tables/transformer_baselines.csv
results/transformer/twitter-indobert-base/metrics.json
results/transformer/twitter-indobert-base/result_row.json
models/transformer/twitter-indobert-base/   # model checkpoint, tidak perlu commit
```

Yang perlu dikirim balik/di-commit untuk Progress 3:
- `results/tables/transformer_baselines.csv`
- `results/transformer/twitter-indobert-base/metrics.json`
- `results/transformer/twitter-indobert-base/result_row.json`

Folder `models/` tidak perlu dikirim karena besar dan sudah masuk `.gitignore`.

---

## 7. Opsi pembanding jika waktu/GPU cukup

### XLM-R Base pada Twitter

```bash
python scripts/run_transformer_baseline.py \
  --dataset twitter \
  --model xlmr-base \
  --epochs 5 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --learning-rate 1e-5 \
  --weight-decay 0.03 \
  --max-length 128 \
  --fp16 \
  --output-dir results/transformer/twitter-xlmr-base \
  --model-output-dir models/transformer/twitter-xlmr-base
```

### IndoBERT Base pada Reddit

```bash
python scripts/run_transformer_baseline.py \
  --dataset reddit \
  --model indobert-base \
  --epochs 5 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --learning-rate 1e-5 \
  --weight-decay 0.03 \
  --max-length 128 \
  --fp16 \
  --output-dir results/transformer/reddit-indobert-base \
  --model-output-dir models/transformer/reddit-indobert-base
```

Untuk laporan Progress 3, cukup satu baseline transformer dulu. Eksperimen tambahan bisa masuk Progress 4 atau Progress 5.

---

## 8. Target pembanding

Progress 2 classical ML terbaik pada Twitter:

| Model | Representasi | F1 |
|---|---|---:|
| Logistic Regression | TF-IDF | 0,7143 |
| SVM | BoW | 0,6850 |
| SVM | TF-IDF | 0,6783 |

Target paper untuk transformer terbaik:

| Dataset | Model terbaik paper | F1 |
|---|---|---:|
| Twitter | XLM-R Large | 0,7692 |
| Reddit | XLM-R Large | 0,6274 |

Catatan: Progress 3 tidak harus menyamai XLM-R Large, karena model yang dijalankan lebih ringan. Yang penting ada pembanding langsung terhadap baseline classical ML.

---

## 9. Troubleshooting

### CUDA out of memory
Turunkan batch size:

```bash
--batch-size 8 --eval-batch-size 16
```

Kalau masih OOM:

```bash
--batch-size 4 --eval-batch-size 8
```

### Training lambat
Kurangi epoch untuk percobaan awal:

```bash
--epochs 3
```

### `accelerate` error
Install ulang:

```bash
pip install -U accelerate transformers
```

Lalu restart runtime Colab.

### Hasil lebih rendah dari classical ML
Itu masih valid untuk dibahas. Jelaskan bahwa transformer butuh tuning lebih lanjut, dan Progress 4 memang disiapkan untuk optimasi terarah.

---

## 10. Setelah selesai run

Kirim/commit file hasil:

```bash
git add results/tables/transformer_baselines.csv results/transformer/
git commit -m "results: add Progress 3 transformer baseline outputs"
git push
```

Setelah hasil ada, bagian laporan yang harus diisi:
- `2.2.2 Model Transformer`
- `3.2.2 Tahapan Eksperimen Transformer`
- `3.3.2 Hasil Model Transformer`
