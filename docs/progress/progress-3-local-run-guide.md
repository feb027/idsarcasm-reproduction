# Progress 3 Local/Colab Run Guide — Paper-Faithful Transformer Baselines

Progress 3 menjalankan **dua baseline transformer** pada benchmark IdSarcasm dan membandingkannya dengan baseline classical ML dari Progress 2.

Target resmi Progress 3:
1. **IndoBERT Base** — `indobenchmark/indobert-base-p1`
2. **XLM-R Base** — `xlm-roberta-base`

Keduanya dijalankan pada **Twitter Indonesia Sarcastic** (`w11wo/twitter_indonesia_sarcastic`) karena dataset ini kecil dan paling aman untuk reproduksi awal.

---

## 1. Apakah setting-nya sama dengan source code paper?

Runner `scripts/run_transformer_baseline.py` sekarang memakai preset default yang mengikuti recipe asli paper di:

- `source-code/original-id-sarcasm/recipes/twitter/baseline/indobert_indonlu_base_twitter.sh`
- `source-code/original-id-sarcasm/recipes/twitter/baseline/xlmr_base_twitter.sh`

Setting yang disamakan:

| Komponen | Nilai |
|---|---:|
| Dataset | `w11wo/twitter_indonesia_sarcastic` |
| Text column | `tweet` |
| Label column | `label` |
| Max sequence length | 128 |
| Train batch size | 32 |
| Eval batch size | 64 |
| Learning rate | 1e-5 |
| LR scheduler | cosine |
| Weight decay | 0.03 |
| Label smoothing | 0.0 |
| Epoch maksimum | 100 |
| Shuffle train split | yes |
| Early stopping | yes, patience 3, threshold 0.01 |
| Padding | pad to max length |
| Metric best model | F1 |
| Seed | 42 |
| FP16 | yes, jika CUDA tersedia |

Perbedaan yang sengaja dibuat:
- tidak memakai `--push_to_hub`, karena tidak perlu upload model ke HuggingFace;
- output model masuk ke `models/transformer/...` dan tidak di-commit;
- metrik ringkas disimpan ke `results/tables/transformer_baselines.csv` dan `results/transformer/...` agar mudah masuk laporan;
- script punya smoke-test mode yang terpisah dari hasil final.

Jadi ini **paper-faithful untuk setting training utama**, tapi tetap disesuaikan untuk kebutuhan repo UAS.

---

## 2. Environment yang disarankan

### Opsi paling aman: Google Colab GPU

Di Colab:
1. buka `Runtime > Change runtime type`,
2. pilih GPU, misalnya T4,
3. jalankan notebook `notebooks/02_transformer_baseline_colab.ipynb`.

### Opsi PC lokal

PC lokal pengguna: i5-12400F, RX 6600 8GB, RAM 16GB.

Kemungkinan:
- **CPU lokal:** bisa untuk Twitter, tapi training bisa lama karena epoch maksimum 100 dengan early stopping.
- **RX 6600:** mungkin bisa jika stack ROCm/HIP aktif, tetapi di Windows/WSL2 biasanya lebih berisiko daripada CUDA.
- **WSL2 tanpa CUDA:** tetap bisa, tapi lambat.

Rekomendasi: pakai **Colab GPU** untuk hasil utama, lokal hanya untuk smoke test atau debugging.

---

## 3. Clone repo

```bash
git clone https://github.com/feb027/idsarcasm-reproduction.git
cd idsarcasm-reproduction
```

Kalau repo sudah ada:

```bash
cd idsarcasm-reproduction
git pull
```

---

## 4. Install dependency

```bash
pip install -r requirements.txt
```

Kalau Colab meminta restart runtime setelah install, restart lalu masuk lagi ke folder repo.

---

## 5. Download/cache dataset

Runner bisa langsung mengambil dataset dari HuggingFace. Namun untuk konsistensi dengan Progress 2, cache CSV dulu:

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

## 6. Smoke test cepat

Jalankan sampel kecil dulu untuk memastikan dependency, dataset, dan GPU aman:

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

Karena memakai `--max-*-samples`, hasil smoke test otomatis masuk ke:

```text
results/tables/transformer_smoke.csv
```

Bukan ke tabel final. Ini mencegah angka sample kecil tercampur dengan hasil Progress 3.

---

## 7. Full baseline 1 — IndoBERT Base Twitter

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

---

## 8. Full baseline 2 — XLM-R Base Twitter

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

## 9. Output final Progress 3

Runner akan menyimpan:

```text
results/tables/transformer_baselines.csv
results/transformer/twitter-indobert-base/metrics.json
results/transformer/twitter-indobert-base/result_row.json
results/transformer/twitter-xlmr-base/metrics.json
results/transformer/twitter-xlmr-base/result_row.json
models/transformer/...   # checkpoint, tidak perlu commit
```

Yang perlu di-commit untuk Progress 3:

```bash
git add results/tables/transformer_baselines.csv results/transformer/
git commit -m "results: add Progress 3 transformer baseline outputs"
git push
```

Folder `models/` tidak perlu dikirim karena besar dan sudah masuk `.gitignore`.

---

## 10. Target pembanding

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

Catatan: Progress 3 memakai IndoBERT Base dan XLM-R Base, bukan XLM-R Large. Jadi target utamanya bukan menyamai XLM-R Large, melainkan mendapatkan baseline transformer yang valid dan bisa dibandingkan dengan classical ML.

---

## 11. Troubleshooting

### CUDA out of memory
Turunkan batch size:

```bash
--batch-size 16 --eval-batch-size 32
```

Kalau masih OOM:

```bash
--batch-size 8 --eval-batch-size 16
```

Catat perubahan ini di laporan karena tidak lagi 100% sama dengan recipe paper.

### Training terlalu lama
Karena epoch maksimum 100, training bisa terlihat panjang. Namun early stopping aktif, jadi training seharusnya berhenti lebih awal jika F1 validasi tidak membaik.

Untuk percobaan saja, boleh gunakan:

```bash
--epochs 5
```

Tapi jangan pakai hasil itu sebagai hasil final paper-faithful Progress 3 tanpa catatan.

### `accelerate` / `transformers` error

```bash
pip install -U accelerate transformers
```

Lalu restart runtime Colab.

### Hasil lebih rendah dari classical ML
Itu tetap valid untuk Progress 3. Jelaskan bahwa transformer base belum tentu langsung unggul tanpa optimasi, dan Progress 4 memang disiapkan untuk tuning.

---

## 12. Setelah hasil ada

Bagian laporan utama yang harus diisi:

- `2.2.2 Model Transformer`
- `3.2.2 Tahapan Eksperimen Transformer`
- `3.3.2 Hasil Model Transformer`

Isi angka hanya setelah full run selesai, bukan dari smoke test.
