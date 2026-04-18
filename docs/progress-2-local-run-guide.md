# Progress 2 — Local Run Guide (Windows PC / WSL2)

Guide ini disiapkan untuk menjalankan Progress 2 di **PC lokal kamu**, bukan di VPS.

Target Progress 2 saat ini:
1. jalankan baseline classical ML pada benchmark IdSarcasm,
2. dapatkan tabel hasil Twitter + Reddit,
3. lalu bandingkan dengan target F1 paper.

## Rekomendasi environment

Paling aman pakai:
- **Windows 11 + WSL2 Ubuntu**
- Python 3.10 atau 3.11
- Jalankan semuanya dari folder repo di WSL2

Kenapa:
- script dan dependency lebih natural dijalankan di environment Linux,
- lebih dekat dengan workflow repo saat ini,
- lebih gampang kalau nanti lanjut ke Colab atau transformer.

## Step 0 — buka repo di WSL2

Kalau repo ada di Windows drive, masuk via WSL2:

```bash
cd /mnt/c/path/ke/idsarcasm-reproduction
```

Kalau repo sudah ada di home WSL2:

```bash
cd ~/idsarcasm-reproduction
```

Cek dulu:

```bash
git status
```

## Step 1 — update repo

```bash
git pull origin master
```

Kalau branch default kamu bukan `master`, sesuaikan.

## Step 2 — buat virtual environment baru

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Step 3 — install dependency

```bash
pip install -r requirements.txt
```

Kalau `requirements.txt` berhasil terpasang, `nltk` sudah ikut ter-install.

## Step 4 — optional: download CSV mentah lokal

Ini optional, buat dokumentasi/backup split lokal.

```bash
python scripts/download_data.py
```

Kalau berhasil, nanti akan muncul file seperti:
- `data/raw/twitter_train.csv`
- `data/raw/twitter_validation.csv`
- `data/raw/twitter_test.csv`
- `data/raw/reddit_train.csv`
- `data/raw/reddit_validation.csv`
- `data/raw/reddit_test.csv`

## Step 5 — jalankan baseline Twitter dulu

Ini jalur paling aman untuk start.

```bash
python scripts/run_classical_baselines.py --dataset twitter --output-dir results/tables
```

Yang akan dijalankan:
- Logistic Regression
- Naive Bayes
- SVM
- BoW
- TF-IDF
- tuning via GridSearchCV + PredefinedSplit

Output utama:
- `results/tables/classical_baselines_twitter.csv`
- beberapa file `*_best_params.json`

## Step 6 — cek hasil Twitter

Buka CSV hasil, lalu bandingkan ke target paper berikut:

### Target Twitter (TF-IDF)
- LR: ~0.7142
- NB: ~0.6721
- SVM: ~0.6782

Kalau hasilmu belum dekat, jangan panik dulu. Yang penting:
- script jalan tanpa error,
- tabel hasil keluar,
- semua model tercatat.

## Step 7 — jalankan baseline Reddit

```bash
python scripts/run_classical_baselines.py --dataset reddit --output-dir results/tables
```

Output utama:
- `results/tables/classical_baselines_reddit.csv`
- file `*_best_params.json`

### Target Reddit (TF-IDF)
- LR: ~0.4887
- NB: ~0.4591
- SVM: ~0.4467

## Step 8 — kalau mau langsung full run sekaligus

Kalau Twitter sudah aman dan kamu mau jalan dua dataset sekaligus:

```bash
python scripts/run_classical_baselines.py --dataset all --output-dir results/tables
```

## Step 9 — file yang perlu kamu kirim / commit setelah run berhasil

Minimal simpan ini:
- `results/tables/classical_baselines_twitter.csv`
- `results/tables/classical_baselines_reddit.csv`
- file `*_best_params.json` yang relevan
- update `docs/progress-2.md` dengan hasil real run

Kalau nanti mau rapih, kita bisa bikin 1 file ringkasan lagi, misalnya:
- `docs/progress-2-results.md`

## Step 10 — troubleshooting cepat

### A. Error NLTK tokenizer / punkt
Kalau muncul error resource NLTK, jalankan:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Lalu ulangi command baseline.

### B. Laptop / PC terasa berat
Coba kurangi paralel worker:

```bash
python scripts/run_classical_baselines.py --dataset twitter --output-dir results/tables --n-jobs 1
```

### C. Mau test cepat dulu
Coba satu model dulu:

```bash
python scripts/run_classical_baselines.py --dataset twitter --models lr --vectorizers tfidf --output-dir results/tables
```

### D. Import error / dependency aneh
Pastikan kamu sedang berada di venv yang benar:

```bash
which python
python --version
pip list | grep -E "datasets|scikit-learn|nltk|pandas|numpy"
```

## Step 11 — setelah run selesai, apa yang dilakukan?

Urutan paling pas setelah kamu selesai run lokal:
1. kirim hasil CSV / angka F1 ke gue,
2. gue bantu baca gap vs paper,
3. gue update `docs/progress-2.md`,
4. lalu kita putuskan lanjut ke:
   - Progress 3 transformer baseline, atau
   - perapihan hasil classical dulu.

## Ringkas super singkat

Kalau mau versi paling cepat:

```bash
cd ~/idsarcasm-reproduction
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_classical_baselines.py --dataset twitter --output-dir results/tables
python scripts/run_classical_baselines.py --dataset reddit --output-dir results/tables
```

Setelah itu cek file di `results/tables/`.
