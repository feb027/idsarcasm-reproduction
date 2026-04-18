# Progress 2: Dataset, EDA, dan Baseline Classical ML

**Status:** 🟡 Berjalan sebagian
**Tanggal mulai:** 12 April 2026
**Revisi struktur:** 18 April 2026

---

## Ringkasan revisi

Dokumen ini sekarang menjadi versi gabungan dari:
- **Progress 2 lama:** Dataset & Preprocessing (EDA)
- **Progress 3 lama:** Classical ML Baseline

Artinya, Progress 2 tidak lagi dianggap selesai hanya karena EDA selesai. Progress ini baru benar-benar selesai jika baseline classical ML sudah diimplementasikan, dijalankan, dan dibandingkan dengan hasil paper.

## Status saat ini

### Sudah selesai
- [x] Dataset di-download dari HuggingFace
- [x] Struktur dataset diverifikasi
- [x] EDA utama selesai
- [x] Figure EDA dibuat dan disimpan
- [x] Mismatch ukuran dataset Twitter berhasil dijelaskan
- [x] Metodologi baseline classical ML dari paper sudah diidentifikasi
- [x] Target metrik paper untuk baseline sudah dicatat

### Belum selesai
- [x] Menyiapkan script baseline classical ML dan local-run guide
- [ ] Menjalankan baseline Twitter: LR, NB, SVM
- [ ] Menjalankan baseline Reddit: LR, NB, SVM
- [ ] Menyimpan tabel hasil baseline aktual dari local run
- [ ] Membuat komparasi awal terhadap paper

---

## 1. Tujuan Progress 2

Progress 2 dirancang sebagai fase eksperimen inti awal. Fokusnya bukan hanya memahami data, tetapi juga menyiapkan fondasi reproduksi yang benar-benar bisa dipakai untuk membandingkan hasil dengan paper.

Secara praktis, Progress 2 harus menghasilkan dua hal:
1. pemahaman dataset yang kuat melalui EDA, dan
2. baseline classical ML yang siap dijadikan titik acuan untuk fase analisis dan improvement.

---

## 2. Dataset Download

Dataset di-download dari HuggingFace menggunakan `datasets` library:

| Dataset | HuggingFace ID | Total |
|---------|---------------|-------|
| Reddit Indonesia Sarcastic | `w11wo/reddit_indonesia_sarcastic` | 14,116 |
| Twitter Indonesia Sarcastic | `w11wo/twitter_indonesia_sarcastic` | 2,684 |

Script: `scripts/download_data.py`

---

## 3. Struktur Dataset

### Reddit
- **Text column:** `text` (PII-masked, sarcasm-tag removed)
- **Kolom lain:** `body`, `label`, `permalink`, `subreddit`, `lang_fastText`, `created_utc`, `author`, `score`
- **Split:** train 9,881 / val 1,411 / test 2,824

### Twitter
- **Text column:** `tweet` (PII-masked)
- **Kolom lain:** `label`
- **Split:** train 1,878 / val 268 / test 538

---

## 4. Klarifikasi Ukuran Dataset Twitter

Paper menyebut Twitter dataset = 12,861, tetapi dataset yang tersedia di HuggingFace untuk eksperimen benchmark berjumlah 2,684.

Dari halaman HuggingFace:
- Total raw = 17,718
- Total cleaned unbalanced = 12,861 (671 sarcastic + 12,190 non-sarcastic)
- Total cleaned balanced = 2,684 (671 sarcastic + 2,013 non-sarcastic, rasio 1:3)

Interpretasi kerja proyek ini:
- angka **12,861** adalah versi cleaned unbalanced yang disebut di paper,
- angka **2,684** adalah versi balanced yang dipublikasikan untuk benchmark,
- reproduksi saat ini menggunakan **versi balanced 2,684**, karena itu yang konsisten dengan split benchmark yang tersedia.

---

## 5. Hasil EDA

### 5.1 Label Distribution

| Dataset | Label | Count | Ratio |
|---------|-------|-------|-------|
| Reddit | Non-sarcasm (0) | 10,587 | 75.00% |
| Reddit | Sarcasm (1) | 3,529 | 25.00% |
| Twitter | Non-sarcasm (0) | 2,013 | 75.00% |
| Twitter | Sarcasm (1) | 671 | 25.00% |

Insight utama:
- kedua dataset memiliki rasio **exact 25% sarcasm**,
- distribusi ini konsisten di train/validation/test,
- konfigurasi ini sangat mungkin sudah mengikuti benchmark style SemEval 1:3 ratio.

### 5.2 Per-Split Distribution

**Reddit**

| Split | Total | Sarcasm | Non-sarcasm |
|-------|-------|---------|-------------|
| Train | 9,881 | 2,470 | 7,411 |
| Val | 1,411 | 353 | 1,058 |
| Test | 2,824 | 706 | 2,118 |

**Twitter**

| Split | Total | Sarcasm | Non-sarcasm |
|-------|-------|---------|-------------|
| Train | 1,878 | 470 | 1,408 |
| Val | 268 | 67 | 201 |
| Test | 538 | 134 | 404 |

### 5.3 Text Length (character count)

| Dataset | Label | Mean | Std | Min | Max |
|---------|-------|------|-----|-----|-----|
| Reddit | Non-sarcasm | 103.6 | 88.9 | 4 | 1,134 |
| Reddit | Sarcasm | 67.3 | 47.7 | 5 | 527 |
| Twitter | Non-sarcasm | 113.8 | 67.8 | 14 | 584 |
| Twitter | Sarcasm | 117.8 | 55.0 | 18 | 297 |

Insight utama:
- pada Reddit, komentar sarkastik cenderung lebih pendek,
- pada Twitter, panjang teks antar label relatif mirip,
- ini mengindikasikan tantangan Twitter mungkin lebih banyak ada di wording, sedangkan Reddit lebih kontekstual.

### 5.4 Data Quality

| Check | Reddit | Twitter |
|-------|--------|---------|
| Missing values | 0 | 0 |
| Duplicate texts | 10 | 0 |

Kesimpulan sementara:
- kualitas data cukup bersih untuk langsung dipakai baseline,
- tidak ada kebutuhan mendesak melakukan pembersihan ulang di tahap ini.

---

## 6. Preprocessing dari Paper

Berdasarkan paper, preprocessing yang dilakukan author:
1. **Language filtering** (Reddit only): fastText language detection, keep Indonesian/Javanese/Minangkabau/Malay/Sundanese
2. **Near-deduplication:** MinHash LSH — mengurangi Twitter sarcastic dari 4,350 → 671
3. **PII masking:** username → `<username>`, hashtag → `<hashtag>`, email → `<email>`, URL → `<link>`
4. **Sarcasm tag removal** (Reddit only): hapus `/s` suffix
5. **Random sampling:** 1:3 ratio sarcastic:non-sarcastic
6. **Split:** 70% train / 10% val / 20% test

Untuk reproduksi proyek ini, preprocessing berat tidak perlu diulang dari nol karena dataset HuggingFace sudah berada pada bentuk benchmark yang siap dipakai.

---

## 7. Baseline Classical ML yang Akan Direproduksi

### Model
- Logistic Regression
- Naive Bayes (Multinomial)
- SVM / SVC

### Feature representation
- Bag of Words (`CountVectorizer`)
- TF-IDF (`TfidfVectorizer`)

### Tokenization
- `nltk.word_tokenize`

### Validation strategy
- `GridSearchCV`
- `PredefinedSplit` dengan skema train + validation digabung, validation dijadikan holdout untuk tuning

### Evaluation metrics
- F1-score (utama)
- Accuracy
- Precision
- Recall

---

## 8. Target Hasil yang Ingin Didekati

### Twitter (TF-IDF)
| Model | Target F1 |
|-------|-----------|
| Logistic Regression | ~0.7142 |
| SVM | ~0.6782 |
| Naive Bayes | ~0.6721 |

### Reddit (TF-IDF)
| Model | Target F1 |
|-------|-----------|
| Logistic Regression | ~0.4887 |
| SVM | ~0.4467 |
| Naive Bayes | ~0.4591 |

Target ini akan menjadi acuan utama ketika baseline benar-benar dijalankan.

---

## 9. Rencana Eksekusi yang Termasuk dalam Progress 2

Bagian ini **belum dikerjakan**, tetapi secara struktur sekarang sudah menjadi bagian resmi dari Progress 2.

### 9.1 Persiapan environment
- install dependency minimum: `nltk`, `datasets`, `scikit-learn`, `pandas`, `numpy`
- download tokenizer NLTK (`punkt`, `punkt_tab`) bila belum ada
- pastikan dataset bisa dimuat tanpa error
- ikuti panduan lokal di `docs/progress-2-local-run-guide.md`

### 9.2 Implementasi baseline
- siapkan script/notebook untuk menjalankan LR, NB, SVM
- jalankan BoW dan TF-IDF untuk tiap model
- gunakan train+val untuk tuning, test untuk evaluasi final

### 9.3 Penyimpanan hasil
- simpan hasil numerik ke `results/tables/`
- dokumentasikan best parameter dan metrik utama
- catat perbedaan terhadap angka di paper

### 9.4 Output minimum agar Progress 2 bisa ditutup
- minimal 1 tabel hasil baseline Twitter
- minimal 1 tabel hasil baseline Reddit
- catatan gap awal terhadap paper
- dokumentasi update di file ini

---

## 10. Output yang Sudah Ada

- [x] Dataset downloaded ke `data/raw/`
- [x] Notebook EDA: `notebooks/01_eda.ipynb`
- [x] Dokumentasi ringkasan paper dan EDA
- [x] Figure EDA tersimpan di `results/figures/`
- [x] Script baseline local-ready: `scripts/run_classical_baselines.py`
- [x] Panduan run lokal: `docs/progress-2-local-run-guide.md`

## 11. Figures Generated

### Label Distribution

![Label Distribution](../results/figures/label_distribution.png)

### Text Length Distribution

![Text Length Distribution](../results/figures/text_length_distribution.png)

### Split Distribution

![Split Distribution](../results/figures/split_distribution.png)

---

## 12. Kriteria Selesai Progress 2

Progress 2 baru boleh ditandai **selesai** jika seluruh poin berikut sudah terpenuhi:
- [ ] baseline classical ML Twitter selesai dijalankan
- [ ] baseline classical ML Reddit selesai dijalankan
- [ ] hasil tersimpan dalam tabel yang rapi
- [ ] target paper vs hasil reproduksi sudah dibandingkan
- [ ] dokumentasi Progress 2 mencakup EDA + baseline

Sampai saat ini, Progress 2 masih **parsial**, bukan selesai penuh.
