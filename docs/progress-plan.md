# Rencana Reproduksi — 6 Progress

**Paper:** IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection
**Target:** Reproduksi baseline classical ML + evaluasi perbandingan

---

## Progress 1: Topik & Paper Selection ✅

**Tanggal:** 12 April 2026

- [x] Pilih topik: Indonesian Sarcasm Detection
- [x] Pilih paper: IdSarcasm (Suhartono et al., IEEE Access 2024)
- [x] Analisis kontribusi, metode, hasil paper
- [x] Identifikasi algoritma dan reproduksi scope
- [x] Setup GitHub repo + struktur project
- [x] Tulis paper summary, progress plan, requirements

**Deliverable:** Repo GitHub, paper summary, rencana reproduksi

---

## Progress 2: Dataset & EDA ✅

**Tanggal:** 12 April 2026

- [x] Download dataset dari HuggingFace (Reddit 14,116 + Twitter 2,684)
- [x] EDA: distribusi label, panjang teks, split info, contoh data
- [x] Analisis class balance (25% sarcasm, 1:3 ratio)
- [x] Data quality check (missing values, duplicates)
- [x] Clarification Twitter dataset size (12.8k unbalanced vs 2.6k balanced)
- [x] Dokumentasi hasil EDA

**Deliverable:** EDA notebook, results/figures, dokumentasi

---

## Progress 3: Classical ML Baseline ⬜

**Target:** ~Minggu 3-4

- [ ] Setup environment: pip install nltk, datasets, scikit-learn
- [ ] Download NLTK punkt tokenizer
- [ ] Reproduksi Twitter: LR, NB, SVM (BoW + TF-IDF)
- [ ] Reproduksi Reddit: LR, NB, SVM (BoW + TF-IDF)
- [ ] GridSearchCV dengan PredefinedSplit
- [ ] Evaluasi: F1, accuracy, precision, recall
- [ ] Bandingkan hasil ke paper (target: match ±0.01)

**Metode reproduksi:**
1. `load_dataset("w11wo/twitter_indonesia_sarcastic")`
2. Vectorize dengan CountVectorizer / TfidfVectorizer
3. Tokenizer: `nltk.word_tokenize`
4. GridSearchCV: LR, SVM, NB
5. PredefinedSplit (train+val digabung, val=holdout)
6. Evaluasi di test set

**Target hasil (Twitter TF-IDF):**
| Model | Target F1 |
|-------|-----------|
| LR | ~0.7142 |
| SVM | ~0.6782 |
| NB | ~0.6721 |

**Deliverable:** Script/classical_ml.py, results/tabel, perbandingan ke paper

---

## Progress 4: Perbandingan & Perluasan ⬜

**Target:** ~Minggu 5

- [ ] Jalankan baseline Reddit (jika belum di Progress 3)
- [ ] Analisis gap: hasil reproduksi vs paper
- [ ] Confusion matrix per model
- [ ] Error analysis: contoh salah klasifikasi
- [ ] (Stretch) Fine-tune 1 transformer via Colab

**Deliverable:** Analisis perbandingan, error analysis

---

## Progress 5: Evaluation & Comparison ⬜

**Target:** ~Minggu 6

- [ ] Tabel komprehensif semua hasil
- [ ] Visualisasi perbandingan (bar chart, confusion matrix)
- [ ] Analisis gap dan penyebab perbedaan
- [ ] Perbandingan Twitter vs Reddit difficulty

**Deliverable:** Tabel evaluasi, visualisasi, analisis

---

## Progress 6: Improvement & Final Report ⬜

**Target:** ~Minggu 7-8

- [ ] Implementasi improvement (pilih 1-2):
  - Data augmentation (back-translation / synonym replacement)
  - Ensemble classical + transformer
  - Weighted loss untuk imbalance
- [ ] Evaluasi improvement vs baseline
- [ ] Tulis laporan akhir
- [ ] Finalisasi GitHub repo

**Deliverable:** Laporan final, code, model

---

## Hardware Plan

| Task | Lokasi | Alasan |
|------|--------|--------|
| Classical ML & EDA | PC Lokal (Windows) | CPU cukup, dataset kecil |
| Transformer fine-tuning | Google Colab (GPU) | Butuh GPU, AMD lokal bermasalah |
| Coding & dokumentasi | VPS / Lokal | Fleksibel |

## Timeline Estimasi

| Progress | Target |
|----------|--------|
| 1 | 12 Apr ✅ |
| 2 | 12 Apr ✅ |
| 3 | 19 Apr - 3 Mei |
| 4 | 3 - 10 Mei |
| 5 | 10 - 17 Mei |
| 6 | 17 - 31 Mei |
