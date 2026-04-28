# Progress 1: Topik & Paper Selection

**Status:** ✅ Selesai
**Tanggal:** 12 April 2026

---

## 1. Topik

**Indonesian Sarcasm Detection** — deteksi sarkasme dalam teks bahasa Indonesia dari media sosial.

Sarkasme merupakan salah satu tantangan terbesar dalam NLP karena sifatnya yang kontekstual dan budaya-spesifik. Untuk bahasa Indonesia, sangat sedikit penelitian dan benchmark publik yang tersedia.

## 2. Paper yang Dipilih

**IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection**

- **Penulis:** Derwin Suhartono, Wilson Wongso, Alif Tri Handoyo
- **Institusi:** Bina Nusantara University, Jakarta
- **Publikasi:** IEEE Access, Volume 12, 2024
- **DOI:** 10.1109/ACCESS.2024.3416955
- **GitHub:** https://github.com/w11wo/id_sarcasm

### Alasan Dipilih
1. **Topik relevan:** NLP untuk bahasa Indonesia (low-resource language)
2. **Dataset publik:** Tersedia di HuggingFace, bisa diakses langsung
3. **Reproducible:** Code dan results tersedia di GitHub
4. **Scope realistis:** Classical ML baseline bisa dijalankan di lokal tanpa GPU
5. **Ada ruang improvement:** Bisa propose enhancement setelah reproduksi

## 3. Kontribusi Paper

1. Dataset benchmark publik pertama untuk Indonesian sarcasm detection
2. Baseline komprehensif: classical ML, fine-tuned transformer, zero-shot LLM
3. Eksperimen data augmentation dan weighted loss untuk handle imbalance
4. Analisis perbandingan Twitter vs Reddit dataset

## 4. Algoritma di Paper

### Classical Machine Learning
| Model | Library | Feature |
|-------|---------|---------|
| Logistic Regression | scikit-learn | BoW, TF-IDF |
| Naive Bayes (Multinomial) | scikit-learn | BoW, TF-IDF |
| SVM | scikit-learn | BoW, TF-IDF |

### Fine-tuned Pre-trained Language Models
| Model | #Params | Type |
|-------|---------|------|
| IndoBERT Base (IndoNLU) | 124M | Monolingual |
| IndoBERT Large (IndoNLU) | 335M | Monolingual |
| IndoBERT Base (IndoLEM) | 111M | Monolingual |
| mBERT | 178M | Multilingual |
| XLM-R Base | 278M | Multilingual |
| XLM-R Large | 560M | Multilingual |

### Zero-shot LLM
| Model | Range |
|-------|-------|
| BLOOMZ | 560M → 7.1B |
| mT0 | Small → XL |

## 5. Hasil Paper (F1-Score)

| Model | Reddit | Twitter |
|-------|--------|---------|
| Logistic Regression (TF-IDF) | 0.4887 | 0.7142 |
| Naive Bayes (TF-IDF) | 0.4591 | 0.6721 |
| SVM (TF-IDF) | 0.4467 | 0.6782 |
| IndoBERT Base (IndoNLU) | 0.6100 | 0.7273 |
| IndoBERT Large (IndoNLU) | 0.6184 | 0.7160 |
| XLM-R Base | 0.5690 | 0.7386 |
| **XLM-R Large** | **0.6274** | **0.7692** |
| BLOOMZ-7.1B (zero-shot) | 0.4036 | 0.3968 |

**Best model:** XLM-R Large (F1: 0.6274 Reddit, 0.7692 Twitter)

## 6. Reproduction Scope yang Dipilih

### Fase utama sekarang (Progress 2 revisi)
- Dataset acquisition dan verifikasi split
- EDA lengkap untuk Reddit dan Twitter
- Reproduksi baseline classical ML pada Twitter dan Reddit
- Model: Logistic Regression, Naive Bayes, SVM
- Feature: BoW + TF-IDF
- Target: mendekati F1 paper dan menghasilkan tabel baseline awal

### Fase lanjutan (Progress 3–5)
- Reproduksi transformer baseline pada benchmark IdSarcasm
- Optimasi transformer secara terarah
- Analisis komparatif dan error analysis atas seluruh hasil

### Stretch / penguatan akhir (Progress 6)
- Finalisasi laporan, repositori, visualisasi, dan narasi hasil proyek

## 7. Output Progress 1

- [x] Paper dipilih dan dianalisis
- [x] GitHub repo dibuat: https://github.com/feb027/idsarcasm-reproduction
- [x] Struktur project setup
- [x] Paper summary ditulis
- [x] Progress plan dibuat
- [x] Requirements.txt
- [x] Google Sheets: algoritma dicatat

## 8. Google Sheets

**Judul proyek yang dipakai di Google Sheets dosen:**
```
Optimasi Performa Model Transformer dalam Klasifikasi Sarkasme Teks Berbahasa Indonesia Berdasarkan Benchmark IdSarcasm
```

**Kolom "Algoritma yang digunakan di paper":**
```
Logistic Regression, Naive Bayes, SVM, IndoBERT, mBERT, XLM-R, BLOOMZ, mT0
```

**Kolom "Algoritma yang direproduksi / dikembangkan":**
```
Progress 2: Logistic Regression, Naive Bayes, SVM; Progress 3-4: IndoBERT Base / XLM-R Base + optimasi transformer
```
