# Rencana Reproduksi — 6 Progress

## Progress 1: Topik & Paper Selection ✅
- [x] Pilih topik: Indonesian Sarcasm Detection
- [x] Pilih paper: IdSarcasm (Suhartono et al., 2024)
- [x] Paper summary & identifikasi algoritma
- [x] Setup GitHub repo

**Algoritma yang digunakan di paper:**
Logistic Regression, Naive Bayes, SVC (classical); IndoBERT Base/Large, mBERT, XLM-R Base/Large (transformer); BLOOMZ, mT0 (zero-shot LLM)

---

## Progress 2: Dataset & Preprocessing (Minggu 3-4)
- [ ] Download dataset dari HuggingFace (Reddit + Twitter)
- [ ] Eksplorasi data: distribusi label, panjang teks, contoh sarcasm/non-sarcasm
- [ ] Reproduksi preprocessing pipeline paper
- [ ] Train/val/test split sesuai paper
- [ ] Analisis class balance

**Deliverable:** Notebook EDA + dataset siap training

---

## Progress 3: Classical ML Baseline (Minggu 5)
- [ ] TF-IDF vectorization
- [ ] Training Logistic Regression, Naive Bayes, SVC
- [ ] Evaluasi F1-score
- [ ] Bandingkan hasil ke paper

**Deliverable:** Script training + hasil tabel

---

## Progress 4: Transformer Fine-tuning (Minggu 6-7)
- [ ] Fine-tune IndoBERT Base (IndoNLU) — kedua dataset
- [ ] Fine-tune XLM-R Base — kedua dataset
- [ ] (Stretch goal: XLM-R Large jika resource cukup / pakai Colab)
- [ ] Evaluasi F1-score
- [ ] Bandingkan hasil ke paper

**Deliverable:** Model weights + hasil evaluasi

---

## Progress 5: Evaluation & Comparison (Minggu 8)
- [ ] Tabel perbandingan semua model (reproduksi vs paper)
- [ ] Analisis gap: kenapa hasil beda (jika ada)
- [ ] Confusion matrix per model
- [ ] Error analysis: contoh salah klasifikasi

**Deliverable:** Laporan evaluasi + visualisasi

---

## Progress 6: Improvement Proposal & Final Report (Minggu 9-10)
- [ ] Implementasi improvement (pilih 1-2):
  - Few-shot learning dengan LLM kecil
  - Data augmentation
  - Ensemble classical + transformer
- [ ] Evaluasi improvement vs baseline
- [ ] Tulis laporan akhir
- [ ] Finalisasi GitHub repo

**Deliverable:** Laporan final + code + model

---

## Hardware Plan
- **Classical ML & EDA:** Local PC (Windows i5-12400F, 16GB RAM)
- **Transformer fine-tuning:** Google Colab (GPU) atau local jika feasible
- **Small experiments:** Local PC

## Timeline
| Progress | Target |
|----------|--------|
| 1 | Apr 12 ✅ |
| 2 | Apr 19-26 |
| 3 | Apr 26 - Mei 3 |
| 4 | Mei 3-17 |
| 5 | Mei 17-24 |
| 6 | Mei 24 - Jun 7 |
