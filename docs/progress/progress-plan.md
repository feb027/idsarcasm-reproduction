# Rencana Reproduksi — 6 Progress (Revisi)

**Paper:** IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection
**Target utama:** membangun fondasi reproduksi dari baseline classical ML, lalu bergerak ke baseline transformer dan optimasi transformer yang terukur pada benchmark IdSarcasm.

---

## Catatan revisi struktur

Per 18 April 2026, struktur progress direvisi agar alurnya lebih realistis dan lebih berbobot:
- **Progress 2 lama (Dataset & EDA)** dan **Progress 3 lama (Classical ML Baseline)** digabung menjadi **Progress 2 baru**.
- Setelah itu, fase berikutnya dibuat lebih rinci supaya alurnya bergerak dari baseline classical → baseline transformer → optimasi transformer → analisis komparatif → finalisasi laporan.

Status saat ini:
- Progress 1: ✅ selesai
- Progress 2: ✅ selesai (EDA + baseline classical Twitter/Reddit sudah jalan dan hasil tabel tersimpan)
- Progress 3: ✅ selesai (12/12 baseline fine-tuned transformer paper pada Twitter + Reddit sudah dijalankan dan hasil/log tersimpan)
- Progress 4: ✅ selesai sebagai complete attempt (Twitter 9/9 selesai; Reddit 5/9 selesai; 4 run Reddit dicoba tetapi runtime/session Colab habis dan log tersimpan)
- Progress 5: ✅ optimasi dan eksperimen lanjutan selesai
- Progress 6: ✅ analisis komparatif, finalisasi laporan, dashboard GitHub Pages, model card, error analysis, confusion matrix, CI ringan, dan perapian repository selesai; update akhir Twitter XLM-R Large lr=2e-5 melampaui paper

---

## Progress 1: Topik, Paper, dan Target Reproduksi ✅

**Tanggal:** 12 April 2026

### Tujuan
Menetapkan paper utama, memahami kontribusi paper, menentukan scope reproduksi yang realistis, dan menyiapkan repositori kerja.

### Cakupan
- [x] Menetapkan topik: Indonesian sarcasm detection
- [x] Memilih paper utama: IdSarcasm (IEEE Access 2024)
- [x] Meninjau kontribusi, dataset, model, dan metrik paper
- [x] Menentukan scope reproduksi awal
- [x] Menyiapkan repo GitHub dan struktur folder proyek
- [x] Menyusun ringkasan paper dan progress plan awal

### Output
- Repo proyek siap pakai
- `docs/paper-summary.md`
- `docs/progress/progress-1.md`
- `docs/progress/progress-plan.md`
- `source-code/original-id-sarcasm/` — snapshot repo asli paper untuk referensi dan adaptasi fase berikutnya

### Gate kelulusan progress
Progress 1 dinyatakan selesai jika paper sudah final, scope sudah jelas, dan environment kerja proyek sudah memiliki struktur dokumentasi dasar.

---

## Progress 2: Dataset, EDA, dan Baseline Classical ML ✅

**Status detail:**
- **Sudah selesai:** akuisisi dataset, validasi split, EDA, klarifikasi mismatch ukuran dataset Twitter, identifikasi metodologi baseline classical ML dari paper, eksekusi baseline Twitter dan Reddit, serta penyimpanan tabel hasil.
- **Hasil utama:** reproduksi baseline classical sekarang sudah cukup dekat dengan repo/paper, terutama pada Twitter TF-IDF Logistic Regression dan Reddit TF-IDF SVM.

### Tujuan
Menggabungkan pekerjaan persiapan eksperimen dan eksekusi baseline classical ML ke dalam satu fase yang utuh, sehingga Progress 2 tidak hanya berhenti di EDA, tetapi menghasilkan fondasi eksperimen yang benar-benar siap dibandingkan dengan paper.

### Cakupan wajib
#### A. Dataset acquisition dan verifikasi
- [x] Download dataset Reddit dan Twitter dari HuggingFace
- [x] Verifikasi split train/validation/test
- [x] Verifikasi kolom teks dan label
- [x] Klarifikasi versi balanced vs unbalanced, terutama untuk Twitter

#### B. Exploratory data analysis
- [x] Label distribution
- [x] Split distribution
- [x] Text-length distribution
- [x] Missing value dan duplicate check
- [x] Dokumentasi figure dan insight utama

#### C. Baseline reproduction preparation
- [x] Menentukan target model: Logistic Regression, Naive Bayes, SVM
- [x] Menentukan feature extraction: BoW dan TF-IDF
- [x] Menentukan validasi: GridSearchCV + PredefinedSplit
- [x] Menentukan target metrik: F1, accuracy, precision, recall
- [x] Menyusun target angka acuan dari paper

#### D. Baseline execution
- [x] Menyiapkan script/notebook baseline classical ML
- [x] Menjalankan Twitter baseline: LR, NB, SVM
- [x] Menjalankan Reddit baseline: LR, NB, SVM
- [x] Menyimpan hasil ke tabel komparatif
- [x] Mencatat gap terhadap target paper

### Target hasil Progress 2
#### Target paper untuk Twitter (TF-IDF)
- Logistic Regression: ~0.7142
- Naive Bayes: ~0.6721
- SVM: ~0.6782

#### Target paper untuk Reddit (TF-IDF)
- Logistic Regression: ~0.4887
- Naive Bayes: ~0.4591
- SVM: ~0.4467

### Output yang diharapkan
- `data/raw/` berisi dataset kerja
- `notebooks/01_eda.ipynb`
- figure EDA di `results/figures/`
- script atau notebook baseline classical ML
- tabel hasil awal reproduksi di `results/tables/`
- dokumentasi lengkap Progress 2 di `docs/progress/progress-2.md`

### Gate kelulusan progress
Progress 2 baru dianggap benar-benar selesai jika:
1. baseline Twitter dan Reddit sudah dijalankan,
2. hasil utama sudah tersimpan dalam tabel,
3. selisih hasil terhadap paper sudah dicatat,
4. dokumentasi setup + hasil awal sudah lengkap.

---

## Progress 3: Reproduksi Transformer Baseline dan Benchmark Lanjutan ✅

### Tujuan
Menaikkan proyek dari level baseline classical ML ke level fine-tuned transformer sesuai tabel baseline paper IdSarcasm.

### Cakupan
- [x] Menentukan model transformer paper baseline: IndoBERT Base/Large, IndoBERT IndoLEM Base, mBERT, XLM-R Base, XLM-R Large
- [x] Mengambil titik awal dari `source-code/original-id-sarcasm/` agar implementasi tetap dekat ke codebase penulis
- [x] Menyamakan setting utama dengan recipe paper: epoch 100, batch 32/64, scheduler cosine, learning rate 1e-5, weight decay 0.03, max length 128, pad-to-max-length, shuffle train, early stopping threshold 0.01, seed 42, fp16
- [x] Menyiapkan pipeline fine-tuning yang rapi dan terdokumentasi (`scripts/run_transformer_baseline.py`)
- [x] Menjalankan smoke test di Colab
- [x] Menjalankan 12/12 baseline fine-tuned transformer: 6 model × 2 dataset
- [x] Menyimpan hasil metrik, konfigurasi, dan log Colab
- [x] Membandingkan hasil transformer dengan angka paper dan baseline classical ML terbaik

### Output
- Script/notebook fine-tuning transformer: `scripts/run_transformer_baseline.py` dan `notebooks/02_transformer_baseline_colab.ipynb`
- Panduan eksekusi Colab: `docs/progress/progress-3-local-run-guide.md`
- Plan complete baseline: `docs/progress/progress-3-paper-baseline-complete-plan.md`
- Tabel hasil baseline transformer: `results/tables/transformer_baselines.csv`
- Artifact per run: `results/transformer/*/metrics.json` dan `results/transformer/*/result_row.json`
- Log Colab: `results/logs/progress-3-*.log`

### Gate kelulusan progress
Progress 3 selesai karena semua 12 baseline transformer paper berhasil dijalankan pada Twitter dan Reddit, hasil tersimpan, dan gap terhadap paper tercatat.

---

## Progress 4: Zero-shot LLM Baseline ✅

### Tujuan
Menguji baseline zero-shot LLM sebagaimana kategori ketiga pada paper IdSarcasm. Progress ini dipisahkan dari Progress 3 karena workflow-nya inference/prompting, bukan fine-tuning transformer.

### Cakupan opsi
- [x] Mengambil prompt zero-shot dari source code asli paper
- [x] Membuat runner zero-shot yang mendukung HuggingFace/Colab dan LM Studio OpenAI-compatible API
- [x] Menambahkan daftar command paper-complete: 9 model × 2 dataset = 18 full runs
- [x] Menyimpan runtime total, latency rata-rata, log, metrics, result row, dan predictions per run
- [x] Membuat notebook Colab/LM Studio untuk eksekusi Progress 4
- [x] Membuat panduan run Progress 4
- [x] Menjalankan smoke test zero-shot pada subset kecil
- [x] Menjalankan semua model zero-shot paper pada Twitter (9/9 selesai)
- [x] Menjalankan Reddit sesuai resource Colab (5/9 selesai)
- [x] Menyimpan log untuk model Reddit yang tidak selesai karena runtime/session habis
- [x] Menyimpan prediksi, metrik, dan log inference untuk run yang selesai
- [x] Membandingkan hasil dengan target zero-shot paper

### Fokus evaluasi
- Akurasi parsing label dari output LLM
- Perbandingan F1 zero-shot vs classical ML vs fine-tuned transformer
- Keterbatasan resource lokal/Colab dan model yang tidak sama persis dengan paper

### Output
- `scripts/run_zeroshot_baseline.py`
- `notebooks/03_zeroshot_baseline_colab_or_lmstudio.ipynb`
- `docs/progress/progress-4-zero-shot-run-guide.md`
- `results/tables/zeroshot_baselines.csv`
- `results/tables/zeroshot_smoke.csv`
- `results/zeroshot/.../predictions.csv` dan `metrics.json`
- `results/logs/progress-4-zeroshot-*.log`
- Dokumentasi Progress 4 di `docs/progress/progress-4.md`

### Gate kelulusan progress
Progress 4 dianggap selesai sebagai **complete attempt** jika semua kombinasi zero-shot paper sudah dicoba satu per satu sesuai resource, run yang berhasil punya artefak hasil, run yang gagal punya log/error, prompt/parsing terdokumentasi, dan hasil dibandingkan dengan angka zero-shot paper. Status saat ini memenuhi gate tersebut: Twitter selesai 9/9, Reddit selesai 5/9, dan 4 run Reddit yang tidak selesai memiliki log runtime/session limitation.

---

## Progress 5: Optimasi dan Eksperimen Lanjutan ✅

### Tujuan
Melakukan optimasi yang benar-benar sesuai dengan framing proyek: bukan sekadar memakai transformer, tetapi mencoba meningkatkan performanya secara metodologis. Progress 5 difokuskan pada XLM-R sebagai model transformer terbaik Progress 3, lalu ditambah eksperimen LLM ringan modern sebagai pembanding praktis.

### Cakupan wajib
- [x] **Threshold tuning XLM-R Large**: Twitter F1 0,7226 → 0,7649; Reddit F1 0,6117 → 0,6241.
- [x] **Small hyperparameter experiment XLM-R Base → XLM-R Large**: lima konfigurasi XLM-R Base selesai; konfigurasi lr=2e-5 menjadi kandidat screening terbaik.
- [x] **Error analysis baseline vs optimized**: transisi prediksi dan contoh error tersimpan di `results/progress5_error_analysis/`.

### Eksperimen tambahan
- [x] **Gemma 4 E4B** zero-shot/few-shot sebagai model ringan modern.
- [x] **Qwen3.5-4B** zero-shot/few-shot sebagai model ringan terbaru.
- [ ] **Cendol atau Bahasa-4B** zero-shot/few-shot sebagai pembanding Indonesia-specific (opsional, tidak dijalankan pada Progress 5).
- [x] Untuk model GGUF/quantized, jalankan via LM Studio/OpenAI-compatible API di PC lokal.

### Fokus evaluasi
- [ ] Bandingkan baseline XLM-R Large vs optimized XLM-R Large.
- [ ] Catat trade-off precision, recall, F1, dan accuracy setelah threshold tuning.
- [ ] Catat konfigurasi hyperparameter yang paling realistis dijalankan di Colab.
- [ ] Bandingkan modern LLM lokal dengan zero-shot BLOOMZ/mT0 Progress 4, bukan sebagai klaim reproduksi exact paper.

### Output
- `scripts/run_transformer_optimization.py`
- `scripts/run_modern_llm_experiments.py`
- `notebooks/04_progress5_optimization_and_modern_llm.ipynb`
- `docs/progress/progress-5.md`
- `docs/progress/progress-5-run-guide.md`
- `results/tables/optimization_runs.csv`
- `results/tables/modern_llm_experiments.csv`
- `results/optimization/*/predictions.csv`
- `results/modern_llm/*/predictions.csv`
- Tabel before/after optimization
- Contoh error analysis

### Gate kelulusan progress
Progress 5 selesai: XLM-R Large threshold tuning, screening XLM-R Base, error analysis, dan eksperimen Qwen/Gemma local LLM sudah masuk laporan.

---

## Progress 6: Analisis Komparatif dan Finalisasi Laporan ✅

### Tujuan
Menyusun analisis matang terhadap seluruh rangkaian hasil dan menutup proyek dalam bentuk laporan/repo yang rapi, dapat diperiksa, dan siap dipresentasikan.

### Cakupan
- [x] Menyusun tabel komparatif semua eksperimen utama: classical ML, transformer baseline, zero-shot, dan optimasi/eksperimen lanjutan
- [x] Membuat confusion matrix untuk model terbaik
- [x] Menganalisis contoh benar/salah klasifikasi
- [x] Membandingkan karakteristik error antara classical ML, fine-tuned transformer, dan zero-shot LLM
- [x] Menjelaskan faktor penyebab gap hasil terhadap paper atau antar konfigurasi
- [x] Menentukan model/fase mana yang paling layak dijadikan highlight laporan
- [x] Finalisasi laporan proyek / laporan akhir
- [x] Rapikan README dan struktur repo
- [x] Rapikan tabel dan figure final
- [x] Tulis kesimpulan, keterbatasan, dan saran pengembangan lanjutan
- [x] Pastikan semua file penting ter-commit

### Output akhir
- Tabel komparatif final eksperimen
- Visual error analysis
- Narasi analitis untuk laporan hasil dan pembahasan
- Repo bersih dan terdokumentasi
- Laporan final
- Figure dan tabel final
- Ringkasan hasil reproduksi + improvement

### Gate kelulusan progress
Orang lain harus bisa membaca repo dan mengerti: apa yang direproduksi, bagaimana hasilnya, apa gap-nya, model mana yang terbaik, jenis error dominan, dan apa improvement yang berhasil/tidak berhasil.

---

## Hardware / Environment Plan

| Kegiatan | Lokasi kerja paling masuk akal | Alasan |
|----------|-------------------------------|--------|
| EDA | Lokal / VPS | ringan |
| Classical ML baseline | Lokal / Colab CPU | aman tanpa GPU besar |
| Transformer baseline | Google Colab GPU | sudah selesai 12/12 run; checkpoint besar tidak di-commit |
| Zero-shot LLM | LM Studio lokal / Colab / HuggingFace | LM Studio memungkinkan inference lokal dengan model quantized di VRAM 8GB; Colab/HF untuk model BLOOMZ/mT0 kecil |
| Optimasi/eksperimen | Colab GPU / lokal bila ringan | sesuaikan dengan model dan resource |
| Dokumentasi & analisis | VPS / lokal | fleksibel |

**Catatan:** VPS saat ini tidak ideal untuk eksekusi baseline karena env Python terkena isu kompatibilitas NumPy/X86_V2. Jadi dokumen ini mengasumsikan baseline akan dijalankan di lokal atau Colab.

## Timeline revisi (estimasi)

| Progress | Estimasi target |
|----------|-----------------|
| 1 | 12 Apr ✅ |
| 2 | 12 Apr - setelah baseline classical selesai |
| 3 | selesai: paper baseline complete fine-tuned transformer ✅ |
| 4 | zero-shot LLM baseline paper-complete (18 command siap; run Colab satu-satu berikutnya) |
| 5 | optimasi dan eksperimen lanjutan |
| 6 | analisis komparatif dan finalisasi laporan |

### Hasil akhir Progress 6
- Laporan akhir diperbarui di `docs/laporan-proyek.md`, termasuk hasil final Twitter F1 0,7905 yang melampaui paper 0,7692.
- Tabel final: `results/tables/final_method_comparison.csv` dan `results/tables/final_method_ranking.csv`.
- Figure final: `results/figures/final_*.png`, termasuk `results/figures/final_confusion_matrices.png`.
- Dashboard statis ditambahkan di `index.html` dengan data `dashboard/data/dashboard-data.json`; target GitHub Pages: `https://feb027.github.io/idsarcasm-reproduction/`.
- Dokumentasi tambahan: `docs/model-card.md`, `docs/error-analysis.md`, `CITATION.cff`, dan `.github/workflows/validate.yml`.
- Repository dirapikan: file review sementara dan artifact percobaan dipisahkan dari current tree; progress docs dipindahkan ke `docs/progress/`.
