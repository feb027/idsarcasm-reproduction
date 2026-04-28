# Teks Google Spreadsheet — Submission Summary

Dokumen ini berisi teks siap copy-paste untuk kolom spreadsheet progres proyek.

## Kolom Progress

| Progress | Isi Kolom yang Disarankan |
|---|---|
| Progress 1 | Melakukan pencarian dan kajian paper terkait deteksi sarkasme dalam NLP, menentukan paper referensi utama yaitu IdSarcasm, mengumpulkan informasi dataset dari HuggingFace, menyusun latar belakang dan referensi awal, serta menyiapkan repositori proyek dan struktur dokumentasi. |
| Progress 2 | Download dan verifikasi dataset IdSarcasm, melakukan EDA, mengecek distribusi label/split/panjang teks, mengklarifikasi karakteristik dataset Twitter, lalu mereproduksi baseline classical ML (Logistic Regression, Naive Bayes, SVM) pada Twitter dan Reddit menggunakan BoW/TF-IDF. Hasil disimpan dalam tabel dan dibandingkan dengan paper. |
| Progress 3 | Mereproduksi baseline fine-tuned transformer sesuai paper IdSarcasm pada dataset Twitter dan Reddit. Model yang diuji meliputi IndoBERT, IndoBERT IndoLEM, mBERT, XLM-R Base, dan XLM-R Large. Hasil tiap run disimpan dalam tabel, log, dan artefak metrik untuk dibandingkan dengan baseline classical ML dan target paper. |
| Progress 4 | Mereproduksi baseline zero-shot LLM dari paper menggunakan model BLOOMZ dan mT0 dengan beberapa prompt. Evaluasi dilakukan pada Twitter dan sebagian Reddit sesuai keterbatasan waktu/runtime Colab. Hasil run yang selesai disimpan dalam tabel, prediction CSV, dan log; run yang tidak selesai dicatat sebagai keterbatasan komputasi. |
| Progress 5 | Melakukan optimasi model transformer terbaik melalui threshold tuning pada XLM-R Large, screening hyperparameter XLM-R Base, serta eksperimen pembanding dengan LLM lokal modern melalui LM Studio (Qwen3.5-4B dan Gemma 4 E4B) pada dataset Twitter. Hasil dianalisis melalui F1, precision, recall, dan transisi error. |
| Progress 6 | Menyusun analisis komparatif akhir seluruh eksperimen, menentukan model terbaik, membuat tabel/figure final, confusion matrix, error analysis, dashboard GitHub Pages, model card, reproducibility guide, README final, citation metadata, dan CI validation. Hasil akhir menunjukkan Twitter XLM-R Large lr=2e-5 melampaui paper, sedangkan Reddit mendekati paper. |

## Kolom Algoritma atau Pendekatan yang Digunakan

Versi pendek:

```text
Classical ML (Logistic Regression, Naive Bayes, SVM dengan BoW/TF-IDF), fine-tuned transformer (IndoBERT, mBERT, XLM-R), zero-shot LLM (BLOOMZ, mT0), modern local LLM via LM Studio (Qwen3.5-4B, Gemma 4 E4B), threshold tuning, dan learning-rate tuning.
```

Versi lebih rapi untuk spreadsheet:

```text
Pendekatan yang digunakan meliputi baseline classical machine learning (Logistic Regression, Naive Bayes, dan SVM dengan BoW/TF-IDF), fine-tuned transformer (IndoBERT, mBERT, XLM-R Base, XLM-R Large), zero-shot LLM sesuai paper (BLOOMZ dan mT0), eksperimen LLM lokal modern melalui LM Studio (Qwen3.5-4B dan Gemma 4 E4B), serta optimasi transformer melalui threshold tuning dan learning-rate tuning.
```

## Catatan Koreksi untuk Kolom Lama

Teks lama:

```text
Logistic Regression, Naive Bayes, SVM (TF-IDF), IndoBERT, XLM-R, Model Zero-Shot terbaru (Gemma E4B)
```

Sebaiknya diganti karena:

1. Classical ML tidak hanya TF-IDF, tetapi juga BoW.
2. Transformer tidak hanya IndoBERT dan XLM-R; ada juga mBERT dan IndoBERT IndoLEM.
3. Zero-shot sesuai paper memakai BLOOMZ dan mT0, bukan Gemma.
4. Gemma 4 E4B masuk kategori modern local LLM via LM Studio, bukan zero-shot paper utama.
5. Optimasi utama proyek adalah threshold tuning dan learning-rate tuning pada XLM-R Large.

## Link Pendukung

| Item | Link |
|---|---|
| Repository | <https://github.com/feb027/idsarcasm-reproduction> |
| Dashboard | <https://feb027.github.io/idsarcasm-reproduction/> |
| Release final | <https://github.com/feb027/idsarcasm-reproduction/releases/tag/v1.0-final-uas> |
| Laporan | [`docs/laporan-proyek.md`](laporan-proyek.md) |
| Error analysis | [`docs/error-analysis.md`](error-analysis.md) |
| Model card | [`docs/model-card.md`](model-card.md) |
