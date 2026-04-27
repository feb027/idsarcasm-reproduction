# Plan Progress 5 — Optimasi dan Eksperimen Lanjutan IdSarcasm

Tanggal inspeksi: 2026-04-27 UTC
Repo: `/home/aqua/idsarcasm-reproduction`
Wiki: `/home/aqua/.hermes/wiki/projects/nlp-uas-idsarcasm.md`

## 1. Temuan konteks saat ini

### Dari wiki
- Progress 1–4 dicatat selesai/complete attempt.
- Progress 5 diarahkan ke optimasi dan eksperimen lanjutan.
- Judul proyek resmi tetap: **Optimasi Performa Model Transformer dalam Klasifikasi Sarkasme Teks Berbahasa Indonesia Berdasarkan Benchmark IdSarcasm**.
- Baseline terbaik transformer:
  - Reddit: XLM-R Large F1 = 0.6117, paper = 0.6274.
  - Twitter: XLM-R Large F1 = 0.7226, paper = 0.7692.
- Zero-shot paper/Progress 4 dicatat rendah, sekitar F1 0.38–0.40, dan sering over-predict sarcasm.

### Dari repo lokal setelah pull ulang
- `results/tables/transformer_baselines.csv` ada dan berisi 12 run full.
- `results/tables/classical_baselines_twitter.csv` dan `classical_baselines_reddit.csv` ada.
- `results/tables/zeroshot_baselines.csv` ada dan berisi 14 full run: Twitter 9/9, Reddit 5/9.
- `docs/laporan-proyek.md` sudah mengisi Progress 4: section 2.2.3, 3.2.3, 3.3.3, dan 3.3.4 analisis komparatif sementara.
- Figure zero-shot sudah ada: `zeroshot_pipeline_architecture.png`, `zeroshot_run_completion_matrix.png`, `zeroshot_f1_vs_paper.png`, `zeroshot_metrics_profile.png`, `zeroshot_runtime_minutes.png`.
- `scripts/run_transformer_baseline.py` belum menyimpan `predictions.csv`/logits untuk test set, jadi error analysis dan threshold tuning transformer tetap perlu ditambahkan pada Progress 5.

## 2. Jawaban arah umum

Progress 5 sebaiknya jangan hanya “coba model baru”. Agar sesuai judul **optimasi performa model transformer**, Progress 5 perlu dibagi menjadi dua jalur:

1. **Optimasi utama**: meningkatkan baseline fine-tuned transformer secara terukur.
2. **Eksperimen lanjutan**: mencoba model ringan terbaru/local/Colab sebagai pembanding modern, terutama zero-shot/few-shot local LLM.

Dengan begitu laporan tetap kuat: ada optimasi metodologis, ada eksperimen model terbaru, dan hasilnya bisa dibandingkan dengan baseline Progress 2–4.

## 3. Model terbaru/ringan kandidat dari pencarian Exa awal

Hasil pencarian Exa awal menunjukkan kandidat berikut:

| Kandidat | Status/konteks | Ukuran | Relevansi |
|---|---:|---:|---|
| `google/gemma-3n-E4B` | HuggingFace, Gemma 3n E4B; dirancang low-resource, effective ~4B walau raw ~8B; Transformers >= 4.53 | effective 4B | Kandidat utama untuk local/Colab zero-shot/few-shot. |
| Gemma 4 E4B / E2B | Google model card bertanggal Apr 2026 dari Exa; open-weight, multilingual, on-device | E2B/E4B | Sangat menarik, tapi perlu verifikasi ketersediaan HF/runtime sebelum dipakai. |
| `Qwen/Qwen3.5-4B` | HuggingFace, Exa menemukan release Mar 2026 | 4B | Kandidat utama ringan terbaru; perlu uji kompatibilitas Transformers/vLLM dan lisensi. |
| `indonlp/cendol` | Koleksi Indonesian LLM; paper 2024, bukan model 2026 terbaru | 300M–13B | Bagus untuk pembanding bahasa Indonesia; pilih varian mT5-base/large/xl atau LLaMA2-7B jika resource cukup. |
| `Bahasalab/Bahasa-4b` | Continued training dari Qwen-4B pada data Indonesia | 4B | Relevan untuk bahasa Indonesia; bukan terbaru 2026 tapi cocok sebagai baseline lokal Indonesia. |
| `aisingapore/Apertus-SEA-LION-v4-8B-IT` | SEA-LION v4, updated Feb 2026, mencakup Indonesian | 8B | Kandidat SEA terbaru, tapi lebih berat; realistis di Colab GPU kuat/quantized lokal. |

Catatan: Gemma/Qwen/Cendol sebaiknya diposisikan sebagai **eksperimen lanjutan zero-shot/few-shot**, bukan “optimasi transformer fine-tuning”, kecuali nanti benar-benar fine-tune/LoRA model tersebut.

## 4. Rekomendasi desain Progress 5

### Jalur A — Optimasi utama: XLM-R Large / XLM-R Base

Fokus model:
- **Twitter** dulu, karena gap XLM-R Large ke paper masih -0.0466 dan baseline classical hampir menyamai transformer. Ini menarik untuk dibahas.
- Setelah ada hasil bagus, ulang 1–2 konfigurasi terbaik di Reddit.

Optimasi yang paling masuk akal:

1. **Prediction artifact + threshold tuning**
   - Tambahkan output `predictions.csv` berisi teks, label asli, prediksi, probabilitas/logit kelas sarkastik.
   - Cari threshold terbaik di validation set untuk F1, lalu apply ke test set.
   - Alasan: XLM-R Large Twitter punya precision 0.6364 dan recall 0.8358. Model terlalu agresif memprediksi sarkasme, jadi threshold >0.5 berpotensi menaikkan F1 tanpa retraining mahal.

2. **Small hyperparameter grid**
   - Learning rate: `5e-6`, `1e-5`, `2e-5`.
   - Max length: `128`, `256`.
   - Weight decay: `0.01`, `0.03`.
   - Label smoothing: `0.0`, `0.05`.
   - Jangan full grid besar. Pakai tahap seleksi:
     - XLM-R Base untuk screening murah.
     - XLM-R Large hanya untuk 2–3 konfigurasi final.

3. **Loss/imbalance experiment**
   - Coba weighted cross-entropy atau focal loss sebagai satu varian.
   - Tapi jangan jadikan prioritas utama, karena dataset 25:75 dan baseline Twitter sudah recall tinggi. Weighted loss bisa makin menaikkan recall tapi menurunkan precision.
   - Lebih aman: threshold tuning dulu.

4. **Seed robustness mini-check**
   - Jalankan config terbaik pada seed `42`, `123`, `3407` jika Colab memungkinkan.
   - Kalau resource terbatas, jalankan hanya Twitter.
   - Tujuan: jangan klaim improvement dari satu run saja kalau selisihnya kecil.

Output minimal jalur A:
- `scripts/run_transformer_baseline.py` diperluas supaya bisa save predictions/logits dan threshold evaluation.
- `results/tables/optimization_runs.csv`.
- `results/optimization/<run-id>/metrics.json`.
- `results/optimization/<run-id>/predictions.csv`.
- Figure before/after F1 dan precision-recall tradeoff.

### Jalur B — Eksperimen model ringan terbaru/local/Colab

Tujuan: bukan mengalahkan XLM-R Large secara wajib, tetapi menguji apakah LLM ringan terbaru lebih baik dari zero-shot BLOOMZ/mT0 paper.

Prioritas model:
1. `google/gemma-3n-E4B` — kandidat utama karena user memang menyebut Gemma E4B dan model ini dirancang efisien.
2. `Qwen/Qwen3.5-4B` — kandidat utama kedua jika benar tersedia dan kompatibel.
3. `indonlp/cendol` varian mT5-base/large/xl — pembanding Indonesia-specific.
4. `Bahasalab/Bahasa-4b` — pembanding Indonesia 4B.
5. `SEA-LION-v4-8B-IT` — opsional jika resource kuat.

Mode eksperimen:
- **Zero-shot**: prompt sama/serupa Progress 4, output diparse ke label.
- **Few-shot 2–4 contoh**: opsional, satu set contoh tetap dari train split, tidak boleh ambil dari test.
- **Local API/LM Studio** untuk model quantized GGUF.
- **HF Transformers** di Colab untuk model yang mudah load langsung.

Output minimal jalur B:
- Tambah alias model modern di runner zero-shot atau buat runner baru `scripts/run_modern_llm_experiments.py`.
- `results/tables/modern_llm_experiments.csv`.
- `results/modern_llm/<dataset>-<model>-<prompt-mode>/predictions.csv`.
- Log runtime/latency supaya pembahasan punya aspek performa vs biaya komputasi.

## 5. Urutan kerja yang disarankan

### Phase 0 — Checkpoint Progress 4
1. Progress 4 sudah lengkap di repo setelah pull ulang:
   - `results/tables/zeroshot_baselines.csv` berisi 14 full run.
   - Twitter selesai 9/9.
   - Reddit selesai 5/9; 4 run Reddit tercatat sebagai runtime/session limitation.
   - `docs/laporan-proyek.md` sudah punya pembahasan Progress 4 dan analisis komparatif sementara.
2. Sebelum eksekusi Progress 5, cukup pastikan working tree bersih dari file plan sementara atau putuskan apakah `.hermes/plans/` mau dikomit/tidak.

### Phase 1 — Riset model final
1. Exa search + crawl halaman resmi/HuggingFace untuk:
   - Gemma 3n E4B / Gemma 4 E4B
   - Qwen3.5-4B
   - Cendol
   - Bahasa-4B
   - SEA-LION v4 8B
2. Catat:
   - tanggal rilis/update,
   - ukuran parameter,
   - lisensi,
   - dukungan Transformers/LM Studio/GGUF,
   - kebutuhan VRAM/RAM,
   - apakah mendukung Indonesian.
3. Pilih maksimal 3 model supaya scope tidak melebar.

### Phase 2 — Siapkan artifact untuk optimasi
1. Patch transformer runner untuk menyimpan predictions/logits.
2. Tambah utility threshold tuning.
3. Tambah tabel `optimization_runs.csv` yang memuat config dan metrik.
4. Tambah notebook `04_optimization_and_modern_llm_experiments_colab.ipynb` dengan cell satu-per-run.

### Phase 3 — Jalankan optimasi murah dulu
1. Run XLM-R Base screening di Twitter:
   - baseline re-run + predictions.
   - threshold tuning.
   - 2–4 konfigurasi hyperparameter ringan.
2. Pilih 2 config terbaik.
3. Jalankan XLM-R Large hanya untuk baseline+threshold dan 1–2 config terbaik.

### Phase 4 — Jalankan eksperimen modern LLM
1. Smoke test 10–20 sample untuk setiap kandidat.
2. Full run Twitter dulu.
3. Reddit hanya untuk 1–2 model terbaik atau jika runtime masih masuk akal.
4. Simpan predictions, metrics, runtime, dan invalid output count.

### Phase 5 — Analisis dan laporan
1. Tabel before/after:
   - best classical,
   - best transformer baseline,
   - optimized transformer,
   - zero-shot paper/Progress 4,
   - modern lightweight LLM.
2. Figure:
   - before/after optimization F1.
   - precision-recall shift akibat threshold tuning.
   - modern LLM vs zero-shot paper vs transformer.
   - confusion matrix model terbaik.
3. Error analysis:
   - ambil contoh false positive dan false negative dari model baseline vs optimized.
   - jelaskan pola: teks pendek, konteks implisit, slang, humor, negasi, atau ekspresi positif yang sebenarnya menyindir.
4. Update laporan:
   - `2.2.4 Optimasi dan Eksperimen Lanjutan`
   - `3.2.4 Tahapan Optimasi dan Eksperimen Lanjutan`
   - `3.3.5 Hasil Optimasi dan Eksperimen Lanjutan`
   - perluasan `3.3.4 Analisis Komparatif` jika sudah ada.

## 6. File yang kemungkinan berubah

- `scripts/run_transformer_baseline.py`
- `scripts/generate_optimization_figures.py`
- `scripts/run_modern_llm_experiments.py` atau patch `scripts/run_zeroshot_baseline.py`
- `notebooks/04_optimization_and_modern_llm_experiments_colab.ipynb`
- `results/tables/optimization_runs.csv`
- `results/tables/modern_llm_experiments.csv`
- `results/figures/optimization_*.png`
- `results/figures/modern_llm_*.png`
- `docs/progress-5.md`
- `docs/laporan-proyek.md`
- `README.md`
- wiki page `~/.hermes/wiki/projects/nlp-uas-idsarcasm.md`

## 7. Risiko dan batasan

- Model generatif 4B/8B tidak otomatis lebih bagus untuk klasifikasi sarkasme dibanding fine-tuned encoder.
- Gemma/Qwen/Cendol zero-shot harus ditulis sebagai eksperimen tambahan, bukan reproduksi exact paper.
- Threshold tuning bisa menaikkan F1, tapi harus dipilih dari validation set, bukan test set.
- Improvement kecil pada satu seed belum cukup kuat; butuh minimal seed check atau narasi keterbatasan.
- Colab runtime bisa putus untuk model besar, jadi semua run harus menyimpan log dan table incrementally.

## 8. Keputusan plan

Rekomendasi final: **Progress 5 = XLM-R optimization + modern lightweight LLM comparison**.

Urutan prioritas:
1. Bereskan/sinkronkan Progress 4 artifact dulu.
2. Optimasi XLM-R dengan prediction artifact + threshold tuning.
3. Small hyperparameter grid terbatas.
4. Coba 2–3 model modern ringan: Gemma 3n E4B, Qwen3.5-4B, dan Cendol/Bahasa-4B.
5. Tulis laporan dengan framing: fine-tuning transformer tetap paling kuat; model ringan modern menjadi pembanding praktis/local, bukan klaim mengganti baseline utama.
