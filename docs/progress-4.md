# Progress 4 — Zero-shot LLM Baseline

**Status:** ⬜ direncanakan  
**Tujuan utama:** mereproduksi/mengevaluasi baseline zero-shot LLM dari paper IdSarcasm setelah baseline fine-tuned transformer selesai.

---

## Latar Belakang

Paper IdSarcasm tidak hanya membandingkan classical ML dan fine-tuned transformer, tetapi juga zero-shot LLM. Karena workflow zero-shot berbeda dari fine-tuning transformer, bagian ini dipindahkan menjadi **Progress 4**.

Progress 4 dapat dijalankan dengan dua jalur:

1. **Colab / Python HuggingFace** untuk model kecil-menengah seperti BLOOMZ-560M atau mT0-small.
2. **LM Studio lokal** untuk model instruction-tuned/quantized yang muat di VRAM 8GB.

---

## Apakah LM Studio Bisa Dipakai?

Bisa, terutama untuk model quantized 4-bit/5-bit yang muat di VRAM 8GB. LM Studio menyediakan OpenAI-compatible local server, sehingga runner Python dapat mengirim prompt ke endpoint lokal seperti:

```text
http://localhost:1234/v1/chat/completions
```

Catatan penting:

- Jika memakai model yang tidak sama dengan paper, hasilnya bukan reproduksi exact, melainkan **zero-shot local LLM baseline**.
- Untuk mendekati paper, model yang perlu dicoba adalah BLOOMZ dan mT0. Namun model-model ini belum tentu praktis di LM Studio tergantung ketersediaan GGUF/quantized model.
- Dengan VRAM 8GB, model 7B quantized biasanya bisa untuk inference pendek, tetapi full test set Reddit/Twitter tetap bisa memakan waktu.

---

## Target Paper Zero-shot

| Model | Reddit F1 | Twitter F1 |
|---|---:|---:|
| BLOOMZ-560M | 0.3870 | 0.3916 |
| BLOOMZ-1.1B | 0.3944 | 0.3987 |
| BLOOMZ-1.7B | 0.3758 | 0.3885 |
| BLOOMZ-3B | 0.4000 | 0.3847 |
| BLOOMZ-7.1B | 0.4036 | 0.3968 |
| mT0 Small | 0.4000 | 0.3988 |
| mT0 Base | 0.3990 | 0.3985 |
| mT0 Large | 0.3998 | 0.3989 |
| mT0 XL | 0.4001 | 0.3988 |

---

## Scope Progress 4

- [ ] Mengambil prompt zero-shot dari Appendix paper atau membuat prompt yang setara jika extraction paper tidak bersih.
- [ ] Membuat runner zero-shot: `scripts/run_zeroshot_baseline.py`.
- [ ] Mendukung dua backend:
  - HuggingFace local/Colab inference.
  - LM Studio OpenAI-compatible API.
- [ ] Menjalankan smoke test pada subset kecil.
- [ ] Menjalankan zero-shot Twitter dan Reddit untuk minimal satu model kecil.
- [ ] Jika memungkinkan, menjalankan beberapa model paper atau local LLM alternatif.
- [ ] Menyimpan prediksi, metrik, dan log.
- [ ] Membandingkan hasil dengan zero-shot paper.

---

## Rencana Output

```text
scripts/run_zeroshot_baseline.py
notebooks/03_zeroshot_baseline_colab_or_lmstudio.ipynb
results/tables/zeroshot_baselines.csv
results/zeroshot/{dataset}-{model}/metrics.json
results/zeroshot/{dataset}-{model}/predictions.csv
results/logs/progress-4-zeroshot-*.log
```

---

## Skema Prompt Awal

Contoh prompt awal untuk klasifikasi biner:

```text
Tentukan apakah teks berikut mengandung sarkasme.
Jawab hanya dengan salah satu label: sarcastic atau not sarcastic.

Teks: {text}
Jawaban:
```

Parsing output:

- `sarcastic` → label 1
- `not sarcastic` → label 0
- output ambigu → dicatat sebagai invalid dan dipetakan dengan aturan fallback yang transparan

---

## Gate Kelulusan Progress 4

Progress 4 dianggap selesai jika:

1. minimal satu model zero-shot berhasil dievaluasi pada Twitter dan Reddit,
2. hasil tersimpan dalam tabel dan file prediksi,
3. prompt dan parsing label terdokumentasi,
4. hasil dibandingkan dengan angka zero-shot paper,
5. keterbatasan LM Studio/Colab dicatat.

---

## Catatan untuk VRAM 8GB

Rekomendasi model lokal via LM Studio:

- mulai dari model 3B atau 7B quantized Q4/Q5,
- batasi `max_tokens` rendah karena output hanya label,
- gunakan temperature 0 untuk deterministic-ish inference,
- jalankan subset dulu sebelum full test set,
- simpan output secara incremental agar aman jika proses terputus.

Progress 4 bukan lagi fine-tuning, jadi beban GPU lebih kecil dari training, tetapi full inference ribuan sampel tetap bisa lama.
