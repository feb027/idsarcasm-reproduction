# Progress 4 — Zero-shot LLM Baseline

**Status:** 🔄 aset eksekusi siap, menunggu run Colab/lokal

**Tujuan utama:** mengevaluasi baseline zero-shot LLM dari paper IdSarcasm setelah baseline fine-tuned transformer selesai.

---

## 1. Latar Belakang

Paper IdSarcasm membandingkan tiga keluarga model: classical ML, fine-tuned transformer, dan zero-shot LLM. Progress 3 sudah menutup baseline fine-tuned transformer. Progress 4 dipisahkan karena zero-shot tidak melakukan fine-tuning, melainkan hanya melakukan inference dari prompt dan memilih label paling mungkin.

Progress 4 sekarang disiapkan dengan dua jalur eksekusi:

1. **HuggingFace/Colab (`hf-logprobs`)** — jalur paling dekat dengan paper. Runner menghitung log probability untuk dua label kandidat: `not sarcastic` dan `sarcastic`, lalu memilih label dengan skor lebih besar.
2. **LM Studio / OpenAI-compatible API (`openai-compatible`)** — jalur praktis untuk PC lokal dengan model quantized. Jalur ini bagus untuk eksperimen laporan, tetapi jika modelnya bukan BLOOMZ/mT0 maka hasilnya harus disebut sebagai **zero-shot local LLM baseline**, bukan reproduksi exact paper.

---

## 2. Jawaban Singkat: Bisa Lokal atau Harus Colab?

**Full paper-like zero-shot sebaiknya di Colab.** Alasannya:

- model BLOOMZ/mT0 tetap butuh RAM/VRAM cukup besar meskipun hanya inference;
- setiap dataset dievaluasi dengan 5 prompt, jadi jumlah inference = jumlah data test × 5;
- Reddit test berisi 2.824 data, berarti 14.120 forward pass per model;
- CPU lokal bisa jalan tetapi sangat lama;
- RX 6600 di Windows/WSL tidak semudah CUDA untuk PyTorch HuggingFace.

**Lokal tetap bisa untuk dua hal:**

1. **Smoke test kecil** (`--max-samples 8/20`) untuk memastikan pipeline dan format output benar.
2. **LM Studio backend** dengan model GGUF/quantized, misalnya model 3B/7B Q4. Ini lebih realistis di PC lokal 8GB VRAM, tetapi bukan reproduksi exact paper jika modelnya berbeda.

Rekomendasi saya untuk Progress 4:

- Jalankan **smoke test di Colab** dulu dengan `mt0-small` pada Twitter.
- Jika berhasil, jalankan full `mt0-small` untuk Twitter dan Reddit.
- Jika waktu Colab masih cukup, tambahkan `bloomz-560m` sebagai pembanding paper-family kedua.
- Jika ingin eksperimen lokal, pakai LM Studio untuk satu model instruction-tuned quantized dan tulis sebagai baseline tambahan, bukan target utama.

---

## 3. Prompt Zero-shot Paper

Runner memakai lima prompt dari source code asli paper (`source-code/original-id-sarcasm/scripts/run_zero_shot_classification.py`):

```text
{text} => Sarcasm:
Text: {text} => Sarcasm:
{text}
Is this text above sarcastic or not?
Is the following text sarcastic?
Text: {text}
Answer:
Text: {text}
Please classify the text above for sarcasm.
```

Label kandidat:

```text
not sarcastic
sarcastic
```

Untuk backend HuggingFace, runner tidak meminta model menghasilkan teks bebas. Runner menghitung log probability untuk dua label tersebut. Cara ini lebih stabil untuk reproduksi karena tidak tergantung variasi jawaban model seperti “yes”, “no”, “sarcastic”, atau kalimat panjang.

Untuk backend LM Studio/OpenAI-compatible, output model tetap berupa teks bebas, sehingga runner menyediakan parser transparan:

- output mengandung `not sarcastic`, `non-sarcastic`, `tidak sarkastik`, atau `bukan sarkastik` → label 0;
- output mengandung `sarcastic`, `sarkastik`, `sarcasm`, atau `sarkasme` → label 1;
- output ambigu → dicatat sebagai `invalid_output=True` dan dipetakan memakai fallback default `not_sarcastic`.

---

## 4. Target Paper Zero-shot

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

Interpretasi awal dari paper: zero-shot LLM jauh tertinggal dari fine-tuned transformer. Ini justru menarik untuk laporan, karena bisa menunjukkan bahwa model besar tidak otomatis memahami sarkasme bahasa Indonesia tanpa fine-tuning.

---

## 5. File yang Disiapkan

```text
scripts/run_zeroshot_baseline.py
notebooks/03_zeroshot_baseline_colab_or_lmstudio.ipynb
docs/progress-4.md
docs/progress-4-zero-shot-run-guide.md
results/tables/zeroshot_baselines.csv        # dibuat setelah full run
results/tables/zeroshot_smoke.csv            # dibuat otomatis untuk --max-samples
results/zeroshot/{dataset}-{backend}-{model}/metrics.json
results/zeroshot/{dataset}-{backend}-{model}/result_row.json
results/zeroshot/{dataset}-{backend}-{model}/predictions.csv
results/logs/progress-4-zeroshot-*.log
```

Setiap run menyimpan:

- metrik mean across 5 prompts (`accuracy`, `precision`, `recall`, `f1`);
- metrik per prompt;
- prediksi setiap data untuk setiap prompt;
- runtime total (`runtime_seconds`);
- latency rata-rata per inference (`avg_latency_seconds`);
- jumlah output invalid khusus backend generatif;
- log lengkap run.

---

## 6. Command Utama

### 6.1 Smoke test Colab/HuggingFace

```bash
python scripts/run_zeroshot_baseline.py --dataset twitter --model mt0-small --backend hf-logprobs --max-samples 8 --dtype float16 --device-map auto --disable-tqdm --write-log
```

Output smoke test otomatis masuk ke:

```text
results/tables/zeroshot_smoke.csv
results/zeroshot/twitter-hf-logprobs-mt0-small/
results/logs/progress-4-zeroshot-twitter-hf-logprobs-mt0-small-smoke.log
```

### 6.2 Full run minimal Progress 4

Minimal agar Progress 4 bisa dilaporkan:

```bash
python scripts/run_zeroshot_baseline.py --dataset twitter --model mt0-small --backend hf-logprobs --dtype float16 --device-map auto --disable-tqdm --write-log
python scripts/run_zeroshot_baseline.py --dataset reddit --model mt0-small --backend hf-logprobs --dtype float16 --device-map auto --disable-tqdm --write-log
```

Jika waktu Colab cukup, tambahkan BLOOMZ-560M:

```bash
python scripts/run_zeroshot_baseline.py --dataset twitter --model bloomz-560m --backend hf-logprobs --dtype float16 --device-map auto --disable-tqdm --write-log
python scripts/run_zeroshot_baseline.py --dataset reddit --model bloomz-560m --backend hf-logprobs --dtype float16 --device-map auto --disable-tqdm --write-log
```

### 6.3 LM Studio lokal

1. Buka LM Studio.
2. Load model quantized.
3. Start local server di `http://localhost:1234/v1`.
4. Jalankan:

```bash
python scripts/run_zeroshot_baseline.py --dataset twitter --model local-model --backend openai-compatible --api-base http://localhost:1234/v1 --max-samples 20 --disable-tqdm --write-log
```

Untuk full run lokal, hapus `--max-samples`, tetapi siap-siap durasinya lama.

---

## 7. Saran Implementasi agar Lancar dan Bagus untuk Laporan

Saya sarankan Progress 4 dibuat sebagai **baseline zero-shot yang jujur**, bukan dipaksakan seolah-olah harus mengalahkan transformer. Struktur pembahasan nanti bisa seperti ini:

1. Zero-shot dipakai untuk menguji kemampuan model tanpa fine-tuning.
2. Paper menunjukkan F1 zero-shot hanya sekitar 0,38–0,40.
3. Runner memakai lima prompt dan mengambil rata-rata metrik, mengikuti paper.
4. Hasil Progress 4 dibandingkan dengan:
   - classical ML terbaik,
   - fine-tuned transformer terbaik,
   - target zero-shot paper.
5. Jika hasil zero-shot rendah, itu bukan gagal. Justru sesuai temuan paper bahwa sarkasme bahasa Indonesia masih sulit untuk LLM tanpa adaptasi dataset.
6. Durasi inference dicatat agar laporan punya aspek komputasi: performa model vs waktu running.

Untuk laporan, metrik yang paling penting tetap F1-score karena kelas sarkastik hanya 25%.

---

## 8. Gate Kelulusan Progress 4

Progress 4 dianggap selesai jika:

1. minimal satu model zero-shot berhasil dievaluasi pada Twitter dan Reddit;
2. hasil tersimpan di `results/tables/zeroshot_baselines.csv`;
3. setiap run punya `metrics.json`, `result_row.json`, `predictions.csv`, dan log;
4. runtime total dan latency rata-rata tercatat;
5. prompt dan parsing label terdokumentasi;
6. hasil dibandingkan dengan angka zero-shot paper;
7. keterbatasan Colab/lokal dicatat.

---

## 9. Catatan Resource

Estimasi jumlah inference per model:

| Dataset | Test examples | Prompt count | Total prediction calls |
|---|---:|---:|---:|
| Twitter | 538 | 5 | 2.690 |
| Reddit | 2.824 | 5 | 14.120 |

Karena itu, Reddit full run jauh lebih lama daripada Twitter. Selalu mulai dari smoke test, lalu Twitter full, baru Reddit full.
