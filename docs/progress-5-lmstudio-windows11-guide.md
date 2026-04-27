# Progress 5 — Panduan LM Studio di Windows 11

Panduan ini untuk menjalankan eksperimen tambahan Progress 5: Gemma/Qwen/Cendol/Bahasa-4B zero-shot/few-shot lewat LM Studio di Windows 11.

Eksperimen ini **bukan optimasi utama transformer**. Ini pembanding modern local LLM terhadap zero-shot BLOOMZ/mT0 Progress 4.

---

## 1. Konsep singkat

Alurnya:

```text
LM Studio Windows 11
  ↓ local server OpenAI-compatible
http://localhost:1234/v1
  ↓ dipanggil dari WSL2 / PowerShell
scripts/run_modern_llm_experiments.py
  ↓ output
results/modern_llm/*
results/tables/modern_llm_experiments.csv
```

Rekomendasi lingkungan:

- **LM Studio:** jalan di Windows 11.
- **Script repo:** jalankan dari WSL2 Ubuntu kalau repo kamu sudah di WSL.
- Kalau WSL tidak bisa akses `localhost:1234`, pakai IP host Windows dari WSL.

---

## 2. Install dan setup LM Studio

1. Download LM Studio dari:

```text
https://lmstudio.ai/
```

2. Install seperti aplikasi Windows biasa.
3. Buka LM Studio.
4. Masuk ke tab **Discover/Search**.
5. Cari model GGUF ringan.

Prioritas model untuk Progress 5:

| Prioritas | Cari di LM Studio | Catatan |
|---|---|---|
| 1 | `Qwen3.5 4B GGUF` atau `Qwen 4B Instruct GGUF` | Kandidat paling praktis. Pilih Q4_K_M/Q4. |
| 2 | `Gemma 3n E4B GGUF` | Kalau belum muncul/unsupported di LM Studio, skip dulu. |
| 3 | `Bahasa 4B GGUF` atau model Indonesia 4B | Pembanding Indonesia-specific. |
| 4 | `Cendol GGUF` | Kalau tidak ada GGUF, skip. Cendol HF biasa tidak otomatis bisa jalan di LM Studio. |

Untuk RX 6600 8GB, pilih quantization:

```text
Q4_K_M / Q4_0 / Q4_K_S
```

Hindari dulu:

```text
Q8, F16, 8B+ besar
```

karena bisa berat di VRAM/RAM.

---

## 3. Load model dan start Local Server

1. Buka model yang sudah didownload di LM Studio.
2. Klik **Load Model**.
3. Masuk ke tab **Developer** atau **Local Server**.
4. Klik **Start Server**.
5. Pastikan server aktif di:

```text
http://localhost:1234/v1
```

6. Kalau ada opsi **Serve on Local Network**, aktifkan jika script dijalankan dari WSL dan `localhost` gagal.

---

## 4. Cek API dari Windows PowerShell

Buka PowerShell biasa, jalankan:

```powershell
Invoke-RestMethod http://localhost:1234/v1/models
```

Kalau berhasil, akan muncul daftar model. Catat `id` modelnya. Contoh:

```text
qwen3.5-4b-instruct-q4_k_m
```

Kalau gagal:

- pastikan server LM Studio sudah Start,
- pastikan model sudah loaded,
- jangan tutup LM Studio.

---

## 5. Cek akses dari WSL2

Masuk WSL Ubuntu, masuk repo:

```bash
cd ~/idsarcasm-reproduction
git pull
```

Tes localhost:

```bash
curl http://localhost:1234/v1/models
```

Kalau berhasil, pakai:

```bash
export API_BASE="http://localhost:1234/v1"
```

Kalau gagal, cari IP host Windows:

```bash
WIN_HOST=$(grep nameserver /etc/resolv.conf | awk '{print $2}')
echo $WIN_HOST
curl http://$WIN_HOST:1234/v1/models
```

Kalau ini berhasil, pakai:

```bash
export API_BASE="http://$WIN_HOST:1234/v1"
```

Kalau tetap gagal:

1. Di LM Studio aktifkan **Serve on Local Network**.
2. Izinkan firewall Windows jika muncul popup.
3. Ulangi `curl` dari WSL.

---

## 6. Setup Python di WSL

Kalau environment belum siap:

```bash
cd ~/idsarcasm-reproduction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kalau sudah pernah setup:

```bash
cd ~/idsarcasm-reproduction
source .venv/bin/activate
```

---

## 7. Ambil model id

Dari WSL, jalankan:

```bash
python3 - <<'PY'
import json, urllib.request, os
api_base = os.environ.get('API_BASE', 'http://localhost:1234/v1').rstrip('/')
with urllib.request.urlopen(api_base + '/models') as r:
    data = json.loads(r.read().decode())
print(json.dumps(data, indent=2))
PY
```

Ambil nilai `id`, lalu set:

```bash
export MODEL_ID="ISI_MODEL_ID_DARI_LM_STUDIO"
```

Contoh:

```bash
export MODEL_ID="qwen3.5-4b-instruct-q4_k_m"
```

Kalau bingung model id-nya, boleh coba:

```bash
export MODEL_ID="local-model"
```

Tapi lebih aman pakai `id` asli dari `/v1/models`.

---

## 8. Smoke test pertama

Jalankan 5 sampel dulu:

```bash
python scripts/run_modern_llm_experiments.py \
  --dataset twitter \
  --model "$MODEL_ID" \
  --model-alias lmstudio-smoke \
  --api-base "$API_BASE" \
  --max-samples 5 \
  --print-every 1
```

Output yang diharapkan:

```text
results/tables/modern_llm_smoke.csv
results/modern_llm/twitter-lmstudio-smoke-zeroshot-smoke/predictions.csv
results/modern_llm/twitter-lmstudio-smoke-zeroshot-smoke/metrics.json
```

Kalau smoke test sukses, lanjut full run.

---

## 9. Run zero-shot Twitter

Untuk Qwen:

```bash
python scripts/run_modern_llm_experiments.py \
  --dataset twitter \
  --model "$MODEL_ID" \
  --model-alias qwen3.5-4b-gguf \
  --api-base "$API_BASE" \
  --temperature 0.0 \
  --max-tokens 8 \
  --print-every 50
```

Untuk Gemma:

```bash
python scripts/run_modern_llm_experiments.py \
  --dataset twitter \
  --model "$MODEL_ID" \
  --model-alias gemma-3n-e4b-gguf \
  --api-base "$API_BASE" \
  --temperature 0.0 \
  --max-tokens 8 \
  --print-every 50
```

Untuk Bahasa/Cendol:

```bash
python scripts/run_modern_llm_experiments.py \
  --dataset twitter \
  --model "$MODEL_ID" \
  --model-alias bahasa-or-cendol-gguf \
  --api-base "$API_BASE" \
  --temperature 0.0 \
  --max-tokens 8 \
  --print-every 50
```

Catatan: jalankan satu model dulu sampai selesai. Jangan ganti model di LM Studio saat script sedang jalan.

---

## 10. Run few-shot Twitter

Few-shot default: 2 contoh sarkastik + 2 contoh non-sarkastik dari train split.

Untuk Qwen:

```bash
python scripts/run_modern_llm_experiments.py \
  --dataset twitter \
  --model "$MODEL_ID" \
  --model-alias qwen3.5-4b-gguf \
  --api-base "$API_BASE" \
  --few-shot \
  --shots-per-class 2 \
  --temperature 0.0 \
  --max-tokens 8 \
  --print-every 50
```

Untuk Gemma:

```bash
python scripts/run_modern_llm_experiments.py \
  --dataset twitter \
  --model "$MODEL_ID" \
  --model-alias gemma-3n-e4b-gguf \
  --api-base "$API_BASE" \
  --few-shot \
  --shots-per-class 2 \
  --temperature 0.0 \
  --max-tokens 8 \
  --print-every 50
```

Few-shot biasanya lebih relevan untuk model chat/instruct karena model diberi contoh format jawaban.

---

## 11. Reddit hanya setelah Twitter selesai

Reddit lebih lama. Jalankan hanya untuk model terbaik dari Twitter.

```bash
python scripts/run_modern_llm_experiments.py \
  --dataset reddit \
  --model "$MODEL_ID" \
  --model-alias qwen3.5-4b-gguf \
  --api-base "$API_BASE" \
  --few-shot \
  --shots-per-class 2 \
  --temperature 0.0 \
  --max-tokens 8 \
  --print-every 100
```

---

## 12. Cek hasil

```bash
python3 - <<'PY'
import csv
from pathlib import Path
for p in [Path('results/tables/modern_llm_smoke.csv'), Path('results/tables/modern_llm_experiments.csv')]:
    print('\n##', p)
    if not p.exists():
        print('missing')
        continue
    rows = list(csv.DictReader(p.open(encoding='utf-8')))
    for r in rows:
        print(r.get('dataset'), r.get('mode'), r.get('model_alias'), 'F1=', r.get('f1'), 'invalid=', r.get('invalid_outputs'), 'n=', r.get('num_examples'))
PY
```

Yang harus diperhatikan:

- `f1`
- `precision`
- `recall`
- `invalid_outputs`
- `runtime_seconds`

Kalau `invalid_outputs` tinggi, model sering tidak menjawab label yang diminta.

---

## 13. Kalau output banyak invalid

Coba ulang dengan instruksi lebih ketat:

```bash
python scripts/run_modern_llm_experiments.py \
  --dataset twitter \
  --model "$MODEL_ID" \
  --model-alias qwen3.5-4b-gguf-strict \
  --api-base "$API_BASE" \
  --few-shot \
  --shots-per-class 2 \
  --temperature 0.0 \
  --max-tokens 12 \
  --system-prompt "You are a strict binary classifier. Answer exactly one label only: sarcastic or not sarcastic. Do not explain." \
  --print-every 50
```

---

## 14. Commit hasil

Setelah minimal satu zero-shot/few-shot full Twitter selesai:

```bash
git status --short
git add results/tables/modern_llm_experiments.csv results/modern_llm
git commit -m "results: add Progress 5 modern local LLM experiments"
git push
```

Kalau hanya smoke test, jangan commit dulu kecuali ingin menyimpan bukti uji awal.

---

## 15. Kapan balik ke agent

Balik ke agent setelah salah satu kondisi ini:

1. Minimal satu full Twitter zero-shot/few-shot selesai.
2. Kamu bingung karena `curl /v1/models` gagal dari WSL.
3. `invalid_outputs` tinggi.
4. Model terlalu lambat atau crash.

Kirim info ini:

```text
Model yang dipakai:
API_BASE yang jalan:
Mode: zero-shot / few-shot
File hasil: results/tables/modern_llm_experiments.csv
Masalah/error jika ada:
```
