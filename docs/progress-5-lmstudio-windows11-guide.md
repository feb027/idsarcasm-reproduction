# Progress 5 — LM Studio Windows 11 Guide

Panduan ini **Windows-first**. Jalankan LM Studio dan script dari Windows PowerShell, bukan WSL. WSL hanya opsional kalau memang API LM Studio bisa diakses dari WSL.

Tujuan eksperimen ini: menjalankan modern local LLM zero-shot/few-shot untuk pembanding Progress 5. Optimasi utama transformer sudah ada di `results/tables/optimization_runs.csv`; bagian LM Studio ini adalah eksperimen tambahan.

---

## 1. Setup LM Studio

1. Install LM Studio:

```text
https://lmstudio.ai/
```

2. Buka LM Studio.
3. Download model GGUF ringan.

Rekomendasi awal:

| Prioritas | Model yang dicari di LM Studio | Catatan |
|---|---|---|
| 1 | Qwen 4B / Qwen3.5 4B Instruct GGUF | Paling praktis untuk smoke/full run. |
| 2 | Gemma 3n E4B GGUF | Pakai kalau tersedia dan bisa load. |
| 3 | Bahasa-4B GGUF | Pembanding Indonesia-specific. |
| 4 | Cendol GGUF | Pakai hanya kalau ada GGUF yang bisa diload LM Studio. |

Untuk RX 6600 8GB, pilih quantization ringan:

```text
Q4_K_M / Q4_0 / Q4_K_S
```

Hindari dulu:

```text
F16 / Q8 / model 8B+ besar
```

---

## 2. Start local server LM Studio

Di LM Studio:

1. Load model.
2. Buka tab **Developer** / **Local Server**.
3. Klik **Start Server**.
4. Pastikan URL:

```text
http://localhost:1234/v1
```

Jangan tutup LM Studio selama eksperimen berjalan.

---

## 3. Test API dari Windows PowerShell

Buka **PowerShell** biasa, jalankan:

```powershell
Invoke-RestMethod http://localhost:1234/v1/models
```

Kalau berhasil, output berisi model `id`. Contoh:

```text
qwen3.5-4b
```

Set variable di PowerShell:

```powershell
$env:API_BASE = "http://localhost:1234/v1"
$env:MODEL_ID = "qwen3.5-4b"
```

Ganti `qwen3.5-4b` sesuai `id` yang muncul dari LM Studio.

Kalau `Invoke-RestMethod` gagal, berarti server LM Studio belum benar-benar aktif atau model belum loaded.

---

## 4. Setup repo di Windows PowerShell

Kalau repo sudah ada di drive Windows seperti contoh `G:\semester 6\idsarcasm-reproduction`, masuk ke folder itu:

```powershell
cd "G:\semester 6\idsarcasm-reproduction"
git pull
```

Kalau belum ada repo Windows-native:

```powershell
cd $env:USERPROFILE
git clone https://github.com/feb027/idsarcasm-reproduction.git
cd idsarcasm-reproduction
```

Buat/aktifkan venv Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kalau PowerShell memblokir activate:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Kalau venv sudah ada:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 5. Penting: pull update parser dulu

Parser sudah diperbarui agar menerima jawaban Indonesia seperti `sarkastis`, `tidak sarkastis`, dan `bukan sarkastis`. Jadi sebelum rerun smoke/full:

```powershell
git pull
```

Kalau smoke sebelumnya menghasilkan `invalid_outputs: 5`, rerun setelah pull. Penyebab umum di Qwen/Gemma lokal: model mengeluarkan token reasoning `<think>` dulu, sehingga `--max-tokens` kecil hanya menangkap bagian berpikir dan belum sampai label final.

---

## 6. Smoke test zero-shot

Jalankan 5 sampel dulu:

```powershell
python scripts/run_modern_llm_experiments.py `
  --dataset twitter `
  --model $env:MODEL_ID `
  --model-alias lmstudio-smoke `
  --api-base $env:API_BASE `
  --max-samples 5 `
  --print-every 1 `
  --print-invalid-outputs
```

Output masuk ke:

```text
results/tables/modern_llm_smoke.csv
results/modern_llm/twitter-lmstudio-smoke-zeroshot-smoke/
```

Cek hasil smoke:

```powershell
Import-Csv results\tables\modern_llm_smoke.csv | Format-Table dataset,mode,model_alias,f1,invalid_outputs,num_examples
```

Kalau `invalid_outputs` masih tinggi, lanjut ke smoke strict di bawah.

---

## 7. Smoke test strict few-shot

Untuk model chat seperti Qwen, few-shot + strict prompt biasanya lebih stabil:

```powershell
python scripts/run_modern_llm_experiments.py `
  --dataset twitter `
  --model $env:MODEL_ID `
  --model-alias qwen3.5-4b-strict-smoke `
  --api-base $env:API_BASE `
  --few-shot `
  --shots-per-class 2 `
  --temperature 0.0 `
  --max-tokens 128 `
  --system-prompt "You are a strict binary classifier. Answer exactly one label only: sarcastic or not sarcastic. Do not explain." `
  --max-samples 10 `
  --print-every 1 `
  --print-invalid-outputs
```

Kalau ini sukses dan `invalid_outputs` kecil/0, pakai pola strict few-shot untuk full run.

---

## 8. Full run Twitter zero-shot

Untuk Qwen:

```powershell
python scripts/run_modern_llm_experiments.py `
  --dataset twitter `
  --model $env:MODEL_ID `
  --model-alias qwen3.5-4b-gguf `
  --api-base $env:API_BASE `
  --temperature 0.0 `
  --max-tokens 128 `
  --system-prompt "You are a strict binary classifier. Answer exactly one label only: sarcastic or not sarcastic. Do not explain." `
  --print-every 50
```

Untuk Gemma, ganti alias:

```powershell
python scripts/run_modern_llm_experiments.py `
  --dataset twitter `
  --model $env:MODEL_ID `
  --model-alias gemma-3n-e4b-gguf `
  --api-base $env:API_BASE `
  --temperature 0.0 `
  --max-tokens 128 `
  --system-prompt "You are a strict binary classifier. Answer exactly one label only: sarcastic or not sarcastic. Do not explain." `
  --print-every 50
```

Untuk Bahasa/Cendol, ganti alias:

```powershell
python scripts/run_modern_llm_experiments.py `
  --dataset twitter `
  --model $env:MODEL_ID `
  --model-alias bahasa-or-cendol-gguf `
  --api-base $env:API_BASE `
  --temperature 0.0 `
  --max-tokens 128 `
  --system-prompt "You are a strict binary classifier. Answer exactly one label only: sarcastic or not sarcastic. Do not explain." `
  --print-every 50
```

---

## 9. Full run Twitter few-shot — direkomendasikan

Ini command yang paling direkomendasikan untuk Qwen/Gemma/Bahasa/Cendol:

```powershell
python scripts/run_modern_llm_experiments.py `
  --dataset twitter `
  --model $env:MODEL_ID `
  --model-alias qwen3.5-4b-gguf `
  --api-base $env:API_BASE `
  --few-shot `
  --shots-per-class 2 `
  --temperature 0.0 `
  --max-tokens 128 `
  --system-prompt "You are a strict binary classifier. Answer exactly one label only: sarcastic or not sarcastic. Do not explain." `
  --print-every 50
```

Kalau ganti model, cukup ganti:

```powershell
$env:MODEL_ID = "ID_MODEL_BARU_DARI_LM_STUDIO"
```

lalu ganti `--model-alias`, misalnya:

```text
gemma-3n-e4b-gguf
bahasa-4b-gguf
cendol-gguf
```

---

## 10. Reddit hanya setelah Twitter selesai

Reddit lebih lama. Jalankan hanya untuk model/mode terbaik dari Twitter:

```powershell
python scripts/run_modern_llm_experiments.py `
  --dataset reddit `
  --model $env:MODEL_ID `
  --model-alias qwen3.5-4b-gguf `
  --api-base $env:API_BASE `
  --few-shot `
  --shots-per-class 2 `
  --temperature 0.0 `
  --max-tokens 128 `
  --system-prompt "You are a strict binary classifier. Answer exactly one label only: sarcastic or not sarcastic. Do not explain." `
  --print-every 100
```

---

## 11. Cek hasil full run

```powershell
Import-Csv results\tables\modern_llm_experiments.csv |
  Format-Table dataset,mode,model_alias,f1,precision,recall,invalid_outputs,num_examples,runtime_seconds
```

Cek contoh output mentah kalau invalid masih tinggi:

```powershell
Import-Csv results\modern_llm\twitter-qwen3.5-4b-gguf-fewshot-full\predictions.csv |
  Select-Object -First 10 sample_idx,true_label,pred_label,raw_output,invalid_output |
  Format-List
```

Path folder bisa beda sesuai alias/mode. Lihat folder:

```powershell
Get-ChildItem results\modern_llm
```

---

## 12. Commit hasil

Setelah minimal satu full Twitter zero-shot/few-shot selesai:

```powershell
git status
git add results\tables\modern_llm_experiments.csv results\modern_llm
git commit -m "results: add Progress 5 modern local LLM experiments"
git push
```

Kalau baru smoke test, tidak perlu commit hasil smoke kecuali ingin disimpan sebagai bukti debugging.

---

## 13. Kalau tetap ingin lewat WSL

Tidak direkomendasikan untuk kasus ini, karena LM Studio jalan di Windows dan `localhost:1234` sering tidak tembus dari WSL.

Kalau tetap mau coba:

1. Aktifkan **Serve on Local Network** di LM Studio.
2. Izinkan Windows Firewall.
3. Dari WSL:

```bash
WIN_HOST=$(grep nameserver /etc/resolv.conf | awk '{print $2}')
curl http://$WIN_HOST:1234/v1/models
```

Kalau gagal, balik ke Windows PowerShell. Itu jalur utama.

---

## 14. Kapan balik ke agent

Balik ke agent setelah:

1. satu full Twitter zero-shot/few-shot selesai, atau
2. `invalid_outputs` masih tinggi setelah strict few-shot, atau
3. model terlalu lambat/crash.

Kirim info:

```text
Model:
Model alias:
Mode: zero-shot/few-shot
F1:
Invalid outputs:
File hasil:
Error kalau ada:
```
