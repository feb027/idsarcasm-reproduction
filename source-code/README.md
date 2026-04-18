# Source Code Snapshot

Folder ini menyimpan snapshot source code asli dari repo paper agar workflow proyek lebih hemat waktu, lebih faithful ke implementasi penulis, dan tidak perlu menulis ulang dari nol setiap kali masuk fase baru.

## Isi utama
- `original-id-sarcasm/` — snapshot read-only dari repo `w11wo/id_sarcasm`

## Aturan pakai
- Anggap snapshot ini sebagai **referensi tetap**, jangan dijadikan area kerja utama.
- Kalau butuh adaptasi untuk Progress 3/4, **copy file yang relevan** ke area kerja proyek ini lalu edit seperlunya.
- Jangan `mv` file dari snapshot, supaya provenance tetap jelas.

## Snapshot saat ini
- Upstream repo: `https://github.com/w11wo/id_sarcasm`
- Snapshot branch: `main`
- Snapshot commit: `ae32cc3c049cd347cf1508d2af9bfdc7c9b52009`

## File upstream yang paling penting untuk fase berikutnya
- `original-id-sarcasm/train_classical.sh`
- `original-id-sarcasm/scripts/run_classical_classification.py`
- `original-id-sarcasm/train_twitter.sh`
- `original-id-sarcasm/train_reddit.sh`
- `original-id-sarcasm/scripts/run_classification.py`
- `original-id-sarcasm/recipes/`

## Kenapa ini penting
1. Mengurangi rewrite kode yang tidak perlu
2. Menjaga eksperimen tetap dekat ke source code penulis
3. Mempermudah justifikasi di laporan bahwa implementasi dimulai dari codebase asli paper
4. Mempermudah fase transformer baseline dan optimasi di progress berikutnya
