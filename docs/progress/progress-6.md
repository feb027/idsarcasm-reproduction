# Progress 6 — Finalisasi Laporan dan Repository

**Status:** selesai.

Progress 6 menutup proyek dengan finalisasi laporan, analisis komparatif akhir, figure/tabel final, dashboard statis GitHub Pages, error explorer, dan perapian repository GitHub.

## Ringkasan Pekerjaan

1. Menyusun analisis komparatif akhir dari seluruh hasil eksperimen.
2. Membuat tabel final:
   - `results/tables/final_method_comparison.csv`
   - `results/tables/final_method_ranking.csv`
3. Membuat figure final:
   - `results/figures/final_method_comparison.png`
   - `results/figures/final_twitter_method_ranking.png`
   - `results/figures/final_reddit_method_ranking.png`
   - `results/figures/final_progress_summary.png`
4. Membuat dashboard statis dan error explorer:
   - `index.html`
   - `dashboard/app.js`
   - `dashboard/styles.css`
   - `dashboard/data/dashboard-data.json`
4. Memperbarui laporan akhir pada `docs/laporan-proyek.md`.
5. Merapikan repository:
      - file konfigurasi lokal dimasukkan ke `.gitignore`,
   - artifact percobaan sementara dipisahkan dari hasil final,
   - dokumentasi progress dipindahkan ke `docs/progress/`,
   - README dibuat lebih ringkas dan profesional.

## Hasil Akhir

| Dataset | Model terbaik | F1 akhir | Target paper |
|---|---|---:|---:|
| Twitter | XLM-R Large lr=2e-5 | 0.7905 | 0.7692 |
| Reddit | XLM-R Large + threshold tuning | 0.6241 | 0.6274 |

Kesimpulan utama: XLM-R Large menjadi pendekatan terbaik pada kedua dataset. Pada Twitter, optimasi learning rate ke 2e-5 berhasil melampaui skor paper. Pada Reddit, threshold tuning masih menjadi hasil terbaik walaupun sedikit di bawah paper. Zero-shot dan modern local LLM berguna sebagai pembanding, namun belum mendekati fine-tuned transformer. Dashboard statis ditambahkan agar hasil final dan contoh error dapat dicek langsung tanpa menjalankan Colab/training ulang.

## File Utama

```text
docs/laporan-proyek.md
docs/reproducibility.md
README.md
scripts/generate_final_analysis.py
results/tables/final_method_comparison.csv
results/tables/final_method_ranking.csv
results/figures/final_*.png
```
