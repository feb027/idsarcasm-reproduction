# Progress 6 — Finalisasi Laporan dan Repository

**Status:** selesai.

Progress 6 menutup proyek dengan finalisasi laporan, analisis komparatif akhir, figure/tabel final, dan perapian repository GitHub.

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
4. Memperbarui laporan akhir pada `docs/laporan-proyek.md`.
5. Merapikan repository:
      - file konfigurasi lokal dimasukkan ke `.gitignore`,
   - artifact percobaan sementara dipisahkan dari hasil final,
   - dokumentasi progress dipindahkan ke `docs/progress/`,
   - README dibuat lebih ringkas dan profesional.

## Hasil Akhir

| Dataset | Model terbaik | F1 akhir | Target paper |
|---|---|---:|---:|
| Twitter | XLM-R Large + threshold tuning | 0.7649 | 0.7692 |
| Reddit | XLM-R Large + threshold tuning | 0.6241 | 0.6274 |

Kesimpulan utama: XLM-R Large setelah threshold tuning menjadi pendekatan terbaik pada kedua dataset. Classical ML masih kuat pada Twitter, tetapi transformer lebih unggul terutama pada Reddit. Zero-shot dan modern local LLM berguna sebagai pembanding, namun belum mendekati fine-tuned transformer.

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
