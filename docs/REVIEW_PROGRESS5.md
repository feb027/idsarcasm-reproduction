# Review Progress 5

## Score
84/100

## Verdict
NEEDS REVISION

## Ringkasan Penilaian
Secara umum, Progress 5 sudah layak sebagai tahap optimasi dan eksperimen tambahan. Scope-nya pas: ada optimasi model utama, screening hyperparameter ringan, error analysis, dan pembanding modern local LLM. Struktur subbab juga sudah rapi, urutan tabel/gambar konsisten, dan mayoritas angka cocok dengan file CSV sumber.

Masalah utama ada pada satu mismatch angka di hasil threshold tuning Reddit. Karena ini menyentuh tabel inti Progress 5, statusnya belum bisa saya anggap pass sebelum diperbaiki.

## Cek yang Lolos
1. Angka Twitter untuk threshold tuning, screening XLM-R Base, error analysis, dan modern LLM sudah cocok dengan:
   - `results/tables/optimization_runs.csv`
   - `results/tables/progress5_transformer_optimization_summary.csv`
   - `results/tables/modern_llm_experiments.csv`
   - `results/tables/progress5_modern_llm_summary.csv`
2. Urutan tabel di `docs/laporan-proyek.md` konsisten dari Tabel 1 sampai Tabel 14.
3. Urutan gambar di `docs/laporan-proyek.md` konsisten dari Gambar 1 sampai Gambar 22.
4. File gambar Progress 5 yang dirujuk memang ada:
   - `progress5_threshold_tuning_f1.png`
   - `progress5_xlmr_base_screening.png`
   - `progress5_threshold_error_transitions.png`
   - `progress5_modern_llm_f1_comparison.png`
   - `progress5_modern_llm_precision_recall.png`
5. Dari sisi isi, Progress 5 sudah cukup untuk disebut tahap optimasi dan eksperimen. Tidak terasa kurang scope.

## Temuan Mayor
1. Angka `Precision Tuned` dan `Recall Tuned` untuk Reddit di tabel threshold tuning tidak cocok dengan CSV sumber.
   - Lokasi: `docs/laporan-proyek.md:414-419`
   - Di laporan tertulis: precision `0,5866`, recall `0,6662`
   - Di CSV sumber tertulis:
     - `results/tables/optimization_runs.csv`: precision `0.5596`, recall `0.7054`
     - `results/tables/progress5_transformer_optimization_summary.csv`: precision `0.5596`, recall `0.7054`
   - Dampak: tabel inti Progress 5 jadi tidak sepenuhnya valid, dan interpretasi Reddit berpotensi meleset.
2. Mismatch angka yang sama juga ada di dokumen pendamping.
   - Lokasi: `docs/progress-5.md:35-45`
   - Nilai Reddit di sana juga tertulis `0.5866` dan `0.6662`, padahal CSV menunjukkan `0.5596` dan `0.7054`.

## Temuan Minor
1. Nada tulisan umumnya sudah natural untuk mahasiswa semester 6, tetapi ada beberapa kalimat yang sedikit terlalu yakin.
   - Contoh: `sudah sangat dekat dengan target paper`, `berhasil menjawab tujuan optimasi proyek`, `masih jauh lebih kuat`
   - Saran: ganti sebagian dengan formulasi lebih akademik seperti `mendekati`, `menunjukkan indikasi`, atau `pada eksperimen ini`.
2. Interpretasi Reddit perlu disesuaikan setelah angka diperbaiki.
   - Dengan angka CSV yang benar, pola Reddit adalah precision tuned turun dari `0.6188` ke `0.5596`, tetapi recall tuned naik dari `0.6048` ke `0.7054`.
   - Jadi narasi yang paling tepat bukan sekadar “naik sedikit”, melainkan “F1 naik kecil karena kenaikan recall lebih besar daripada penurunan precision”.
3. Header identitas di awal `docs/laporan-proyek.md` masih berisi placeholder (`[ISI NIM]`, `[ISI KELAS]`, `[ISI NAMA DOSEN]`).
   - Ini bukan isu khusus Progress 5, tetapi untuk kualitas laporan final sebaiknya segera diisi.

## Fixes Spesifik
1. Perbaiki `docs/laporan-proyek.md:419`
   - Dari: `| Reddit | 0,26 | 0,6117 | 0,6241 | +0,0124 | 0,5866 | 0,6662 |`
   - Menjadi: `| Reddit | 0,26 | 0,6117 | 0,6241 | +0,0124 | 0,5596 | 0,7054 |`
2. Perbaiki `docs/progress-5.md:38`
   - Dari: `| Reddit | 0.26 | 0.6117 | 0.6241 | +0.0124 | 0.5866 | 0.6662 |`
   - Menjadi: `| Reddit | 0.26 | 0.6117 | 0.6241 | +0.0124 | 0.5596 | 0.7054 |`
3. Revisi narasi Reddit di `docs/laporan-proyek.md:412`
   - Tambahkan penjelasan bahwa threshold `0,26` menaikkan recall cukup besar, tetapi precision turun, sehingga kenaikan F1 tetap kecil.
4. Lunakkan 2-3 kalimat yang terlalu absolut, terutama di:
   - `docs/laporan-proyek.md:410-412`
   - `docs/laporan-proyek.md:465`
   - `docs/progress-5.md:136`
5. Isi placeholder identitas pada halaman awal laporan sebelum final submit.

## Kesimpulan Dosen
Secara akademik, Progress 5 ini sudah matang dan cukup untuk menutup fase optimasi. Masalahnya bukan pada scope atau arah eksperimen, melainkan pada konsistensi satu tabel penting. Kalau angka Reddit threshold tuning dan narasinya dibetulkan, bagian ini bisa naik ke status pass.
