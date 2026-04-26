# Review Final Laporan Progress 4

**Skor:** 86/100  
**Status:** NEEDS REVISION

Secara umum, pembaruan Progress 4 sudah kuat. Angka pada bagian zero-shot konsisten dengan `results/tables/zeroshot_baselines.csv`, status run Twitter vs Reddit juga sudah sejalan dengan log yang tersedia, dan penomoran tabel/gambar beserta link gambar sudah rapi. Namun, untuk standar laporan akademik semester 6, masih ada beberapa bagian yang perlu direvisi agar klaimnya lebih hati-hati, lebih jelas secara teknis, dan lebih natural sebagai tulisan mahasiswa.

## Penilaian per aspek

### 1. Kelengkapan hasil zero-shot Progress 4
Bagian ini **hampir lengkap**. Tabel 8 sudah memuat 9 model Twitter dan status 5/9 selesai pada Reddit, lalu narasi juga sudah menyebut 4 run Reddit yang tidak selesai.

Perbaikan yang masih perlu:
- Tambahkan satu kalimat yang sangat eksplisit bahwa `zeroshot_baselines.csv` hanya berisi **14 full run yang selesai**, sedangkan 4 run Reddit yang gagal hanya didukung oleh log parsial.
- Pada bagian analisis komparatif, beri penanda yang lebih tegas bahwa “best zero-shot” untuk Reddit berarti **best zero-shot dari run yang selesai**, bukan best dari seluruh 9 model paper.

### 2. Konsistensi dengan `results/tables/zeroshot_baselines.csv` dan log
Untuk angka F1 pada Tabel 8, saya nilai **konsisten** dengan file hasil. Contoh:
- Twitter BLOOMZ-7.1B = 0,3965.
- Twitter mT0 Base = 0,3986.
- Twitter mT0 Large = 0,3989.
- Twitter mT0 XL = 0,3988.
- Reddit BLOOMZ-560M = 0,3857.
- Reddit BLOOMZ-1.1B = 0,3938.
- Reddit BLOOMZ-1.7B = 0,3763.
- Reddit BLOOMZ-3B = 0,3995.
- Reddit mT0 Small = 0,4000.

Status log juga sesuai secara umum:
- `progress-4-zeroshot-reddit-hf-logprobs-bloomz-7b1-full.log`
- `progress-4-zeroshot-reddit-hf-logprobs-mt0-base-full.log`
- `progress-4-zeroshot-reddit-hf-logprobs-mt0-large-full.log`
- `progress-4-zeroshot-reddit-hf-logprobs-mt0-xl-full.log`

Catatan revisi penting:
- Kalimat “pipeline zero-shot yang dipakai sudah sesuai dengan metode paper untuk dataset Twitter” terlalu pasti. Hasil yang sangat dekat memang mendukung itu, tetapi **kedekatan angka belum membuktikan kesesuaian metode secara penuh**. Lebih aman ditulis: “hasil Twitter sangat mendekati paper, sehingga implementasi yang dipakai kemungkinan sudah cukup dekat dengan metode paper.”
- Kalimat “model yang lebih besar lebih rentan terhenti” juga terlalu disederhanakan. Data log menunjukkan run yang gagal memang semuanya berada di Reddit, tetapi salah satunya `mt0-base`, jadi argumen “lebih besar pasti lebih rentan” tidak sepenuhnya bersih. Lebih aman: “run yang lebih berat pada Reddit lebih rentan terputus karena keterbatasan runtime/session Colab.”

### 3. Penomoran tabel/gambar dan link gambar
Bagian ini **sudah baik**.

Temuan:
- Penomoran tabel berurutan dari **Tabel 1 sampai Tabel 9**.
- Penomoran gambar berurutan dari **Gambar 1 sampai Gambar 17**.
- Semua file gambar yang dirujuk pada laporan tersedia di `results/figures/`.

Tidak ada revisi mayor pada aspek ini.

### 4. Nada bahasa Indonesia mahasiswa
Secara umum nadanya sudah cukup natural dan masih terasa seperti laporan mahasiswa, bukan tulisan mesin. Namun masih ada beberapa frasa yang terasa terlalu “catatan proyek” atau terlalu campuran teknis-operasional.

Yang perlu dibenahi:
- Frasa seperti `full run`, `runtime/session habis`, `resource`, dan `run yang selesai` sebaiknya dibuat sedikit lebih formal.
- Misalnya:
  - `full run` -> `eksekusi penuh`
  - `runtime/session Colab habis` -> `sesi Colab berakhir sebelum evaluasi selesai`
  - `keterbatasan resource` -> `keterbatasan sumber daya komputasi`

Bahasanya tidak perlu dibuat terlalu kaku, tetapi untuk laporan kuliah sebaiknya lebih konsisten dalam bahasa Indonesia akademik ringan.

### 5. Kejelasan teknis: Twitter lengkap vs Reddit terbatas runtime
Aspek ini **sudah ada**, tetapi masih bisa dibuat lebih tajam.

Masalah utamanya:
- Laporan sudah benar menyebut Twitter 9/9 selesai dan Reddit 5/9 selesai.
- Namun belum dijelaskan secara cukup tegas bahwa pada Reddit yang gagal, beberapa log hanya berisi tahap awal loading, sedangkan `mt0-base` bahkan sempat mencatat metrik per prompt secara parsial, tetapi **tidak memiliki metrik akhir rata-rata** yang sah untuk dimasukkan ke tabel.

Saran perbaikan:
- Tambahkan satu kalimat seperti ini:  
  “Karena itu, empat run Reddit yang terputus tidak dimasukkan ke tabel hasil akhir dan tidak dipakai dalam perbandingan kuantitatif, walaupun beberapa log masih menyimpan jejak proses atau metrik parsial.”
- Ini penting supaya pembaca paham perbedaan antara:
  - run selesai dan punya `metrics.json` final,
  - run gagal yang hanya punya log parsial.

### 6. Klaim yang belum cukup didukung
Masih ada beberapa klaim yang sebaiknya dilunakkan.

Klaim yang perlu direvisi:
- “bagian Twitter bisa dikatakan berhasil direproduksi dengan sangat dekat”  
  Ini masih bisa diterima, tetapi sebaiknya tetap diberi pagar akademik: “sangat mendekati hasil paper”.
- “pipeline zero-shot yang dipakai sudah sesuai dengan metode paper”  
  Ini terlalu kuat, karena yang benar-benar terlihat dari bukti adalah **hasilnya dekat**, bukan verifikasi metodologis penuh.
- “model yang lebih besar lebih rentan terhenti”  
  Ini perlu dilunakkan menjadi kemungkinan, bukan kesimpulan umum.
- “Ini mendukung temuan paper IdSarcasm bahwa model besar tanpa fine-tuning belum cukup kuat...”  
  Klaim ini cukup masuk akal, tetapi akan lebih aman jika ditulis berdasarkan hasil Anda sendiri terlebih dahulu, misalnya: “Pada reproduksi ini, performa zero-shot masih jauh di bawah transformer fine-tuned, sejalan dengan temuan paper.”

## Kesimpulan dosen

Laporan Progress 4 sudah **kuat pada data, rapi pada aset, dan cukup matang pada struktur hasil**. Masalah utamanya bukan pada angka, melainkan pada cara menyampaikan kesimpulan. Untuk standar laporan akhir yang baik, Anda perlu sedikit memperketat bahasa akademik, memperjelas status run parsial Reddit, dan melunakkan klaim yang saat ini terlalu pasti.

## Perbaikan yang wajib dilakukan

1. Tegaskan bahwa `zeroshot_baselines.csv` hanya memuat 14 eksekusi penuh yang selesai.
2. Jelaskan bahwa 4 run Reddit yang gagal hanya punya log parsial, dan karena itu tidak dipakai dalam tabel akhir.
3. Lunakkan klaim “sudah sesuai dengan metode paper” menjadi “sangat mendekati hasil paper” atau “kemungkinan sudah cukup dekat”.
4. Ubah generalisasi “model yang lebih besar lebih rentan terhenti” menjadi penjelasan yang lebih hati-hati dan spesifik pada kondisi run Reddit.
5. Rapikan beberapa istilah campuran Inggris-Indonesia agar nada laporan lebih natural sebagai tulisan mahasiswa.

Setelah lima poin di atas diperbaiki, laporan ini layak naik ke kategori **PASS**.
