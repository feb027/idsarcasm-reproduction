# Laporan Proyek — Optimasi Performa Model Transformer dalam Klasifikasi Sarkasme Teks Berbahasa Indonesia Berdasarkan Benchmark IdSarcasm

## Progress 1: Penetapan Paper & Target Reproduksi

---

## 1. Latar Belakang

Deteksi sarkasme merupakan salah satu tantangan besar dalam bidang Natural Language Processing (NLP). Sarkasme sendiri adalah bentuk ironi di mana penutur menyampaikan makna yang berlawanan dengan kata-kata yang diucapkan [1]. Hal ini membuat sistem NLP sering salah membaca sentimen sebuah teks — kalimat yang terlihat positif bisa jadi sebenarnya bernada negatif karena sarkasme. Akibatnya, aplikasi seperti analisis sentimen, moderasi konten, dan pemantauan opini publik bisa menghasilkan kesimpulan yang keliru jika sarkasme tidak terdeteksi.

Untuk bahasa Inggris, penelitian deteksi sarkasme sudah cukup matang, mulai dari metode berbasis aturan sampai model deep learning [1]. Namun untuk bahasa Indonesia, penelitian di bidang ini masih jauh tertinggal. Salah satu penyebabnya adalah ketersediaan sumber daya NLP untuk bahasa Indonesia yang terbatas, seperti dataset beranotasi dan benchmark publik [2]. Padahal, Indonesia memiliki salah satu ekosistem digital terbesar di dunia. Berdasarkan laporan DataReportal 2025, terdapat sekitar 143 juta pengguna media sosial aktif di Indonesia pada Januari 2025, atau sekitar 50,2% dari total populasi [6]. Dengan jumlah pengguna sebesar itu, konten sarkastik di platform seperti Twitter dan Reddit Indonesia sangatlah banyak — mulai dari komentar politik, kritik sosial, sampai humor.

Beberapa penelitian sebelumnya sudah mencoba mengatasi masalah ini. Lunando dan Purwarianti [2] menerapkan deteksi sarkasme pada analisis sentimen media sosial Indonesia menggunakan pendekatan klasik. Ranti dan Girsang [3] kemudian menunjukkan bahwa CNN bisa meningkatkan performa dibandingkan metode klasik. Khotijah *et al.* [4] mengeksplorasi pendekatan LSTM berbasis konteks untuk data berbahasa Indonesia dan Inggris. Jeremy [7] juga meneliti pengaruh tahapan preprocessing terhadap akurasi deteksi sarkasme di media sosial Indonesia. Meskipun kontribusinya penting, penelitian-penelitian ini belum menghasilkan benchmark yang bisa diakses publik dan dievaluasi secara terbuka.

Kekurangan tersebut akhirnya diisi oleh Suhartono, Wongso, dan Handoyo [5] melalui paper "IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection". Paper ini memperkenalkan benchmark deteksi sarkasme bahasa Indonesia pertama yang tersedia secara publik, dengan dataset dari Reddit dan Twitter, serta membandingkan tiga kelas model secara komprehensif: classical machine learning, fine-tuned pre-trained language models, dan zero-shot large language models.

Proyek reproduksi ini bertujuan untuk memvalidasi hasil yang dilaporkan dalam paper IdSarcasm [5], dengan fokus pada reproduksi baseline classical machine learning. Ruang lingkup ini dipilih karena realistis untuk dikerjakan dengan perangkat yang tersedia, namun tetap menghasilkan temuan yang bermakna secara akademis.

---

## 2. Referensi

[1] A. Joshi, P. Bhattacharyya, and M. J. Carman, "Automatic Sarcasm Detection: A Survey," *ACM Computing Surveys*, vol. 50, no. 5, art. no. 73, pp. 1-22, 2017, doi: 10.1145/3124420.

[2] E. Lunando and A. Purwarianti, "Indonesian Social Media Sentiment Analysis with Sarcasm Detection," in *2013 International Conference on Advanced Computer Science and Information Systems (ICACSIS)*, Bali, Indonesia, 2013, pp. 195-198, doi: 10.1109/ICACSIS.2013.6761557.

[3] K. S. Ranti and A. S. Girsang, "Indonesian Sarcasm Detection Using Convolutional Neural Network," *International Journal of Emerging Trends in Engineering Research*, vol. 8, no. 9, pp. 6448-6453, 2020, doi: 10.30534/ijeter/2020/10892020.

[4] K. Khotijah, J. Tirtawangsa, and A. B. W. Putra, "Using LSTM for Context Based Approach of Sarcasm Detection in Indonesian and English," in *2020 International Conference on Data Science and Its Applications (ICoDSA)*, Bandung, Indonesia, 2020, pp. 1-6, doi: 10.1109/ICoDSA50139.2020.9212955.

[5] D. Suhartono, W. Wongso, and A. T. Handoyo, "IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection," *IEEE Access*, vol. 12, pp. 87323-87332, 2024, doi: 10.1109/ACCESS.2024.3416955.

[6] DataReportal, "Digital 2025: Indonesia," Feb. 2025. [Online]. Available: https://datareportal.com/reports/digital-2025-indonesia. [Accessed: Apr. 16, 2026].

[7] N. H. Jeremy, "The Impact of Text Preprocessing in Sarcasm Detection on Indonesian Social Media Contents," *Engineering, Mathematics and Computer Science Journal (EMACS)*, vol. 7, no. 1, 2025, doi: 10.33021/emacs.v7i1.13503.

---

## 3. Identifikasi Paper dan Judul Proyek

| Item | Detail |
|------|--------|
| **Judul Proyek (Google Sheets)** | Optimasi Performa Model Transformer dalam Klasifikasi Sarkasme Teks Berbahasa Indonesia Berdasarkan Benchmark IdSarcasm |
| **Paper Acuan** | IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection |
| **Penulis** | Derwin Suhartono, Wilson Wongso, Alif Tri Handoyo |
| **Institusi** | Bina Nusantara University, Jakarta |
| **Publikasi** | IEEE Access, Volume 12, 2024 |
| **DOI** | 10.1109/ACCESS.2024.3416955 |
| **GitHub** | https://github.com/w11wo/id_sarcasm |

---

## 4. Dataset

| Dataset | Train | Val | Test | Total |
|---------|-------|-----|------|-------|
| Reddit Indonesia Sarcastic | 9,881 | 1,411 | 2,824 | 14,116 |
| Twitter Indonesia Sarcastic | 1,878 | 268 | 538 | 2,684 |

**Source:** HuggingFace — w11wo/reddit_indonesia_sarcastic & w11wo/twitter_indonesia_sarcastic

---

## 5. Algoritma di Paper

### Classical Machine Learning
- Logistic Regression (BoW, TF-IDF)
- Naive Bayes / Multinomial NB (BoW, TF-IDF)
- Support Vector Machine / SVM (BoW, TF-IDF)

### Fine-tuned Pre-trained Language Models
- IndoBERT Base (IndoNLU) — 124M parameters
- IndoBERT Large (IndoNLU) — 335M parameters
- IndoBERT Base (IndoLEM) — 111M parameters
- mBERT — 178M parameters
- XLM-R Base — 278M parameters
- XLM-R Large — 560M parameters

### Zero-shot LLM
- BLOOMZ (560M – 7.1B)
- mT0 (Small – XL)

---

## 6. Target Reproduksi dan Arah Pengembangan

### Progress 2 (fondasi wajib)
Reproduksi baseline classical ML pada benchmark IdSarcasm:
- Logistic Regression
- Naive Bayes
- SVM

### Progress 3 (lebih berbobot, sejalan dengan judul proyek)
Reproduksi baseline transformer pada benchmark IdSarcasm:
- IndoBERT Base atau XLM-R Base
- Fokus awal pada Twitter, lalu diperluas jika resource cukup

### Progress 4 (inti optimasi)
Optimasi performa transformer melalui eksperimen terarah, misalnya:
- hyperparameter tuning,
- class weighting,
- variasi preprocessing / normalization,
- atau pengaturan max length / input strategy.

### Tujuan akhir
Bukan hanya memvalidasi baseline paper, tetapi menunjukkan apakah model transformer yang dioptimasi dapat memberikan peningkatan performa yang jelas pada tugas klasifikasi sarkasme berbahasa Indonesia.

---

## 7. Target Metrik (F1-Score)

| Model | Twitter (Target) | Reddit (Target) |
|-------|-----------------|-----------------|
| Logistic Regression (TF-IDF) | 0.7142 | 0.4887 |
| Naive Bayes (TF-IDF) | 0.6721 | 0.4591 |
| SVM (TF-IDF) | 0.6782 | 0.4467 |

---

## 8. Environment

- **Lokal:** Windows 11, i5-12400F, RX 6600 8GB, 16GB RAM
- **Target eksekusi:** Local PC untuk classical ML, Google Colab untuk transformer
- **Stack:** Python 3.10+, scikit-learn, PyTorch, HuggingFace Transformers
- **Dosen tracking:** Google Sheets (6 progress stages)

---

### Progress 2: Analisis Dataset, Eksplorasi Data, dan Baseline Klasikal

---

## 2. Analisis Proyek

### 2.1 Objek dan Dataset

Objek penelitian dalam proyek ini adalah teks berbahasa Indonesia yang mengandung sarkasme, yang bersumber dari dua platform media sosial yaitu Reddit dan Twitter. Dataset yang digunakan merupakan dataset benchmark IdSarcasm yang dirilis oleh Suhartono *et al.* [5] melalui platform HuggingFace. Dataset ini dikumpulkan dari komentar dan cuitan pengguna media sosial Indonesia yang telah dianotasi sebagai sarkastik atau non-sarkastik oleh penulis aslinya.

Dataset Reddit Indonesia Sarcastic terdiri dari 14.116 komentar yang dibagi menjadi tiga subset: train (9.881 data), validasi (1.411 data), dan test (2.824 data). Sementara itu, dataset Twitter Indonesia Sarcastic berisi 2.684 cuitan dengan pembagian train (1.878 data), validasi (268 data), dan test (538 data). Kedua dataset memiliki proporsi kelas yang konsisten di seluruh subset, yaitu 25% label sarkastik dan 75% label non-sarkastik (rasio 1:3). Meskipun secara teknis tidak seimbang, proporsi ini seragam antara train, validasi, dan test, sehingga tidak ada subset yang lebih "berat" dari yang lain [5].

Selama tahap eksplorasi data awal (Exploratory Data Analysis / EDA), dilakukan pemeriksaan kualitas data yang mencakup pengecekan nilai kosong, duplikasi, dan distribusi panjang teks. Hasilnya menunjukkan bahwa tidak ada nilai kosong pada kedua dataset. Untuk dataset Reddit, ditemukan 10 data duplikat, sedangkan dataset Twitter tidak memiliki duplikat sama sekali. Distribusi panjang teks menunjukkan bahwa rata-rata komentar sarkastik di Reddit cenderung lebih pendek dibandingkan non-sarkastik (67 vs 104 karakter), sementara di Twitter perbedaannya tidak signifikan (118 vs 114 karakter).

![Distribusi Label per Dataset](../../results/figures/distribusi_label_placeholder.png)

<!-- TODO: Generate EDA figure — distribusi label (bar chart) dari notebook 01_eda -->

![Distribusi Panjang Teks per Dataset](../../results/figures/distribusi_panjang_placeholder.png)

<!-- TODO: Generate EDA figure — distribusi panjang teks (histogram) dari notebook 01_eda -->

Perbedaan ukuran kedua dataset ini cukup mencolok. Dataset Reddit memiliki volume data sekitar lima kali lipat lebih besar dibandingkan Twitter. Ini perlu diperhatikan karena data lebih banyak belum tentu hasilnya lebih bagus kalau karakteristik teksnya beda. Berdasarkan temuan EDA, teks Reddit memiliki variasi panjang yang lebih lebar, sedangkan teks Twitter lebih seragam.

### 2.2 Algoritma atau Metode

Untuk Progress 2, tiga algoritma classical machine learning direproduksi sesuai dengan yang digunakan dalam paper IdSarcasm [5], yaitu Logistic Regression, Naive Bayes (Multinomial), dan Support Vector Machine (SVM). Ketiga algoritma ini merupakan baseline standar dalam tugas klasifikasi teks yang telah banyak digunakan dalam penelitian NLP sebelumnya, di antaranya untuk klasifikasi sentimen dan deteksi sarkasme [2][9][13].

**Logistic Regression (LR)** adalah model klasifikasi yang memprediksi probabilitas sebuah teks termasuk ke dalam kelas tertentu menggunakan fungsi sigmoid. Meskipun namanya mengandung kata "regression", model ini sebenarnya digunakan untuk klasifikasi. Dalam konteks klasifikasi teks, LR bekerja dengan mempelajari bobot (weight) untuk setiap fitur kata yang merepresentasikan seberapa kuat kata tersebut mengindikasikan kelas sarkastik atau non-sarkastik. Hyperparameter utama yang digunakan dalam eksperimen ini adalah **C**, yaitu parameter yang mengontrol seberapa ketat model mengikuti data latih. C kecil (misalnya 0,01) berarti regularisasi kuat, sehingga model cenderung lebih sederhana dan tidak overfit. Sebaliknya, C besar (misalnya 100) membuat model lebih fleksibel dalam menyesuaikan data pelatihan, tetapi berisiko menghafal data (overfitting). Pada paper, rentang pencarian C adalah [0,01, 0,1, 1, 10, 100] [5].

**Naive Bayes (Multinomial NB)** adalah algoritma klasifikasi probabilistik yang didasarkan pada Teorema Bayes dengan asumsi bahwa setiap fitur (kata) bersifat independen satu sama lain. Meskipun asumsi independensi ini jarang terpenuhi dalam data teks nyata, Naive Bayes tetap menjadi baseline yang kompetitif karena kesederhanaan dan kecepatannya [9]. Hyperparameter utama yang digunakan adalah **alpha** (α), yaitu parameter smoothing Laplace yang mengatur bagaimana model menangani kata-kata yang tidak muncul dalam data pelatihan. Nilai alpha yang terlalu kecil membuat model terlalu bergantung pada frekuensi kata yang terlihat, sedangkan alpha yang terlalu besar membuat distribusi probabilitas terlalu seragam. Pada paper, alpha dicari dalam rentang 0,001 sampai 1 sebanyak 50 titik menggunakan `linspace` [5].

**Support Vector Machine (SVM)**, dalam implementasinya disebut juga Support Vector Classification (SVC), adalah algoritma yang bekerja dengan mencari hyperplane (bidang pemisah) yang memaksimalkan margin antara dua kelas dalam ruang fitur [10]. SVM efektif untuk klasifikasi teks karena mampu menangani dimensi fitur yang tinggi. Hyperparameter yang digunakan meliputi **C** (parameter regularisasi, sama seperti pada LR) dan **kernel**, yaitu fungsi yang mengukur kemiripan antar data sehingga SVM bisa menemukan batas keputusan yang lebih kompleks. Pada paper, dua jenis kernel dievaluasi: **linear** (pemisahan langsung menggunakan garis lurus) dan **rbf** (Radial Basis Function, yang mengukur jarak antar data dan mampu menangkap pola non-linear) [5].

Sebagai representasi fitur teks, digunakan dua metode vektorisasi yaitu **Bag of Words (BoW)** dan **TF-IDF** (Term Frequency-Inverse Document Frequency). BoW merepresentasikan setiap dokumen sebagai vektor frekuensi kemunculan setiap kata dalam vocabulary. Metode ini sederhana tetapi tidak mempertimbangkan penting-tidaknya sebuah kata dalam korpus secara keseluruhan. TF-IDF memperbaiki kelemahan ini dengan memberikan bobot lebih tinggi pada kata yang sering muncul dalam satu dokumen tetapi jarang muncul di dokumen lain, sehingga kata-kata umum seperti "dan" atau "yang" mendapat bobot rendah [8]. Proses tokenisasi teks dilakukan menggunakan `nltk.word_tokenize` yang memecah kalimat menjadi token-token kata sebelum vektorisasi.

### 2.3 Analisis Kebutuhan Proyek

Untuk menjalankan eksperimen baseline classical ML pada proyek ini, kebutuhan yang harus dipenuhi mencakup aspek perangkat lunak dan perangkat keras. Dari sisi perangkat lunak, dibutuhkan Python 3.10 ke atas dengan pustaka scikit-learn untuk implementasi ketiga algoritma, pustaka pandas untuk manipulasi data, serta pustaka nltk untuk tokenisasi teks. Dataset diperoleh dari HuggingFace dan telah di-cache secara lokal dalam format CSV untuk mempercepat proses loading.

Dari sisi perangkat keras, eksperimen classical ML tidak membutuhkan GPU karena scikit-learn berjalan di CPU. Eksperimen ini dapat dijalankan pada komputer lokal dengan spesifikasi standar (i5-12400F, 16GB RAM) tanpa kendala. Seluruh eksperimen Progress 2 diselesaikan dalam waktu beberapa menit, menjadikannya sangat efisien untuk iterasi dan debugging.

---

## 3. Pemodelan/Sistem/Aplikasi

### 3.1 Ilustrasi atau Arsitektur Proyek

Alur kerja (pipeline) eksperimen classical ML pada proyek ini terdiri dari beberapa tahap utama yang saling berurutan. Pertama, data mentah dimuat dari file CSV yang telah di-cache secara lokal. Kemudian, teks diproses melalui tahap tokenisasi menggunakan `nltk.word_tokenize` untuk memecah kalimat menjadi kata-kata individual. Setelah itu, teks yang sudah ditokenisasi direpresentasikan sebagai vektor numerik menggunakan Bag of Words (CountVectorizer) atau TF-IDF (TfidfVectorizer). Vektor fitur ini kemudian digunakan untuk melatih model klasifikasi (LR, NB, atau SVM) dengan pencarian hyperparameter melalui GridSearchCV. Terakhir, model terbaik dievaluasi pada subset test menggunakan metrik accuracy, precision, recall, dan F1-score.

![Arsitektur Pipeline Eksperimen Classical ML](../../results/figures/pipeline_architecture.png)

### 3.2 Tahapan

Eksperimen dilaksanakan dalam beberapa tahap sebagai berikut. Pertama, dataset dimuat dari HuggingFace dan disimpan dalam format CSV lokal. Tahap ini mencakup pembagian data menjadi subset train, validasi, dan test sesuai dengan split yang telah ditentukan oleh penulis paper. Kedua, dilakukan tahap EDA untuk memahami karakteristik dataset, termasuk distribusi kelas, panjang teks, dan kualitas data. Ketiga, teks diproses melalui tokenisasi dan vektorisasi. Keempat, ketiga model (LR, NB, SVM) dilatih menggunakan GridSearchCV dengan PredefinedSplit untuk menemukan kombinasi hyperparameter terbaik pada masing-masing dataset (Twitter dan Reddit) dan masing-masing metode vektorisasi (BoW dan TF-IDF). GridSearchCV bekerja dengan mencoba semua kombinasi hyperparameter yang ditentukan, lalu mengevaluasi tiap kombinasi menggunakan cross-validation [11]. Dalam eksperimen ini, data train dan validasi digabungkan, kemudian PredefinedSplit digunakan agar data validasi tetap menjadi holdout (tidak dilatih ulang) selama pencarian. Cara ini memastikan bahwa proses tuning hyperparameter konsisten dengan pendekatan paper asli. Terakhir, model dengan hyperparameter terbaik dievaluasi pada subset test untuk menghitung accuracy, precision, recall, dan F1-score.

Untuk memastikan reproduktibilitas, seluruh proses eksperimen dijalankan melalui skrip Python (`scripts/run_classical_baselines.py`) yang dapat dijalankan ulang secara konsisten. Hasil evaluasi disimpan dalam format CSV di direktori `results/tables/` untuk kemudian dianalisis dan dibandingkan dengan hasil yang dilaporkan paper.

### 3.3 Hasil dan Evaluasi

Untuk mengevaluasi performa model, digunakan empat metrik klasifikasi standar: accuracy, precision, recall, dan F1-score [12][14]. **Accuracy** mengukur proporsi prediksi yang benar dari seluruh data test, yaitu seberapa sering model memprediksi dengan tepat. Namun, pada dataset yang tidak seimbang, accuracy bisa menyesatkan karena model bisa mendapat accuracy tinggi hanya dengan selalu memprediksi kelas mayoritas [14]. **Precision** mengukur dari seluruh data yang diprediksi sebagai sarkastik, berapa persen yang benar-benar sarkastik. **Recall** mengukur dari seluruh data yang benar-benar sarkastik, berapa persen yang berhasil dideteksi oleh model. **F1-score** adalah rata-rata harmonik antara precision dan recall, yang memberikan satu angka tunggal yang menyeimbangkan keduanya. F1-score menjadi metrik utama dalam paper IdSarcasm karena kemampuannya menangkap trade-off antara precision dan recall pada dataset yang tidak seimbang [5][12].

Berikut adalah hasil eksperimen baseline classical ML pada dataset Twitter:

**Hasil Eksperimen — Dataset Twitter**

| Vektorisasi | Model | Best Params | Accuracy | Precision | Recall | F1-Score |
|-------------|-------|-------------|----------|-----------|--------|----------|
| BoW | Logistic Regression | C=100 | 0,8587 | 0,7101 | 0,7313 | 0,7206 |
| BoW | Naive Bayes | α=0,450 | 0,8532 | 0,7570 | 0,6045 | 0,6722 |
| BoW | SVM | C=100, kernel=rbf | 0,8513 | 0,7250 | 0,6493 | 0,6850 |
| TF-IDF | Logistic Regression | C=10 | 0,8662 | 0,7627 | 0,6716 | 0,7143 |
| TF-IDF | Naive Bayes | α=0,103 | 0,8197 | 0,7761 | 0,3881 | 0,5174 |
| TF-IDF | SVM | C=10, kernel=rbf | 0,8625 | 0,8125 | 0,5821 | 0,6783 |

![Perbandingan F1-Score pada Dataset Twitter](../../results/figures/f1_twitter_bow_vs_tfidf.png)

Berikut adalah hasil eksperimen pada dataset Reddit:

**Hasil Eksperimen — Dataset Reddit**

| Vektorisasi | Model | Best Params | Accuracy | Precision | Recall | F1-Score |
|-------------|-------|-------------|----------|-----------|--------|----------|
| BoW | Logistic Regression | C=1 | 0,7840 | 0,6000 | 0,4079 | 0,4857 |
| BoW | Naive Bayes | α=0,531 | 0,7890 | 0,6389 | 0,3584 | 0,4592 |
| BoW | SVM | C=0,1, kernel=linear | 0,7851 | 0,6592 | 0,2904 | 0,4031 |
| TF-IDF | Logistic Regression | C=10 | 0,7847 | 0,5980 | 0,4235 | 0,4959 |
| TF-IDF | Naive Bayes | α=0,062 | 0,7776 | 0,6500 | 0,2394 | 0,3499 |
| TF-IDF | SVM | C=1, kernel=linear | 0,7886 | 0,6461 | 0,3414 | 0,4467 |

![Perbandingan F1-Score pada Dataset Reddit](../../results/figures/f1_reddit_bow_vs_tfidf.png)

Untuk memvalidasi reproduktibilitas, hasil eksperimen dibandingkan dengan target F1-score yang dilaporkan dalam paper IdSarcasm [5]:

**Perbandingan Hasil Reproduksi vs Paper (TF-IDF)**

| Model | Twitter Paper | Twitter Reproduksi | Selisih | Reddit Paper | Reddit Reproduksi | Selisih |
|-------|--------------|-------------------|---------|-------------|-------------------|---------|
| Logistic Regression | 0,7142 | 0,7143 | +0,0001 | 0,4887 | 0,4959 | +0,0072 |
| Naive Bayes | 0,6721 | 0,5174 | -0,1547 | 0,4591 | 0,3499 | -0,1092 |
| SVM | 0,6782 | 0,6783 | +0,0001 | 0,4467 | 0,4467 | 0,0000 |

![Perbandingan F1-Score Reproduksi vs Paper](../../results/figures/f1_reproduksi_vs_paper.png)

Dari tabel perbandingan di atas, terlihat bahwa reproduksi untuk Logistic Regression dan SVM pada dataset Twitter menghasilkan F1-score yang sangat mendekati bahkan identik dengan yang dilaporkan paper. Hal ini menunjukkan bahwa implementasi eksperimen berhasil mereproduksi hasil paper dengan baik untuk kedua model tersebut. Untuk Logistic Regression pada dataset Reddit, hasil reproduksi sedikit di atas target paper (+0,0072), yang kemungkinan disebabkan oleh perbedaan versi pustaka atau seed random yang berbeda saat GridSearchCV.

Namun, untuk Naive Bayes terdapat gap yang cukup signifikan, terutama pada dataset Twitter (-0,1547) dan Reddit (-0,1092). Penyebab utama gap ini adalah perbedaan dataset yang digunakan. Paper IdSarcasm [5] melaporkan bahwa dataset Twitter versi asli mereka berisi 12.861 data yang tidak seimbang, sedangkan versi benchmark yang dirilis di HuggingFace dan digunakan dalam reproduksi ini hanya berisi 2.684 data dengan rasio kelas 25:75. Perbedaan ukuran yang hampir lima kali lipat ini berdampak besar pada Naive Bayes, karena algoritma ini mengestimasi probabilitas kelas secara langsung dari frekuensi kata per kelas — jadi distribusi kata yang berubah karena perbedaan ukuran dan proporsi data akan langsung mengubah probabilitas yang dipelajari [9]. Sebagai perbandingan, Logistic Regression dan SVM lebih tahan terhadap perubahan ukuran dataset karena keduanya mengoptimalkan fungsi loss (fungsi kerugian) atas seluruh data pelatihan, yang membuat boundary keputusan yang dihasilkan lebih stabil meskipun jumlah data berubah. Meskipun demikian, pola umum hasil tetap konsisten dengan paper: Logistic Regression dan SVM cenderung lebih baik daripada Naive Bayes, dan TF-IDF umumnya menghasilkan performa yang lebih stabil dibandingkan BoW.

---

## 4. Rencana Pengembangan Proyek

Berdasarkan hasil baseline classical ML pada Progress 2, rencana pengembangan selanjutnya adalah memperluas eksperimen ke model transformer sesuai dengan judul proyek. Pada Progress 3, akan dilakukan reproduksi baseline transformer menggunakan IndoBERT Base atau XLM-R Base pada dataset Twitter. Pemilihan model transformer kecil ini didasarkan pada keterbatasan perangkat keras yang tersedia, sehingga model besar seperti XLM-R Large (560M parameter) tidak dijadikan target awal.

Pada Progress 4, akan dilakukan optimasi terarah terhadap model transformer, misalnya melalui penyetelan hyperparameter (learning rate, batch size, jumlah epoch), penyesuaian class weighting untuk menangani ketidakseimbangan kelas, atau variasi strategi preprocessing teks. Tujuan dari tahap optimasi ini adalah untuk mengeksplorasi apakah peningkatan performa yang signifikan dapat dicapai dibandingkan baseline classical ML dan baseline transformer standar.

Pada Progress 5 dan 6, seluruh hasil eksperimen (classical ML dan transformer) akan dianalisis secara komprehensif, termasuk error analysis untuk memahami jenis-jenis teks sarkastik yang masih sulit dideteksi oleh model. Hasil analisis ini akan menjadi dasar untuk penyusunan laporan akhir yang memuat narasi perbandingan performa, pembahasan keterbatasan, dan usulan pengembangan di masa mendatang.

---

## Referensi Tambahan (Progress 2)

[8] C. D. Manning, P. Raghavan, and H. Schütze, *Introduction to Information Retrieval*. Cambridge, U.K.: Cambridge University Press, 2008, doi: 10.1017/CBO9780511809071.

[9] A. McCallum and K. Nigam, "A Comparison of Event Models for Naive Bayes Text Classification," in *AAAI-98 Workshop on Learning for Text Categorization*, Madison, WI, USA, 1998, pp. 41-48.

[10] C. Cortes and V. Vapnik, "Support-Vector Networks," *Machine Learning*, vol. 20, no. 3, pp. 273-297, 1995, doi: 10.1007/BF00994018.

[11] F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011, doi: 10.5555/1953048.2078195.

[12] M. Sokolova and G. Lapalme, "A Systematic Analysis of Performance Measures for Classification Tasks," *Information Processing & Management*, vol. 45, no. 4, pp. 427-437, 2009, doi: 10.1016/j.ipm.2009.03.002.

[13] K. Taha, P. D. Yoo, C. Y. Yeun, D. Homouz, and A. Taha, "A Comprehensive Survey of Text Classification Techniques and Their Research Applications: Observational and Experimental Insights," *Computer Science Review*, vol. 54, art. no. 100664, 2024, doi: 10.1016/j.cosrev.2024.100664.

[14] G. Naidu *et al.*, "Accuracy, Precision, Recall, F1-Score, or MCC? Empirical Evidence from Advanced Statistics, ML, and XAI for Evaluating Business Predictive Models," *Journal of Big Data*, vol. 12, art. no. 1313, 2025, doi: 10.1186/s40537-025-01313-4.
