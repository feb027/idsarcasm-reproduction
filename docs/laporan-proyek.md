# Laporan Proyek — Reproduksi IdSarcasm

## Progress 1: Penetapan Paper & Target Reproduksi

---

## 1. Latar Belakang

Sarkasme merupakan salah satu bentuk ironi verbal yang menampilkan pertentangan antara makna literal dan maksud komunikatif penutur, sehingga menjadikannya fenomena linguistik yang kompleks untuk diproses secara komputasional [1]. Dalam tugas *sentiment analysis*, keberadaan sarkasme sangat problematis karena ujaran yang secara permukaan tampak positif dapat sesungguhnya menyampaikan evaluasi negatif, kritik, atau ejekan. Akibatnya, kegagalan mengenali sarkasme dapat menurunkan akurasi sistem NLP pada berbagai aplikasi turunan, seperti analisis opini publik, pemantauan reputasi, moderasi konten, dan pemahaman percakapan daring.

Urgensi persoalan tersebut semakin tinggi dalam konteks Indonesia. Ekosistem digital Indonesia merupakan salah satu yang terbesar di dunia, dengan lebih dari 170 juta pengguna internet dan sekitar 139 juta identitas pengguna media sosial aktif pada Januari 2024 [6]. Dalam ruang digital yang sangat besar tersebut, ekspresi sarkastik muncul secara luas pada platform berbasis percakapan singkat dan diskusi komunitas, khususnya X/Twitter dan Reddit Indonesia, yang sering menjadi medium penyampaian kritik sosial, komentar politik, humor, serta respons terhadap isu populer. Karakter tuturan di kedua platform ini cenderung informal, kontekstual, dan sarat implikatur, sehingga menambah tingkat kesulitan deteksi sarkasme dalam bahasa Indonesia.

Di ranah *Natural Language Processing* (NLP), deteksi sarkasme telah berkembang pesat untuk bahasa Inggris melalui pendekatan berbasis aturan, fitur linguistik, hingga model *deep learning* modern [1]. Sebaliknya, penelitian untuk bahasa Indonesia masih relatif terbatas. Keterbatasan tersebut bukan hanya disebabkan oleh kompleksitas sarkasme sebagai fenomena pragmatik, tetapi juga oleh kondisi bahasa Indonesia yang masih menghadapi keterbatasan sumber daya NLP, seperti ketersediaan dataset beranotasi, model pralatih, serta *benchmark* publik yang memadai [2].

Sejumlah studi awal telah memberi fondasi bagi penelitian ini. Lunando dan Purwarianti [2] mengkaji deteksi sarkasme dalam analisis sentimen media sosial Indonesia dengan pendekatan klasik. Ranti dan Girsang [3] kemudian menunjukkan bahwa *Convolutional Neural Network* (CNN) dapat digunakan untuk meningkatkan performa deteksi sarkasme bahasa Indonesia. Selanjutnya, Khotijah *et al.* [4] mengeksplorasi pendekatan berbasis konteks menggunakan LSTM untuk data berbahasa Indonesia dan Inggris. Meskipun penting, penelitian-penelitian tersebut masih menyisakan sejumlah keterbatasan, yaitu belum tersedianya *benchmark* publik yang standar, belum adanya evaluasi yang sistematis lintas keluarga model, dan belum optimalnya pengujian model bahasa modern untuk kasus sarkasme bahasa Indonesia.

Kesenjangan tersebut direspons oleh Suhartono, Wongso, dan Handoyo melalui paper *IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection* [5]. Studi ini penting karena memperkenalkan *benchmark* deteksi sarkasme bahasa Indonesia yang disusun dari dua platform media sosial, yaitu Reddit dan Twitter, sekaligus membandingkan tiga kelompok pendekatan secara komprehensif: *classical machine learning*, *fine-tuned pre-trained language models*, dan *zero-shot large language models*. Dengan demikian, paper ini tidak hanya menyediakan dataset dan kerangka evaluasi yang lebih sistematis, tetapi juga menjadi rujukan utama untuk pengembangan penelitian sarkasme berbahasa Indonesia.

Berdasarkan konteks tersebut, proyek reproduksi ini dilakukan untuk memvalidasi hasil yang dilaporkan dalam paper IdSarcasm [5], dengan fokus awal pada reproduksi *baseline* *classical machine learning*. Pemilihan ruang lingkup ini bersifat metodologis dan realistis: reproduksi tetap diarahkan untuk menghasilkan temuan yang bermakna secara akademik, sambil menyesuaikan keterbatasan sumber daya komputasi yang tersedia pada tahap awal pengerjaan.

---

## 2. Referensi

[1] A. Joshi, P. Bhattacharyya, and M. J. Carman, "Automatic Sarcasm Detection: A Survey," *ACM Computing Surveys*, vol. 50, no. 5, art. no. 73, pp. 1-22, 2017, doi: 10.1145/3124420.

[2] E. Lunando and A. Purwarianti, "Indonesian Social Media Sentiment Analysis with Sarcasm Detection," in *2013 International Conference on Advanced Computer Science and Information Systems (ICACSIS)*, Bali, Indonesia, 2013, pp. 195-198, doi: 10.1109/ICACSIS.2013.6761557.

[3] K. S. Ranti and A. S. Girsang, "Indonesian Sarcasm Detection Using Convolutional Neural Network," *International Journal of Emerging Trends in Engineering Research*, vol. 8, no. 9, pp. 6448-6453, 2020, doi: 10.30534/ijeter/2020/10892020.

[4] K. Khotijah, J. Tirtawangsa, and A. B. W. Putra, "Using LSTM for Context Based Approach of Sarcasm Detection in Indonesian and English," in *2020 International Conference on Data Science and Its Applications (ICoDSA)*, Bandung, Indonesia, 2020, pp. 1-6, doi: 10.1109/ICoDSA50139.2020.9212955.

[5] D. Suhartono, W. Wongso, and A. T. Handoyo, "IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection," *IEEE Access*, vol. 12, pp. 87323-87332, 2024, doi: 10.1109/ACCESS.2024.3416955.

[6] DataReportal, "Digital 2024: Indonesia," 2024. [Online]. Available: https://datareportal.com/reports/digital-2024-indonesia. [Accessed: Apr. 16, 2026].

[7] S. M. Muhammad, I. A. Waseem, and M. N. A. Khan, "Sarcasm Detection: A Survey on Sarcasm," *Grenze International Journal of Engineering and Technology*, vol. 10, no. 2, pp. 10-14, 2024.

---

## 3. Identifikasi Paper

| Item | Detail |
|------|--------|
| **Judul** | IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection |
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

## 6. Target Reproduksi

### Primary Scope (Wajib)
Reproduksi baseline classical ML pada dataset Twitter:
- Logistic Regression
- Naive Bayes
- SVM

### Secondary Scope
Reproduksi baseline classical ML pada dataset Reddit.

### Stretch Goal
Fine-tune 1 model transformer (IndoBERT Base / XLM-R Base) via Google Colab.

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
