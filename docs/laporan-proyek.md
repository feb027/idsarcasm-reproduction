# Laporan Proyek — Reproduksi IdSarcasm

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
