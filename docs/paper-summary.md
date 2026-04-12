# Paper Summary — IdSarcasm

## Informasi Umum
- **Judul:** IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection
- **Penulis:** Derwin Suhartono, Wilson Wongso, Alif Tri Handoyo
- **Publikasi:** IEEE Access, 2024
- **DOI:** 10.1109/ACCESS.2024.3416955
- **GitHub:** https://github.com/w11wo/id_sarcasm

## Masalah
Sarcasm detection merupakan salah satu tantangan terbesar dalam NLP karena sifatnya yang kontekstual dan budaya-spesifik. Untuk bahasa Indonesia, belum ada benchmark publik sebelum paper ini. Dataset sarcasm untuk bahasa rendah-sumber (low-resource) sangat terbatas.

## Kontribusi
1. **Dataset baru:** Dua benchmark dataset publik pertama untuk sarcasm detection bahasa Indonesia:
   - Reddit Indonesia Sarcastic (14,116 komentar)
   - Twitter Indonesia Sarcastic (12,861 tweet)
2. **Benchmark komprehensif:** Evaluasi 3 kategori model (classical ML, fine-tuned transformer, zero-shot LLM)
3. **Analisis perbandingan:** Menunjukkan model mana yang paling efektif untuk tiap dataset

## Metode

### Preprocessing
- Cleaning text (hapus URL, mention, special char)
- Tokenisasi
- Train/val/test split

### Model yang Diuji

#### Classical Machine Learning
| Model | Metode |
|-------|--------|
| Logistic Regression | TF-IDF features |
| Naive Bayes | TF-IDF features |
| SVC | TF-IDF features |

#### Fine-tuned Transformer
| Model | #Params |
|-------|---------|
| IndoBERT Base (IndoNLU) | 124M |
| IndoBERT Large (IndoNLU) | 335M |
| IndoBERT Base (IndoLEM) | 111M |
| mBERT | 178M |
| XLM-R Base | 278M |
| XLM-R Large | 560M |

#### Zero-shot LLM
| Model | Range |
|-------|-------|
| BLOOMZ | 560M → 7.1B |
| mT0 | Small → XL |

## Hasil (F1-Score)

| Model | Reddit | Twitter |
|-------|--------|---------|
| Logistic Regression | 0.4887 | 0.7142 |
| Naive Bayes | 0.4591 | 0.6721 |
| SVC | 0.4467 | 0.6782 |
| IndoBERT Base (IndoNLU) | 0.6100 | 0.7273 |
| IndoBERT Large (IndoNLU) | 0.6184 | 0.7160 |
| IndoBERT Base (IndoLEM) | 0.5671 | 0.6462 |
| mBERT | 0.5338 | 0.6467 |
| XLM-R Base | 0.5690 | 0.7386 |
| **XLM-R Large** | **0.6274** | **0.7692** |
| BLOOMZ-7.1B (zero-shot) | 0.4036 | 0.3968 |
| mT0 XL (zero-shot) | 0.4001 | 0.3988 |

## Analisis
1. **XLM-R Large** perform terbaik di kedua dataset
2. **Fine-tuned transformer > Classical ML > Zero-shot LLM**
3. Zero-shot approach sangat buruk (~0.39) — menunjukkan sarcasm butuh fine-tuning
4. Twitter dataset lebih mudah dari Reddit (F1 lebih tinggi across the board)
5. Reddit lebih kontekstual/sulit karena thread-based conversation

## Potensi Improvement
- Model Indo lebih baru (IndoBERTweet, Gemma)
- Few-shot learning (bukan zero-shot)
- Data augmentation untuk class imbalance
- Ensemble methods
- Attention analysis untuk interpretability
