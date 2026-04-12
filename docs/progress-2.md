# Progress 2: Dataset & Preprocessing (EDA)

**Status:** ✅ Selesai
**Tanggal:** 12 April 2026

---

## 1. Dataset Download

Dataset di-download dari HuggingFace menggunakan `datasets` library:

| Dataset | HuggingFace ID | Total |
|---------|---------------|-------|
| Reddit Indonesia Sarcastic | `w11wo/reddit_indonesia_sarcastic` | 14,116 |
| Twitter Indonesia Sarcastic | `w11wo/twitter_indonesia_sarcastic` | 2,684 |

Script: `scripts/download_data.py`

## 2. Dataset Structure

### Reddit
- **Text column:** `text` (PII-masked, tag-removed)
- **Other columns:** `body`, `label`, `permalink`, `subreddit`, `lang_fastText`, `created_utc`, `author`, `score`
- **Split:** train 9,881 / val 1,411 / test 2,824

### Twitter
- **Text column:** `tweet` (PII-masked)
- **Other columns:** `label`
- **Split:** train 1,878 / val 268 / test 538

## 3. Clarification: Twitter Dataset Size

Paper menyebut Twitter dataset = 12,861, tapi HuggingFace hanya 2,684.

Dari halaman HuggingFace:
- Total (raw) = 17,718
- Total (cleaned; unbalanced) = 12,861 (671 sarcastic + 12,190 non-sarcastic) ← angka di paper
- Total (cleaned; balanced) = 2,684 (671 sarcastic + 2,013 non-sarcastic, 1:3 ratio) ← versi HuggingFace

Paper meng-mention angka unbalanced di abstrak, tapi experiments menggunakan balanced version. Kita menggunakan balanced version yang benar.

## 4. EDA Results

### 4.1 Label Distribution

| Dataset | Label | Count | Ratio |
|---------|-------|-------|-------|
| Reddit | Non-sarcasm (0) | 10,587 | 75.00% |
| Reddit | Sarcasm (1) | 3,529 | 25.00% |
| Twitter | Non-sarcasm (0) | 2,013 | 75.00% |
| Twitter | Sarcasm (1) | 671 | 25.00% |

Class balance EXACT 25% sarcasm di kedua dataset, semua split. Kemungkinan sudah di-undersample oleh author (1:3 ratio following SemEval-2022 Task 6).

### 4.2 Per-Split Distribution

**Reddit:**
| Split | Total | Sarcasm | Non-sarcasm |
|-------|-------|---------|-------------|
| Train | 9,881 | 2,470 | 7,411 |
| Val | 1,411 | 353 | 1,058 |
| Test | 2,824 | 706 | 2,118 |

**Twitter:**
| Split | Total | Sarcasm | Non-sarcasm |
|-------|-------|---------|-------------|
| Train | 1,878 | 470 | 1,408 |
| Val | 268 | 67 | 201 |
| Test | 538 | 134 | 404 |

### 4.3 Text Length (character count)

| Dataset | Label | Mean | Std | Min | Max |
|---------|-------|------|-----|-----|-----|
| Reddit | Non-sarcasm | 103.6 | 88.9 | 4 | 1,134 |
| Reddit | Sarcasm | 67.3 | 47.7 | 5 | 527 |
| Twitter | Non-sarcasm | 113.8 | 67.8 | 14 | 584 |
| Twitter | Sarcasm | 117.8 | 55.0 | 18 | 297 |

**Insight:** Di Reddit, sarcastic comments jauh lebih pendek (mean 67 vs 104 char). Di Twitter, hampir sama (118 vs 114).

### 4.4 Data Quality

| Check | Reddit | Twitter |
|-------|--------|---------|
| Missing values | 0 | 0 |
| Duplicate texts | 10 | 0 |

## 5. Pre-processing dari Paper

Berdasarkan paper, preprocessing yang dilakukan author:

1. **Language filtering** (Reddit only): fastText language detection, keep Indonesian/Javanese/Minangkabau/Malay/Sundanese
2. **Near-deduplication:** MinHash LSH — mengurangi Twitter sarcastic dari 4,350 → 671
3. **PII masking:** username → `<username>`, hashtag → `<hashtag>`, email → `<email>`, URL → `<link>`
4. **Sarcasm tag removal** (Reddit only): hapus `/s` suffix
5. **Random sampling:** 1:3 ratio sarcastic:non-sarcastic
6. **Split:** 70% train / 10% val / 20% test

Kita tidak perlu reproduce preprocessing karena dataset sudah bersih di HuggingFace.

## 6. Column Names Penting

| Dataset | Text Column | Label Column |
|---------|-------------|--------------|
| Reddit | `text` | `label` |
| Twitter | `tweet` | `label` |

Untuk reproduksi, script paper menerima `--text_column_name` arg. Twitter butuh `--text_column_name tweet`.

## 7. Output Progress 2

- [x] Dataset downloaded ke `data/raw/`
- [x] EDA notebook: `notebooks/01_eda.ipynb`
- [x] Label distribution analysis
- [x] Text length analysis
- [x] Split distribution analysis
- [x] Data quality check
- [x] Clarification Twitter dataset size
- [x] Dokumentasi hasil EDA

## 8. Figures Generated

- `results/figures/label_distribution.png`
- `results/figures/text_length_distribution.png`
- `results/figures/split_distribution.png`
