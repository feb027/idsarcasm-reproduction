# Progress 3 — Paper Baseline Complete Plan

Dokumen ini adalah rencana eksekusi **paper baseline complete** untuk baseline fine-tuned transformer IdSarcasm. Tujuannya bukan hanya menjalankan dua model awal, tetapi berusaha mereproduksi seluruh tabel baseline transformer paper pada dua dataset: Twitter dan Reddit.

## 1. Sumber yang dicek

Sumber lokal:

- `paper.md` — paper IEEE Access IdSarcasm. File ini UTF-16; angka hasil juga diringkas di README source-code.
- `source-code/original-id-sarcasm/README.md` — tabel hasil paper.
- `source-code/original-id-sarcasm/scripts/run_classification.py` — script training asli berbasis HuggingFace `Trainer`.
- `source-code/original-id-sarcasm/recipes/{twitter,reddit}/baseline/*.sh` — recipe baseline transformer asli.

Riset kompatibilitas library:

- HuggingFace Transformers versi baru mengganti/deprecate beberapa API. `Trainer(tokenizer=...)` digantikan oleh `processing_class` pada versi baru.
- `TrainingArguments.overwrite_output_dir` juga dideprecate/dihapus pada versi Transformers baru.
- Karena Colab memakai library rolling/latest, runner repo perlu kompatibilitas dinamis. Patch compatibility sudah ditambahkan di `scripts/run_transformer_baseline.py`.

## 2. Target paper baseline complete

Tabel target F1 paper:

| Model | HF model | Reddit F1 paper | Twitter F1 paper |
|---|---|---:|---:|
| IndoBERT Base (IndoNLU) | `indobenchmark/indobert-base-p1` | 0.6100 | 0.7273 |
| IndoBERT Large (IndoNLU) | `indobenchmark/indobert-large-p1` | 0.6184 | 0.7160 |
| IndoBERT Base (IndoLEM) | `indolem/indobert-base-uncased` | 0.5671 | 0.6462 |
| mBERT | `bert-base-multilingual-cased` | 0.5338 | 0.6467 |
| XLM-R Base | `xlm-roberta-base` | 0.5690 | 0.7386 |
| XLM-R Large | `xlm-roberta-large` | 0.6274 | 0.7692 |

Total run baseline transformer: **6 model × 2 dataset = 12 run**.

## 3. Setting paper-faithful dari recipe original

Semua recipe baseline original memakai setting utama yang sama:

| Komponen | Nilai |
|---|---:|
| Max sequence length | 128 |
| Train batch size | 32 |
| Eval batch size | 64 |
| Learning rate | 1e-5 |
| LR scheduler | cosine |
| Weight decay | 0.03 |
| Label smoothing | 0.0 |
| Epoch maksimum | 100 |
| Shuffle train split | yes |
| Early stopping | patience 3, threshold 0.01 di script original |
| Metric best model | F1 |
| Seed | 42 |
| FP16 | yes |

Perbedaan repo UAS yang disengaja:

- tidak `push_to_hub`;
- hasil disimpan ke `results/tables/transformer_baselines.csv` dan `results/transformer/...`;
- checkpoint besar disimpan di `models/` dan tidak di-commit;
- log Colab disimpan di `results/logs/` sebagai bukti.

## 4. Status saat ini

Sudah selesai dan ter-commit:

| Dataset | Model | F1 hasil | Status |
|---|---|---:|---|
| Twitter | IndoBERT Base | 0.6812 | selesai |
| Twitter | XLM-R Base | 0.7000 | selesai |

Belum dijalankan:

| Dataset | Model |
|---|---|
| Twitter | IndoBERT Large |
| Twitter | IndoBERT Base IndoLEM |
| Twitter | mBERT |
| Twitter | XLM-R Large |
| Reddit | IndoBERT Base |
| Reddit | IndoBERT Large |
| Reddit | IndoBERT Base IndoLEM |
| Reddit | mBERT |
| Reddit | XLM-R Base |
| Reddit | XLM-R Large |

## 5. Environment Colab yang disarankan

### Opsi A — lanjut dengan environment Colab saat ini

Ini paling praktis. Runner sudah punya compatibility filter untuk API Transformers baru.

Cek versi dan GPU:

```bash
!python - <<'PY'
import torch, transformers, datasets, accelerate
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
print('accelerate', accelerate.__version__)
print('cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY
```

### Opsi B — pinned environment untuk hasil lebih stabil

Jika ingin mengurangi perbedaan API dan nondeterminism library, di runtime baru jalankan:

```bash
!pip install -q "transformers==4.44.2" "accelerate==0.33.0" "datasets>=2.14" "evaluate>=0.4"
```

Lalu restart runtime. Catat versi di log sebelum training.

Catatan: hasil tetap belum dijamin identik dengan paper karena GPU/fp16 dan versi PyTorch/CUDA bisa berbeda.

## 6. Urutan eksekusi yang disarankan

Prioritas agar coverage cepat naik:

1. Reddit IndoBERT Base
2. Reddit XLM-R Base
3. Twitter mBERT
4. Twitter IndoBERT Base IndoLEM
5. Reddit mBERT
6. Reddit IndoBERT Base IndoLEM
7. Twitter IndoBERT Large
8. Reddit IndoBERT Large
9. Twitter XLM-R Large
10. Reddit XLM-R Large

Alasan: base models lebih mungkin selesai di Colab T4/L4; large models paling rawan OOM.

## 7. Template command paper-faithful

Gunakan format ini untuk setiap run agar log tersimpan:

```bash
!mkdir -p results/logs

!python scripts/run_transformer_baseline.py \
  --dataset DATASET \
  --model MODEL_ALIAS \
  --epochs 100 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --learning-rate 1e-5 \
  --lr-scheduler-type cosine \
  --weight-decay 0.03 \
  --label-smoothing-factor 0.0 \
  --max-length 128 \
  --early-stopping-threshold 0.01 \
  --seed 42 \
  --pad-to-max-length \
  --shuffle-train-dataset \
  --fp16 \
  --output-dir results/transformer/DATASET-MODEL_ALIAS \
  --model-output-dir models/transformer/DATASET-MODEL_ALIAS \
  2>&1 | tee results/logs/progress-3-DATASET-MODEL_ALIAS.log
```

Alias yang tersedia di runner:

| Alias | HF model |
|---|---|
| `indobert-base` | `indobenchmark/indobert-base-p1` |
| `indobert-large` | `indobenchmark/indobert-large-p1` |
| `indobert-indolem-base` | `indolem/indobert-base-uncased` |
| `mbert-base` | `bert-base-multilingual-cased` |
| `xlmr-base` | `xlm-roberta-base` |
| `xlmr-large` | `xlm-roberta-large` |

## 8. Full command list — 12 baseline runs

### Twitter

```bash
# Twitter IndoBERT Base (already done)
!python scripts/run_transformer_baseline.py --dataset twitter --model indobert-base --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --output-dir results/transformer/twitter-indobert-base --model-output-dir models/transformer/twitter-indobert-base 2>&1 | tee results/logs/progress-3-twitter-indobert-base.log

# Twitter IndoBERT Large
!python scripts/run_transformer_baseline.py --dataset twitter --model indobert-large --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --output-dir results/transformer/twitter-indobert-large --model-output-dir models/transformer/twitter-indobert-large 2>&1 | tee results/logs/progress-3-twitter-indobert-large.log

# Twitter IndoBERT Base IndoLEM
!python scripts/run_transformer_baseline.py --dataset twitter --model indobert-indolem-base --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --output-dir results/transformer/twitter-indobert-indolem-base --model-output-dir models/transformer/twitter-indobert-indolem-base 2>&1 | tee results/logs/progress-3-twitter-indobert-indolem-base.log

# Twitter mBERT
!python scripts/run_transformer_baseline.py --dataset twitter --model mbert-base --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --output-dir results/transformer/twitter-mbert-base --model-output-dir models/transformer/twitter-mbert-base 2>&1 | tee results/logs/progress-3-twitter-mbert-base.log

# Twitter XLM-R Base (already done)
!python scripts/run_transformer_baseline.py --dataset twitter --model xlmr-base --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --output-dir results/transformer/twitter-xlmr-base --model-output-dir models/transformer/twitter-xlmr-base 2>&1 | tee results/logs/progress-3-twitter-xlmr-base.log

# Twitter XLM-R Large
!python scripts/run_transformer_baseline.py --dataset twitter --model xlmr-large --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --output-dir results/transformer/twitter-xlmr-large --model-output-dir models/transformer/twitter-xlmr-large 2>&1 | tee results/logs/progress-3-twitter-xlmr-large.log
```

### Reddit

```bash
# Reddit IndoBERT Base
!python scripts/run_transformer_baseline.py --dataset reddit --model indobert-base --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --output-dir results/transformer/reddit-indobert-base --model-output-dir models/transformer/reddit-indobert-base 2>&1 | tee results/logs/progress-3-reddit-indobert-base.log

# Reddit IndoBERT Large
!python scripts/run_transformer_baseline.py --dataset reddit --model indobert-large --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --output-dir results/transformer/reddit-indobert-large --model-output-dir models/transformer/reddit-indobert-large 2>&1 | tee results/logs/progress-3-reddit-indobert-large.log

# Reddit IndoBERT Base IndoLEM
!python scripts/run_transformer_baseline.py --dataset reddit --model indobert-indolem-base --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --output-dir results/transformer/reddit-indobert-indolem-base --model-output-dir models/transformer/reddit-indobert-indolem-base 2>&1 | tee results/logs/progress-3-reddit-indobert-indolem-base.log

# Reddit mBERT
!python scripts/run_transformer_baseline.py --dataset reddit --model mbert-base --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --output-dir results/transformer/reddit-mbert-base --model-output-dir models/transformer/reddit-mbert-base 2>&1 | tee results/logs/progress-3-reddit-mbert-base.log

# Reddit XLM-R Base
!python scripts/run_transformer_baseline.py --dataset reddit --model xlmr-base --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --output-dir results/transformer/reddit-xlmr-base --model-output-dir models/transformer/reddit-xlmr-base 2>&1 | tee results/logs/progress-3-reddit-xlmr-base.log

# Reddit XLM-R Large
!python scripts/run_transformer_baseline.py --dataset reddit --model xlmr-large --epochs 100 --batch-size 32 --eval-batch-size 64 --learning-rate 1e-5 --lr-scheduler-type cosine --weight-decay 0.03 --label-smoothing-factor 0.0 --max-length 128 --early-stopping-threshold 0.01 --seed 42 --pad-to-max-length --shuffle-train-dataset --fp16 --output-dir results/transformer/reddit-xlmr-large --model-output-dir models/transformer/reddit-xlmr-large 2>&1 | tee results/logs/progress-3-reddit-xlmr-large.log
```

## 9. Jika XLM-R Large / IndoBERT Large OOM

Pertama, simpan log OOM apa adanya. Itu valid sebagai bukti run tidak feasible di resource Colab tertentu.

Jika ingin attempt fallback, gunakan command non-paper-faithful dan beri suffix output berbeda:

```bash
!python scripts/run_transformer_baseline.py \
  --dataset twitter \
  --model xlmr-large \
  --epochs 100 \
  --batch-size 4 \
  --eval-batch-size 8 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing \
  --learning-rate 1e-5 \
  --lr-scheduler-type cosine \
  --weight-decay 0.03 \
  --label-smoothing-factor 0.0 \
  --max-length 128 \
  --early-stopping-threshold 0.01 \
  --seed 42 \
  --pad-to-max-length \
  --shuffle-train-dataset \
  --fp16 \
  --output-dir results/transformer/twitter-xlmr-large-colab-fallback \
  --model-output-dir models/transformer/twitter-xlmr-large-colab-fallback \
  2>&1 | tee results/logs/progress-3-twitter-xlmr-large-colab-fallback.log
```

Catatan laporan wajib: fallback ini tidak memakai batch fisik paper (`32`) sehingga bukan reproduksi strict; hanya attempt resource-aware.

## 10. Checklist setelah setiap run

```bash
!tail -80 results/logs/progress-3-DATASET-MODEL_ALIAS.log
!cat results/transformer/DATASET-MODEL_ALIAS/result_row.json
!cat results/tables/transformer_baselines.csv
```

Pastikan:

- `sample_limited` = `false` untuk full run;
- tidak ada secret/token di log;
- jika run gagal/OOM, log tetap disimpan dan diberi status gagal di laporan.

Cek token:

```bash
!grep -RniE "hf_[A-Za-z0-9]|password|secret|api_key|apikey" results/logs || true
```

## 11. Commit hasil

```bash
!git add results/tables/transformer_baselines.csv results/transformer/ results/logs/
!git commit -m "results: extend Progress 3 paper baseline transformer runs"
!git push
```

Folder `models/` tetap tidak di-commit.

## 12. Kriteria Progress 3 paper baseline complete

Minimal untuk klaim paper baseline complete attempt:

- 12 command baseline dicoba atau dijalankan;
- run sukses menyimpan `result_row.json` dan `metrics.json`;
- run gagal/OOM menyimpan log kegagalan di `results/logs/`;
- tabel laporan membedakan `success`, `failed_oom`, dan `fallback_non_paper_batch`;
- semua hasil dibandingkan dengan target F1 paper.

Jika semua 12 sukses, Progress 3 dapat disebut **paper baseline complete**. Jika large models gagal karena Colab resource, Progress 3 dapat disebut **paper baseline complete attempt with documented resource failures**.
