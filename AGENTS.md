# IdSarcasm Reproduction — Agent Instructions

## Project
Reproducing: "IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection" (Suhartono et al., IEEE Access 2024)

Paper DOI: 10.1109/ACCESS.2024.3416955
Original repo: https://github.com/w11wo/id_sarcasm

## Stack
- Python 3.10+
- PyTorch + HuggingFace Transformers
- scikit-learn, pandas, numpy
- Jupyter notebooks for EDA/analysis

## Build & Run
```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download data
python scripts/download_data.py

# Run EDA notebook
jupyter nbconvert --execute notebooks/01_eda.ipynb

# Run experiments
python scripts/train_model.py  # (when created)
```

## Code Standards
- Python 3.10+, type hints on all functions
- Use `ruff check .` for linting
- Use `ruff format .` for formatting
- Notebooks: save outputs, keep cells runnable
- Commit figures to `results/figures/` (NOT gitignored)

## Project Structure
```
data/raw/           — Raw CSV datasets from HuggingFace
data/processed/     — Preprocessed datasets
notebooks/          — Jupyter notebooks (numbered: 01_, 02_, ...)
scripts/            — Python scripts (download, train, evaluate)
results/tables/     — Evaluation results (CSV)
results/figures/    — Plots & visualizations (PNG)
docs/               — Progress documentation per milestone
```

## Conventions
- Datasets: Reddit col='text', Twitter col='tweet'
- Both datasets: 25% sarcasm ratio
- Paper uses 12.8k Twitter (unbalanced), we use 2.6k (balanced)
- Progress stages tracked in Google Sheets (6 stages)
- Figures committed to repo (not gitignored) — dosen requirement
- Comprehensive per-progress docs in `docs/`

## Git Rules
- Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`
- One logical change per commit
- Branch per feature/experiment
- Run tests before committing

## Testing
```bash
python -m pytest tests/ -q  # (when tests exist)
```
- Validate data preprocessing outputs before training
- Check notebook cells execute without errors

## Done Criteria
1. Code runs without errors
2. Notebooks execute fully
3. Results match paper's reported metrics (within tolerance)
4. Documentation updated in `docs/`
5. Figures saved to `results/figures/`
