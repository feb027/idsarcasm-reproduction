#!/usr/bin/env python3
"""
Download IdSarcasm datasets from HuggingFace and save to data/raw/

Usage:
    pip install datasets pandas
    python scripts/download_data.py
"""

from datasets import load_dataset
import pandas as pd
import os

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

DATASETS = {
    "reddit": "w11wo/reddit_indonesia_sarcastic",
    "twitter": "w11wo/twitter_indonesia_sarcastic",
}

for name, hf_id in DATASETS.items():
    print(f"Downloading {name} ({hf_id})...")
    ds = load_dataset(hf_id)

    for split in ds:
        df = pd.DataFrame(ds[split])
        out_path = os.path.join(RAW_DIR, f"{name}_{split}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path} ({len(df)} rows)")

print("\nDone. Files in data/raw/:")
for f in sorted(os.listdir(RAW_DIR)):
    size = os.path.getsize(os.path.join(RAW_DIR, f))
    print(f"  {f} ({size/1024:.1f} KB)")
