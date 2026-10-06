"""Build data/hindi_news.csv (text,label,lang) from Hindi and English datasets. 1 = FAKE, 0 = REAL.

Run once per dataset; use --append for the second one.

  # Hindi: CONSTRAINT-2021 Hindi ('fake' vs 'non-hostile')
  python prepare_data.py --files hindi_train.csv hindi_valid.csv
  # English: CONSTRAINT-2021 English COVID fake news (tweet, label = fake/real)
  python prepare_data.py --files english_train.csv english_val.csv --text_col tweet --label_col label \
      --fake_values fake --real_values real --append
"""
import argparse, os
import pandas as pd
from data import detect_lang

ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--text_col", default="Post")
ap.add_argument("--label_col", default="Labels Set")
ap.add_argument("--fake_values", nargs="+", default=["fake"])
ap.add_argument("--real_values", nargs="+", default=["non-hostile"])
ap.add_argument("--out", default="data/hindi_news.csv")
ap.add_argument("--append", action="store_true", help="add to existing output instead of overwriting")
a = ap.parse_args()

df = pd.concat([pd.read_csv(f) for f in a.files], ignore_index=True)
missing = {a.text_col, a.label_col} - set(df.columns)
if missing:
    raise SystemExit(f"Columns {missing} not found. Available: {list(df.columns)}")

fake, real = {v.lower() for v in a.fake_values}, {v.lower() for v in a.real_values}

def to_label(v):
    tags = {t.strip().lower() for t in str(v).split(",")}
    if tags & fake:
        return 1
    if tags <= real:
        return 0
    return None

df["label"] = df[a.label_col].map(to_label)
df = df.dropna(subset=["label"]).rename(columns={a.text_col: "text"})[["text", "label"]]
df["label"] = df["label"].astype(int)
df["lang"] = df["text"].map(detect_lang)

if a.append and os.path.exists(a.out):
    df = pd.concat([pd.read_csv(a.out), df], ignore_index=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
df.to_csv(a.out, index=False)
print(f"Saved {len(df)} rows to {a.out}")
print(pd.crosstab(df["lang"], df["label"].map({1: "FAKE", 0: "REAL"})))
