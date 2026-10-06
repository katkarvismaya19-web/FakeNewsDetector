import re
import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_PATH, TEXT_COL, LABEL_COL, SEED


def clean_text(t: str) -> str:
    """Works for Hindi (Devanagari) and English."""
    t = str(t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+|#", " ", t)
    t = re.sub(r"[^\u0900-\u097F\sA-Za-z0-9?!,.']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def devanagari_ratio(t: str) -> float:
    letters = [c for c in t if c.isalpha()]
    return sum("\u0900" <= c <= "\u097F" for c in letters) / max(len(letters), 1)


def detect_lang(t: str) -> str:
    return "hi" if devanagari_ratio(t) >= 0.3 else "en"


def load_splits():
    """70 / 15 / 15 split, stratified on label AND language so both languages appear in every split."""
    df = pd.read_csv(DATA_PATH)
    df = df[[c for c in [TEXT_COL, LABEL_COL, "lang"] if c in df.columns]].dropna(subset=[TEXT_COL, LABEL_COL])
    df[TEXT_COL] = df[TEXT_COL].map(clean_text)
    df = df[df[TEXT_COL].str.len() > 10].drop_duplicates(TEXT_COL)
    df["lang"] = df[TEXT_COL].map(detect_lang)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    strat = df[LABEL_COL].astype(str) + "_" + df["lang"]
    train, temp = train_test_split(df, test_size=0.30, stratify=strat, random_state=SEED)
    strat_t = temp[LABEL_COL].astype(str) + "_" + temp["lang"]
    val, test = train_test_split(temp, test_size=0.50, stratify=strat_t, random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
