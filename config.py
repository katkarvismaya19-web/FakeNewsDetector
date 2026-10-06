import os

DATA_PATH = os.getenv("DATA_PATH", "data/hindi_news.csv")   # columns: text,label  (1 = FAKE, 0 = REAL)
TEXT_COL, LABEL_COL = "text", "label"
OUT_DIR = "outputs"
SEED = 42
MAX_LEN = 128

BERT_MODEL = "google/muril-base-cased"                     # BERT pre-trained on 17 Indian languages + English
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")  # zero-shot LLM baseline
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # for source matching

FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY", "")     # Google Fact Check Tools API key
SIM_THRESHOLD = 0.60                                         # min similarity for a source to count as "same claim"

os.makedirs(OUT_DIR, exist_ok=True)

# Deployment: trained models live in a Hugging Face model repo (too big for GitHub)
MODEL_REPO = os.getenv("MODEL_REPO", "")        # e.g. "your-hf-username/fake-news-models"
ENABLE_LLM = os.getenv("ENABLE_LLM", "1") == "1"
LOW_MEMORY = os.getenv("LOW_MEMORY", "0") == "1"   # load BERT in bfloat16 (halves RAM) for free hosting
TFIDF_THRESHOLD = 0.30                              # source-match threshold when sentence-transformers isn't installed
