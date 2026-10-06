# Fake News Detection (Hindi + English) — RNN vs LSTM vs BERT vs LLM

Input: news text or claim in Hindi or English (language auto-detected). UI is in English, with login.
Output: FAKE / REAL, calibrated confidence score, per-model comparison, and matching sources (fact-checks + news) with links.

## Pipeline

```
Hindi/English text ──► detect language ──► clean ──► RNN / LSTM / MuRIL-BERT / Qwen LLM ──► P(fake), temperature-calibrated
           └─► Google Fact Check API + Google News (same language) ──► similarity filter ──► evidence verdict
                                     final score = weighted(BERT, evidence)  ──► verdict + confidence + sources
```

| Model | What it is | Training |
|---|---|---|
| RNN | BiRNN + word embeddings | trained from scratch |
| LSTM | BiLSTM + word embeddings | trained from scratch |
| BERT | `google/muril-base-cased` (BERT pre-trained on 17 Indian languages) | fine-tuned |
| LLM | `Qwen/Qwen2.5-1.5B-Instruct`, zero-shot prompt | none |

## Validation

- **Model validation:** 70/15/15 train/val/test split, stratified on label and language, early stopping on val, final numbers only on the untouched test set. Metrics: accuracy, macro precision/recall/F1, ROC-AUC, confusion matrix — overall and separately for Hindi and English.
- **Confidence validation:** temperature scaling fitted on the val set, then Expected Calibration Error (ECE) on test. Low ECE = a "90% confident" prediction really is right ~90% of the time.
- **Claim validation:** the app checks the claim against published fact-checks (BOOM, Vishvas News, Alt News, PIB Fact Check etc. via Google Fact Check Tools API) and news coverage in the same language. Sources are kept only if semantic similarity ≥ 0.60.

## Setup (Google Colab with T4 GPU recommended)

```bash
pip install -r requirements-train.txt   # training machine / Colab
```

### 1. Data
You need both a Hindi and an English labelled dataset. Build `data/hindi_news.csv` (`text,label,lang`, 1 = FAKE):

```bash
# Hindi — CONSTRAINT@AAAI-2021 Hindi ('fake' vs 'non-hostile')
python prepare_data.py --files hindi_train.csv hindi_valid.csv
# English — e.g. CONSTRAINT@AAAI-2021 English COVID fake news (tweet, label = fake/real)
python prepare_data.py --files english_train.csv english_val.csv --text_col tweet --label_col label \
    --fake_values fake --real_values real --append
```
Any other dataset works the same way with its own `--text_col / --label_col / --fake_values / --real_values`. Keep the two languages roughly balanced.

### 2. Train
```bash
python train_rnn_lstm.py both     # ~minutes
python train_bert.py              # ~20–40 min on T4
```

### 3. Compare
```bash
python compare.py --models rnn,lstm,bert,llm --llm_samples 300
```
Writes `outputs/results.csv`, `outputs/results_by_language.csv`, `outputs/comparison.png`, `outputs/confusion_matrices.png`.

### 4. Sources (optional but recommended)
Google Cloud Console → enable **Fact Check Tools API** → create API key:
```bash
export FACTCHECK_API_KEY=your_key
```
Without it, only Google News coverage is used.

Login accounts are stored in `users.db` (SQLite, salted PBKDF2 hashes).

### 5. App
```bash
streamlit run app.py
```
On Colab: `!streamlit run app.py & npx localtunnel --port 8501`

## Deploy

See [DEPLOY.md](DEPLOY.md): GitHub for code, Hugging Face Hub (free model storage) for trained models, Streamlit Community Cloud (free) for the live app.

## Files
| File | Purpose |
|---|---|
| `config.py` | paths, model names, thresholds |
| `prepare_data.py` | raw dataset → `text,label` CSV |
| `data.py` | cleaning, language detection, splits |
| `models_rnn.py` / `train_rnn_lstm.py` | RNN & LSTM |
| `train_bert.py` | MuRIL fine-tuning |
| `llm_zeroshot.py` | LLM baseline |
| `calibration.py` | temperature scaling + ECE |
| `predict.py` | unified calibrated predictor |
| `compare.py` | metrics, charts, confusion matrices |
| `evidence.py` | fact-check + news retrieval, source matching |
| `auth.py` | login / registration |
| `app.py` | Streamlit UI: login page + dashboard |
| `ui.py`, `.streamlit/config.toml` | styling, theme and UI components |

## Expected outcome
Fine-tuned BERT should beat RNN/LSTM on F1 and AUC: MuRIL understands both Hindi and English context and handles unseen words via subwords. RNN/LSTM usually show a bigger gap between the two languages. The zero-shot LLM typically trails fine-tuned BERT because it has no dataset-specific training — that gap is itself a result worth reporting.
