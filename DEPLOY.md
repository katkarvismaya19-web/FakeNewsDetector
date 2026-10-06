# Download → GitHub → Live app (100% free)

| What | Where | Cost |
|---|---|---|
| Code | GitHub | free |
| Trained models (~1 GB, too big for GitHub) | Hugging Face model repo | free |
| Live app | Streamlit Community Cloud | free |

## 1. Hugging Face token
https://huggingface.co/settings/tokens → **Create new token** → type **Write** → name `fake-news-deploy` → copy (`hf_…`).

## 2. GitHub repo
https://github.com/new → name `fake-news-detector` → **Public** → no README / .gitignore / license → **Create repository**.

## 3. GitHub token (your password for git push)
GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate new token:
name `fake-news-push`, expiration 90 days, **Only select repositories** → `fake-news-detector`,
permission **Contents: Read and write** → Generate → copy (`github_pat_…`).

## 4. Push the code
Inside the unzipped `fake-news-detector` folder:
```bash
git init
git add .
git commit -m "Initial commit: fake news detector"
git branch -M main
git remote add origin https://github.com/<github-username>/fake-news-detector.git
git push -u origin main
```
Username = your GitHub username, password = the `github_pat_…` token.

## 5. Train on Colab and upload models
Colab → New notebook → rename `fake_news_training.ipynb` → Runtime → Change runtime type → **T4 GPU**.
Name your datasets `hindi_train.csv`, `hindi_valid.csv`, `english_train.csv`, `english_val.csv`.

```python
!git clone https://github.com/<github-username>/fake-news-detector.git
%cd fake-news-detector
!pip install -q -r requirements-train.txt
```
```python
from google.colab import files
files.upload()   # pick the 4 CSVs
```
```python
!python prepare_data.py --files hindi_train.csv hindi_valid.csv
!python prepare_data.py --files english_train.csv english_val.csv --text_col tweet --label_col label --fake_values fake --real_values real --append
!python train_rnn_lstm.py both
!python train_bert.py
!python compare.py --models rnn,lstm,bert,llm
```
```python
from huggingface_hub import login
login()          # paste hf_… token
!python upload_models.py <hf-username>/fake-news-models
```
```python
!zip -r results.zip outputs/results.csv outputs/results_by_language.csv outputs/comparison.png outputs/confusion_matrices.png
files.download("results.zip")
```

## 6. Deploy on Streamlit Community Cloud
1. https://share.streamlit.io → **Continue with GitHub** → authorize.
2. **Create app** → **Deploy a public app from GitHub**.

| Field | Enter |
|---|---|
| Repository | `<github-username>/fake-news-detector` |
| Branch | `main` |
| Main file path | `app.py` |
| App URL | `fake-news-detector-<hf-username>` (any free name) |

3. **Advanced settings**:
   - Python version: **3.11**
   - Secrets (paste exactly, with your values):
```toml
MODEL_REPO = "<hf-username>/fake-news-models"
HF_TOKEN = "hf_your_token_here"
LOW_MEMORY = "1"
ENABLE_LLM = "0"
FACTCHECK_API_KEY = ""
```
4. **Save** → **Deploy**. First build takes ~5–10 minutes.

Every `git push` to `main` redeploys automatically.

## Limits (free tier)
- RAM is capped, so the app runs BERT in low-memory mode and the LLM option is hidden. The LLM is still compared in `compare.py` on Colab.
- Without `sentence-transformers` installed, source matching uses character n-gram similarity, so Hindi claims match Hindi sources and English match English.
- Apps sleep after ~12 hours without visitors; the next visitor clicks "wake up" and waits ~1 minute.
- Login accounts are stored in a local file and reset when the app restarts or redeploys.

## Optional: Google Fact Check API key
console.cloud.google.com → New project `fake-news-detector` → APIs & Services → Library → **Fact Check Tools API** → Enable →
Credentials → Create credentials → API key → restrict to Fact Check Tools API → copy.
Streamlit app → ⋮ → **Settings** → **Secrets** → set `FACTCHECK_API_KEY = "your_key"` → Save.
