"""Upload trained models to a Hugging Face model repo so the deployed app can download them.
Usage (after training):  huggingface-cli login   then   python upload_models.py your-hf-username/fake-news-models
"""
import sys
from huggingface_hub import HfApi

from config import OUT_DIR

repo = sys.argv[1]
api = HfApi()
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path=OUT_DIR, repo_id=repo, repo_type="model",
                  ignore_patterns=["bert_ckpt/*", "*.png", "*.csv"])
print(f"Uploaded {OUT_DIR}/ to https://huggingface.co/{repo}")
