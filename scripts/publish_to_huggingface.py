"""Script to upload Ari-LLM RewriteBot 135M to Hugging Face Hub."""

import os
import sys
from huggingface_hub import HfApi, login

CHECKPOINT_DIR = "artifacts/checkpoints/ari_rewritebot_135m_dpo"
DEFAULT_REPO_ID = "sriyuthlucky/ari-rewritebot-135m-dpo"

def publish(repo_id=DEFAULT_REPO_ID, token=None):
    if token is None:
        token = os.environ.get("HF_TOKEN")
        
    if not token:
        print("\n" + "=" * 60)
        print(" Hugging Face Authentication Required")
        print("=" * 60)
        print("Please obtain a WRITE token from https://huggingface.co/settings/tokens")
        token = input("Enter your Hugging Face write token: ").strip()

    print(f"\nAuthenticating with Hugging Face...")
    login(token=token)
    api = HfApi()

    print(f"Creating / verifying repository: {repo_id}...")
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)

    print(f"Uploading model checkpoint folder ({CHECKPOINT_DIR}) to {repo_id}...")
    api.upload_folder(
        folder_path=CHECKPOINT_DIR,
        repo_id=repo_id,
        commit_message="Initial release: Ari-LLM RewriteBot 135M DPO-Aligned Paraphraser"
    )

    print("\n" + "=" * 60)
    print(f" SUCCESSFULLY PUBLISHED TO HUGGING FACE!")
    print(f" Model Page: https://huggingface.co/{repo_id}")
    print("=" * 60)

if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO_ID
    publish(repo_id=repo)
