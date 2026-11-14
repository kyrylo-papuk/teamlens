"""Deploy the public demo to a Hugging Face Space (Docker).

    hf auth login                               once, with a token that can write
    python scripts/deploy_space.py              creates or updates <you>/teamlens
    python scripts/deploy_space.py --with-key   also stores ANTHROPIC_API_KEY from .env as a Space secret

A Space keeps running while it has visitors and only pauses after about two days without any, which suits a
demo better than hosts that sleep after minutes. The demo runs the fictional company in one container, with
the daily cap on model requests and the notice line described in the README.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
NAME = "teamlens"
VARIABLES = {
    "TEAMLENS_FAKE": "1",
    "TEAMLENS_DAILY_MODEL_REQUESTS": "200",
    "TEAMLENS_NOTICE": "Public demo on a fictional company. Data is shared between visitors and resets when the Space restarts.",
}
CARD = """---
title: TeamLens
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: Agentic delivery assistant over Jira and HR data
---

# TeamLens

A public demo of [TeamLens](https://github.com/kyrylo-papuk/teamlens): an agentic assistant for an engineering
team's delivery. It runs on a fictional company held in memory; nothing here is a real person or a real Jira.

Source, architecture and evaluation results: https://github.com/kyrylo-papuk/teamlens
"""


def retry(what: str, job, attempts: int = 6):
    for n in range(1, attempts + 1):
        try:
            return job()
        except Exception as exc:  # the connection to the Hub drops now and then; the calls are safe to repeat
            if n == attempts:
                raise
            print(f"  {what}: attempt {n} failed ({type(exc).__name__}), retrying")
            time.sleep(4)


def main() -> None:
    api = HfApi()
    user = retry("whoami", api.whoami)["name"]
    repo = f"{user}/{NAME}"
    retry("create space", lambda: api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True))
    for key, value in VARIABLES.items():
        retry(f"variable {key}", lambda k=key, v=value: api.add_space_variable(repo, k, v))
    if "--with-key" in sys.argv:
        from teamlens.config import Settings

        key = Settings(_env_file=ROOT / ".env").anthropic_api_key
        if not key:
            sys.exit("ANTHROPIC_API_KEY is not set in .env")
        retry("secret", lambda: api.add_space_secret(repo, "ANTHROPIC_API_KEY", key))
        print("  ANTHROPIC_API_KEY stored as a Space secret")

    with tempfile.TemporaryDirectory() as tmp:
        build = Path(tmp)
        for name in ("Dockerfile", "pyproject.toml"):
            shutil.copy(ROOT / name, build / name)
        shutil.copytree(ROOT / "src", build / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        (build / "README.md").write_text(CARD, encoding="utf-8")
        retry("upload", lambda: api.upload_folder(folder_path=str(build), repo_id=repo, repo_type="space",
                                                  commit_message="Deploy TeamLens demo"))
    print(f"deployed: https://huggingface.co/spaces/{repo}")
    print(f"the app:  https://{repo.replace('/', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
