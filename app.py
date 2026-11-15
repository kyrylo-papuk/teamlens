"""Entrypoint for the public demo on Vercel. Everything else starts with `python -m teamlens.web`.

Vercel runs the app as a function: the filesystem is read-only apart from the temp folder, and an idle
instance is recycled, which takes the in-memory company and the files below with it. That suits a demo on
fictional data and nothing else; the notice on the page says so. A deployment that has to keep its state
runs the container (Dockerfile, docker-compose.yml).
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

os.environ.setdefault("TEAMLENS_FAKE", "1")  # the fictional company; no real Jira or people behind the demo
os.environ.setdefault("TEAMLENS_DAILY_MODEL_REQUESTS", "200")  # no login, so cap what a visitor can spend
os.environ.setdefault("TEAMLENS_DATA_DIR", str(Path(tempfile.gettempdir()) / "teamlens"))
os.environ.setdefault("TEAMLENS_NOTICE", "Public demo on a fictional company. Data is shared between visitors "
                                         "and resets after a period of inactivity.")

from teamlens.web.app import create_app  # noqa: E402

app = create_app()
