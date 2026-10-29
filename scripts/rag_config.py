"""Settings shared by build_index.py and chat.py. Override with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_DB = Path(os.environ.get("RAG_INDEX", ROOT / "index" / "sources.db"))

GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "<your-project-id>")
GCP_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768
CHAT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


GEMINI_TIMEOUT_MS = 40_000


def genai_client():
    # The SDK has no timeout by default, so a pooled connection that went stale
    # while a Cloud Run instance sat idle could hang a request until the
    # service's own limit. With a timeout it fails fast and retry() can redo it.
    from google import genai
    from google.genai import types
    return genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION,
                        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))


def retry(call, attempts=2):
    """Run a Gemini call, once more if it times out or its connection drops."""
    import httpx
    for attempt in range(attempts):
        try:
            return call()
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == attempts - 1:
                raise
