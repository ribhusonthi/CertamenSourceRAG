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


def genai_client():
    from google import genai
    return genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
