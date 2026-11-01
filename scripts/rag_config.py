"""Settings shared by build_index.py and chat.py. Override with environment variables."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_DB = Path(os.environ.get("RAG_INDEX", ROOT / "index" / "sources.db"))

def _gcloud_project():
    try:
        out = subprocess.run(["gcloud", "config", "get-value", "project"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or _gcloud_project()
GCP_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768
CHAT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


# Which books to trust first. Primary sourcebooks are checked before secondary
# ones, and the semi-official books are a last resort. Keys are source ids from
# extract_pages.py; edit freely.
PRIMARY, SECONDARY, SEMI_OFFICIAL = 1, 2, 3
SOURCE_TIERS = {
    # Mythology: Tripp first, then the others.
    "tripp": PRIMARY,
    "morford-lenardon": SECONDARY,
    "march": SECONDARY,
    "grimal": SEMI_OFFICIAL,
    # History: the two textbooks, then the chronicles.
    "cary-scullard": PRIMARY,
    "ward-heichelheim": PRIMARY,
    "matyszak-republic": SECONDARY,
    "scarre-emperors": SECONDARY,
    # Literature
    "conte": PRIMARY,
    "hadas": SECONDARY,
    # Life and culture: the two textbooks, then the archaeological guide.
    "adkins-handbook": PRIMARY,
    "johnston-private-life": PRIMARY,
    "claridge-rome": SECONDARY,
    # Language: the NJCL-listed grammar and phrase books, then the vocabulary lists.
    "allen-greenough": PRIMARY,
    "ehrlich-amo-amas-amat": PRIMARY,
    "lodge-vocabulary": SECONDARY,
    "brunel-vocabulary": SECONDARY,
    "schaeffer-derivatives": SECONDARY,
    "greenwald-derivatives": SECONDARY,
}
TIER_NAMES = {PRIMARY: "primary", SECONDARY: "secondary", SEMI_OFFICIAL: "semi-official"}
# Search score multiplier per tier: a lower-tier passage has to match clearly
# better to outrank a primary one.
TIER_WEIGHTS = {PRIMARY: 1.0, SECONDARY: 0.8, SEMI_OFFICIAL: 0.6}


def source_tier(source_id):
    return SOURCE_TIERS.get(source_id, SECONDARY)


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
