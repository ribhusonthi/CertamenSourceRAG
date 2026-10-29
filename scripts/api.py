#!/usr/bin/env python3
"""Source-check API for certamenreader (runs as its own Cloud Run service).

certamenreader's server calls this; browsers never do. Responses carry only the
cited sentence plus one sentence of context on each side, never whole passages or
PDFs, so the textbooks themselves are not redistributed.

  POST /ask      {"message": str, "history": [[question, answer], ...]}
                 header X-Sources-Secret must equal $SOURCES_SECRET
  GET  /healthz

Run locally:  python3 scripts/api.py            (127.0.0.1:8771, no secret)
On Cloud Run: PORT and SOURCES_SECRET are set; it then listens on 0.0.0.0.
"""
import hmac
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat import Chat, Index  # noqa: E402

SECRET = os.environ.get("SOURCES_SECRET", "")
MAX_MESSAGE = 2000
MAX_CONTEXT = 300  # characters of the neighbouring sentence shown on each side

index = None


def neighbour_sentences(text, start, end):
    """The sentence before and after text[start:end], trimmed to MAX_CONTEXT."""
    before = re.split(r"(?<=[.!?])\s+", text[:start].strip())[-1] if text[:start].strip() else ""
    after = re.split(r"(?<=[.!?])\s+", text[end:].strip())[0] if text[end:].strip() else ""
    if len(before) > MAX_CONTEXT:
        before = "…" + before[-MAX_CONTEXT:]
    if len(after) > MAX_CONTEXT:
        after = after[:MAX_CONTEXT] + "…"
    return before, after


def public_citation(c):
    text, start, end = c["text"], c["quote_start"], c["quote_end"]
    if end <= start:  # quote not located: show the opening sentence only
        end = len(re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0])
        start = 0
    before, after = neighbour_sentences(text, start, end)
    return {
        "n": c["n"], "cite": c["cite"], "page_ref": c["page_ref"],
        "quote": text[start:end], "before": before, "after": after,
        "exact": c["exact"], "alternate": c["alternate"],
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            self.send_json(200, {"ok": True})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/ask":
            return self.send_error(404)
        if SECRET and not hmac.compare_digest(self.headers.get("X-Sources-Secret", ""), SECRET):
            return self.send_json(403, {"error": "forbidden"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            message = str(req.get("message", "")).strip()[:MAX_MESSAGE]
            history = [(str(u)[:MAX_MESSAGE], str(a)[:8000]) for u, a in req.get("history", [])][-3:]
            if not message:
                return self.send_json(400, {"error": "empty message"})
            result = Chat(index, history).ask_cited(message)
            self.send_json(200, {"answer": result["answer"],
                                 "citations": [public_citation(c) for c in result["citations"]]})
        except Exception as e:  # report model/API failures to the caller instead of dropping the request
            print(f"ask failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            self.send_json(500, {"error": "the source check failed; try again"})

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}\n")


def main():
    global index
    index = Index()
    # Without a secret the API is open, so only listen on this machine.
    host = "0.0.0.0" if SECRET else "127.0.0.1"
    port = int(os.environ.get("PORT", 8771))
    print(f"source-check API on http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
