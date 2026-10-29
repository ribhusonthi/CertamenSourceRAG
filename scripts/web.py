#!/usr/bin/env python3
"""Local web UI for the source chatbot.

  python3 scripts/web.py            # then open http://127.0.0.1:8770

Routes:
  GET  /                 the chat page (web/index.html)
  POST /api/ask          {"message": str, "history": [[question, answer], ...]} -> cited answer
  GET  /pdf/<source_id>  the textbook PDF, for "open page" links (#page=N)
"""
import argparse
import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat import Chat, Index  # noqa: E402
from extract_pages import BOOKS, find_pdf  # noqa: E402
from rag_config import ROOT  # noqa: E402

PAGE = ROOT / "web" / "index.html"
PDFS = {b["id"]: b["match"] for b in BOOKS}

index = None
index_lock = threading.Lock()  # one SQLite connection is shared by the request threads


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/pdf/") and self.path[5:] in PDFS:
            pdf = find_pdf(PDFS[self.path[5:]])
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(pdf.stat().st_size))
            self.end_headers()
            with open(pdf, "rb") as f:
                shutil.copyfileobj(f, self.wfile)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/ask":
            return self.send_error(404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            message = str(req.get("message", "")).strip()
            history = [(str(u), str(a)) for u, a in req.get("history", [])][-3:]
            if not message:
                return self.send_json(400, {"error": "empty message"})
            with index_lock:
                result = Chat(index, history).ask_cited(message)
            self.send_json(200, result)
        except Exception as e:  # surface model/API errors in the UI instead of a dropped request
            self.send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}\n")


def main():
    global index
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    index = Index()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Certamen sources UI on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
