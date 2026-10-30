#!/usr/bin/env python3
"""Transcribe every page image with Gemini, keeping macrons and Greek.

The books' own OCR layers drop macrons and garble Latin ("praenōmen" came out as
"preen.Giilen"), and Tesseract's English model has no macron letters at all, so
page text comes from Gemini reading the page image instead. extract_pages.py
picks these transcriptions up; page numbers still come from the PDFs.

  python3 scripts/gemini_ocr.py               # every book whose text comes from Gemini
  python3 scripts/gemini_ocr.py tripp hadas   # just these books
  python3 scripts/gemini_ocr.py --check       # list pages to review (no API calls)

Writes extracted/gemini-ocr/<id>/<pdf page>.txt, one per page; reruns skip pages
already done. Pages Gemini refused or cut off are listed in failed.json there.
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_pages import BOOKS, GEMINI_DIR, find_pdf  # noqa: E402
from rag_config import CHAT_MODEL, genai_client  # noqa: E402

DPI = 200
THREADS_PER_BOOK = 6
PRICE_IN, PRICE_OUT = 0.75, 3.75  # $ per million tokens (gemini-3.8-flash, until 2026-12-31)

PROMPT = (
    "Transcribe this book page exactly as printed. Keep every macron and other diacritic "
    "(ā ē ī ō ū ȳ, æ, œ) and any Greek letters. Plain text only: no markdown, no commentary. "
    "Keep paragraph breaks; join lines within a paragraph and rejoin words hyphenated across "
    "lines. Skip the running header, the page number, and scanner watermarks or library stamps "
    "(e.g. \"Digitized by Google\"). Include footnotes and captions. If the page has no text, "
    "reply with nothing."
)


def transcribe(client, png):
    """(text, finish reason, input tokens, output tokens); retries rate limits and dropped connections."""
    from google.genai import types
    for attempt in range(6):
        try:
            res = client.models.generate_content(
                model=CHAT_MODEL,
                contents=[types.Part.from_bytes(data=png, mime_type="image/png"), PROMPT],
                config=types.GenerateContentConfig(
                    temperature=0, max_output_tokens=8192,
                    thinking_config=types.ThinkingConfig(thinking_level="low")))
            cand = res.candidates[0] if res.candidates else None
            reason = str(cand.finish_reason).split(".")[-1] if cand else "NO_CANDIDATE"
            um = res.usage_metadata
            return res.text or "", reason, um.prompt_token_count or 0, (um.candidates_token_count or 0) + (um.thoughts_token_count or 0)
        except Exception as e:  # 429s, timeouts, 5xx: back off and retry
            if attempt == 5:
                return "", f"ERROR: {str(e)[:200]}", 0, 0
            time.sleep(min(60, 3 * 2 ** attempt))


def ocr_book(book_id):
    book = next(b for b in BOOKS if b["id"] == book_id)
    out = GEMINI_DIR / book_id
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(find_pdf(book["match"]))
    client = genai_client()
    stats = {"done": 0, "skipped": 0, "blank": 0, "failed": {}, "in": 0, "out": 0}

    def work(page_no, png):
        text, reason, tin, tout = transcribe(client, png)
        return page_no, text, reason, tin, tout

    with ThreadPoolExecutor(THREADS_PER_BOOK) as pool:
        futures = []
        for i in range(doc.page_count):
            path = out / f"{i + 1}.txt"
            if path.exists():
                stats["skipped"] += 1
                continue
            page = doc[i]
            if not page.get_text().strip() and not page.get_images():
                path.write_text("")
                stats["blank"] += 1
                continue
            futures.append(pool.submit(work, i + 1, page.get_pixmap(dpi=DPI).tobytes("png")))
            # Keep only a few rendered pages queued, so memory stays flat.
            while sum(not f.done() for f in futures) > THREADS_PER_BOOK * 2:
                time.sleep(0.2)
        for f in as_completed(futures):
            page_no, text, reason, tin, tout = f.result()
            stats["in"] += tin
            stats["out"] += tout
            if reason == "STOP":
                (out / f"{page_no}.txt").write_text(unicodedata.normalize("NFC", text))
                stats["done"] += 1
            else:
                stats["failed"][page_no] = reason
    (out / "failed.json").write_text(json.dumps(stats["failed"], indent=1))
    return book_id, stats


WORD = re.compile(r"[^\W\d_]{3,}")


def words(text):
    plain = unicodedata.normalize("NFKD", text.lower())
    return WORD.findall("".join(c for c in plain if not unicodedata.combining(c)))


def check(book_ids):
    """Pages to review: failures, and transcriptions much shorter than the old text layer."""
    for book_id in book_ids:
        book = next(b for b in BOOKS if b["id"] == book_id)
        doc = fitz.open(find_pdf(book["match"]))
        out = GEMINI_DIR / book_id
        failed = json.loads((out / "failed.json").read_text()) if (out / "failed.json").exists() else {}
        missing = [i + 1 for i in range(doc.page_count) if not (out / f"{i + 1}.txt").exists()]
        short = []
        for i in range(doc.page_count):
            path = out / f"{i + 1}.txt"
            if not path.exists():
                continue
            old = words(doc[i].get_text())
            new = words(path.read_text())
            if len(old) >= 60 and len(new) < 0.6 * len(old):
                short.append((i + 1, len(new), len(old)))
        print(f"{book_id}: {doc.page_count} pages, missing {len(missing)}, failed {len(failed)}, much shorter than old text {len(short)}")
        for page, reason in list(failed.items())[:10]:
            print(f"    failed pdf {page}: {reason}")
        for page, n_new, n_old in short[:10]:
            print(f"    short pdf {page}: {n_new} words vs {n_old} in the old text layer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--books-at-once", type=int, default=4)
    args = ap.parse_args()
    ids = args.ids or [b["id"] for b in BOOKS if b.get("text", "gemini") == "gemini"]
    unknown = set(ids) - {b["id"] for b in BOOKS}
    if unknown:
        raise SystemExit(f"unknown ids: {sorted(unknown)}")
    if args.check:
        return check(ids)

    started, total_in, total_out = time.time(), 0, 0
    with ProcessPoolExecutor(args.books_at_once) as pool:
        for fut in as_completed([pool.submit(ocr_book, i) for i in ids]):
            book_id, s = fut.result()
            total_in += s["in"]
            total_out += s["out"]
            print(f"{book_id}: {s['done']} transcribed, {s['skipped']} already done, {s['blank']} blank, "
                  f"{len(s['failed'])} failed  ({(time.time() - started) / 60:.0f} min)", flush=True)
    cost = total_in / 1e6 * PRICE_IN + total_out / 1e6 * PRICE_OUT
    print(f"tokens in {total_in:,} out {total_out:,}  ~${cost:.2f}")


if __name__ == "__main__":
    main()
