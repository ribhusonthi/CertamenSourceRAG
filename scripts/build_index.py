#!/usr/bin/env python3
"""Build the search index from extracted/pages/*.jsonl.

Every chunk stays inside one printed page, so each search hit cites exactly one
page. The index is a single SQLite file (index/sources.db) holding:
  chunks      the passages with their citation fields and embedding (float32 BLOB)
  chunks_fts  an FTS5 keyword index over the same passages (BM25 ranking)

Embeddings come from Vertex AI (gemini-embedding-001) using your gcloud
application-default credentials. They are cached in index/embed-cache.db by text
hash, so rebuilding after a re-extraction only pays for passages that changed.

Usage:
  python3 scripts/build_index.py
"""
import hashlib
import json
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_config import EMBED_DIM, EMBED_MODEL, INDEX_DB, ROOT, genai_client  # noqa: E402

PAGES = ROOT / "extracted" / "pages"
CACHE_DB = INDEX_DB.parent / "embed-cache.db"

TARGET_WORDS, MAX_WORDS, MIN_WORDS = 250, 400, 40
# Dictionary pages hold many short entries; small chunks keep one entry from being
# drowned out by its neighbours in the embedding.
DICTIONARIES = {"grimal", "march", "tripp"}
DICT_TARGET_WORDS = 100
MIN_QUALITY = 0.6     # pages below this are OCR junk, plates, or index pages
BATCH = 50
WORKERS = 8


def split_long(para, target):
    """Split a paragraph longer than MAX_WORDS at sentence boundaries."""
    if len(para.split()) <= MAX_WORDS:
        return [para]
    out, cur = [], []
    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'‘“(])", para):
        if cur and len(" ".join(cur + [sent]).split()) > target:
            out.append(" ".join(cur))
            cur = []
        cur.append(sent)
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_page(text, target=TARGET_WORDS):
    paras = [p for raw in text.split("\n\n") for p in split_long(raw.strip(), target) if p]
    chunks, cur = [], []
    for p in paras:
        if cur and len(" ".join(cur + [p]).split()) > target:
            chunks.append("\n\n".join(cur))
            cur = []
        cur.append(p)
    if cur:
        tail = "\n\n".join(cur)
        # Fold a short leftover into the previous chunk on the same page.
        if chunks and len(tail.split()) < min(MIN_WORDS, target // 3):
            chunks[-1] += "\n\n" + tail
        else:
            chunks.append(tail)
    return chunks


def load_chunks():
    chunks = []
    for f in sorted(PAGES.glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if not r["text"] or (r["quality"] or 0) < MIN_QUALITY:
                continue
            dictionary = r["source_id"] in DICTIONARIES
            for text in chunk_page(r["text"], DICT_TARGET_WORDS if dictionary else TARGET_WORDS):
                if len(text.split()) < (5 if dictionary else 15):
                    continue
                chunks.append({k: r[k] for k in ("source_id", "cite", "author", "title", "page", "pdf_page")}
                              | {"text": text})
    return chunks


def embed_text(c):
    """What gets embedded: the source name gives the vector some context."""
    return f"{c['cite']}\n\n{c['text']}"


def text_hash(s):
    return hashlib.sha1(f"{EMBED_MODEL}:{EMBED_DIM}:{s}".encode()).hexdigest()


def embed_batch(client, texts):
    from google.genai import types
    cfg = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=EMBED_DIM)
    for attempt in range(6):
        try:
            res = client.models.embed_content(model=EMBED_MODEL, contents=texts, config=cfg)
            vecs = [e.values for e in res.embeddings]
            if len(vecs) != len(texts):
                raise RuntimeError(f"asked for {len(texts)} embeddings, got {len(vecs)}")
            return vecs
        except Exception as e:  # rate limits and transient errors: back off and retry
            if attempt == 5:
                raise
            wait = 2 ** attempt * 2
            print(f"  embed retry in {wait}s: {str(e)[:120]}", flush=True)
            time.sleep(wait)


def embed_all(chunks):
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    cache = sqlite3.connect(CACHE_DB)
    cache.execute("CREATE TABLE IF NOT EXISTS emb (h TEXT PRIMARY KEY, v BLOB)")
    hashes = [text_hash(embed_text(c)) for c in chunks]
    have = {}
    for i in range(0, len(hashes), 900):
        part = hashes[i:i + 900]
        q = f"SELECT h, v FROM emb WHERE h IN ({','.join('?' * len(part))})"
        have.update(dict(cache.execute(q, part)))
    todo = sorted({h: i for i, h in enumerate(hashes) if h not in have}.items(), key=lambda x: x[1])
    print(f"{len(chunks)} chunks; {len(chunks) - len(todo)} cached embeddings, {len(todo)} to embed", flush=True)

    client = genai_client()
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    done = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [(b, pool.submit(embed_batch, client, [embed_text(chunks[i]) for _, i in b])) for b in batches]
        for b, fut in futures:
            vecs = fut.result()
            rows = [(h, np.asarray(v, dtype=np.float32).tobytes()) for (h, _), v in zip(b, vecs)]
            cache.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            cache.commit()
            have.update(rows)
            done += len(b)
            if done % 2000 < BATCH:
                print(f"  embedded {done}/{len(todo)}", flush=True)
    cache.close()
    vecs = np.stack([np.frombuffer(have[h], dtype=np.float32) for h in hashes])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def write_index(chunks, vecs):
    INDEX_DB.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_DB.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, source_id TEXT, cite TEXT, author TEXT, title TEXT,
            page TEXT, pdf_page INTEGER, text TEXT, embedding BLOB);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text, content='chunks', content_rowid='id', tokenize='unicode61 remove_diacritics 2');
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    db.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(i, c["source_id"], c["cite"], c["author"], c["title"], c["page"], c["pdf_page"], c["text"],
          vecs[i].tobytes()) for i, c in enumerate(chunks)])
    db.execute("INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunks")
    db.executemany("INSERT INTO meta VALUES (?, ?)",
                   [("embed_model", EMBED_MODEL), ("embed_dim", str(EMBED_DIM)),
                    ("built_at", time.strftime("%Y-%m-%d %H:%M:%S"))])
    db.commit()
    db.close()
    tmp.replace(INDEX_DB)


def main():
    chunks = load_chunks()
    vecs = embed_all(chunks)
    write_index(chunks, vecs)
    by_book = {}
    for c in chunks:
        by_book[c["source_id"]] = by_book.get(c["source_id"], 0) + 1
    print("chunks per book:", ", ".join(f"{k} {v}" for k, v in sorted(by_book.items())))
    print(f"wrote {INDEX_DB} ({INDEX_DB.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
