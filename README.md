# CertamenSourceRAG

Search index and question-answering API over the certamen textbooks. It powers
**Source Check** on certamenreader (`/source-check`).

The textbooks (`Textbooks/`), the text extracted from them (`extracted/`) and the
index (`index/`) are copyrighted material or derived from it. The index reaches Cloud Run only inside the
container image, which `gcloud run deploy --source .` builds from this folder
(see `.gcloudignore`).

## Pipeline

```bash
python3 scripts/gemini_ocr.py      # page images -> extracted/gemini-ocr/ (Gemini transcription, keeps macrons)
python3 scripts/extract_pages.py   # PDFs -> extracted/pages/*.jsonl, one record per printed page
python3 scripts/build_index.py     # chunks + embeddings -> index/sources.db
python3 scripts/chat.py            # terminal chat
python3 scripts/web.py             # local UI with PDF links, http://127.0.0.1:8770
python3 scripts/api.py             # the API certamenreader calls, http://127.0.0.1:8771
```

Page text comes from Gemini reading each page image: the books' own OCR layers
and Tesseract drop macrons, so quotes with macrons were not found. `gemini_ocr.py`
skips pages already done; `--check` lists pages to review. Page numbers still
come from the PDFs (labels or printed headers). Index and contents pages are left
out of the search index.

Which books are trusted first (primary, secondary, semi-official) is set in
`SOURCE_TIERS` in `scripts/rag_config.py`.

The scripts use Vertex AI (Gemini 3.8 Flash, gemini-embedding-001) through your
gcloud application-default credentials, project
