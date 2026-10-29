#!/usr/bin/env python3
"""Extract the textbooks page by page, keeping the printed page number of every page.

Output (in extracted/):
  txt/<id>.txt      readable text, one "===== [cite p. N] =====" marker per page
  pages/<id>.jsonl  one JSON record per PDF page (the source of truth for chunking)
  report.md         page-number method, OCR counts and pages worth spot-checking
  ocr-cache/        raw tesseract output per page, so reruns skip the slow OCR

Printed page numbers come from the PDF's embedded page labels when the book has
them, otherwise from the page numbers printed in the running headers/footers.
Pages with no usable text layer are OCR'd with tesseract.

Usage:
  python3 scripts/extract_pages.py                 # all books
  python3 scripts/extract_pages.py grimal hadas    # just these ids
"""
import argparse
import collections
import json
import re
import subprocess
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
TEXTBOOKS = ROOT / "Textbooks"
OUT = ROOT / "extracted"

# match: a substring of the PDF filename in Textbooks/.
# pages: "labels" = embedded PDF page labels, "header" = numbers printed on the page.
# ocr:   "missing" = only pages without a text layer, "all" = ignore the text layer.
BOOKS = [
    dict(id="adkins-handbook", match="Handbook to Life", pages="labels", ocr="missing",
         author="Lesley Adkins and Roy A. Adkins", title="Handbook to Life in Ancient Rome (Updated Edition)",
         cite="Adkins & Adkins, Handbook to Life in Ancient Rome"),
    dict(id="cary-scullard", match="Cary and Scullard", pages="labels", ocr="missing",
         author="M. Cary and H. H. Scullard", title="A History of Rome down to the Reign of Constantine",
         cite="Cary & Scullard, A History of Rome"),
    dict(id="claridge-rome", match="Oxford Archeological", pages="labels", ocr="missing",
         author="Amanda Claridge", title="Rome: An Oxford Archaeological Guide",
         cite="Claridge, Rome (Oxford Archaeological Guide)"),
    dict(id="conte", match="Conte", pages="labels", ocr="missing",
         author="Gian Biagio Conte (trans. Joseph B. Solodow)", title="Latin Literature: A History",
         cite="Conte, Latin Literature"),
    dict(id="grimal", match="Grimal", pages="labels", ocr="missing",
         author="Pierre Grimal (trans. A. R. Maxwell-Hyslop)", title="The Dictionary of Classical Mythology",
         cite="Grimal, Dictionary of Classical Mythology"),
    dict(id="hadas", match="Hadas", pages="labels", ocr="missing",
         author="Moses Hadas", title="A History of Latin Literature",
         cite="Hadas, A History of Latin Literature"),
    dict(id="johnston-private-life", match="Private_Life", pages="header", ocr="missing",
         author="Harold Whetstone Johnston", title="The Private Life of the Romans",
         cite="Johnston, The Private Life of the Romans"),
    dict(id="march", match="March", pages="labels", ocr="missing",
         author="Jenny March", title="Dictionary of Classical Mythology",
         cite="March, Dictionary of Classical Mythology"),
    dict(id="matyszak-republic", match="Roman Republic", pages="header", ocr="all",
         author="Philip Matyszak", title="Chronicle of the Roman Republic",
         cite="Matyszak, Chronicle of the Roman Republic"),
    dict(id="morford-lenardon", match="Morford", pages="labels", ocr="missing",
         author="Mark P. O. Morford and Robert J. Lenardon", title="Classical Mythology (7th Edition)",
         cite="Morford & Lenardon, Classical Mythology"),
    dict(id="scarre-emperors", match="Scarre", pages="header", ocr="missing",
         author="Chris Scarre", title="Chronicle of the Roman Emperors",
         cite="Scarre, Chronicle of the Roman Emperors"),
    dict(id="tripp", match="Tripp", pages="labels", ocr="missing",
         author="Edward Tripp", title="The Meridian Handbook of Classical Mythology",
         cite="Tripp, Handbook of Classical Mythology"),
    dict(id="ward-heichelheim", match="Heichelheim", pages="labels", ocr="missing",
         author="Allen M. Ward, Fritz M. Heichelheim and Cedric A. Yeo", title="A History of the Roman People (6th Edition)",
         cite="Ward, Heichelheim & Yeo, A History of the Roman People"),
]

TOP_ZONE, BOTTOM_ZONE = 0.11, 0.90  # fraction of page height treated as header / footer
MIN_TEXT_CHARS = 50                  # below this a page has no usable text layer
OCR_DPI = 300
TEXT_FLAGS = (fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES & ~fitz.TEXT_PRESERVE_IMAGES)

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
             "ﬅ": "st", "ﬆ": "st", "­": ""}
STANDALONE_NUM = re.compile(r"^\W*(\d{1,4})\W*$")
SPACED_DIGITS = re.compile(r"(?<=\d) (?=\d)")
ROMAN = re.compile(r"^\W*[ivxlcdm]{1,7}\W*$", re.I)
LEADING_NUM = re.compile(r"^\W*(\d{1,4})\b")
TRAILING_NUM = re.compile(r"\b(\d{1,4})\W*$")
WORDLIKE = re.compile(r"^[(\[\"'‘“]*([A-Za-z][a-z'’]*(-[A-Za-z][a-z'’]*)*|\d[\d,.–-]*)[)\]\"'’”.,;:!?]*$")


def clean_line(s):
    s = unicodedata.normalize("NFC", s)
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    s = re.sub(r"[ -​ \t]", " ", s)
    return re.sub(r" {2,}", " ", s).strip()


def head_key(s):
    """Normalized form of a header line, for spotting running heads that repeat across pages."""
    return re.sub(r"[\W\d_]+", "", s.lower())


def ocr_page(page, cache_dir):
    cached = cache_dir / f"{page.number + 1}.txt"
    if cached.exists():
        return cached.read_text()
    png = page.get_pixmap(dpi=OCR_DPI).tobytes("png")
    res = subprocess.run(["tesseract", "stdin", "stdout", "-l", "eng", "--psm", "3"],
                         input=png, capture_output=True, check=True)
    text = res.stdout.decode("utf-8", "replace")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(text)
    return text


def page_has_scan(page):
    area = page.rect.width * page.rect.height
    for img in page.get_image_info():
        x0, y0, x1, y1 = img["bbox"]
        if (x1 - x0) * (y1 - y0) > 0.5 * area:
            return True
    return False


def read_page(page, ocr_mode, ocr_cache):
    """Return (blocks, ocr_used). A block is a list of lines; a line is {text, zone}.

    zone is "top" / "bottom" / None. For OCR text there are no coordinates, so the
    first two and last two lines of the page stand in for the header/footer zone.
    """
    blocks = []
    if ocr_mode != "all":
        h = page.rect.height
        for b in page.get_text("dict", flags=TEXT_FLAGS)["blocks"]:
            if b["type"] != 0:
                continue
            lines = []
            for ln in b["lines"]:
                text = clean_line("".join(sp["text"] for sp in ln["spans"]))
                if not text:
                    continue
                y0, y1 = ln["bbox"][1], ln["bbox"][3]
                zone = "top" if y1 <= TOP_ZONE * h else "bottom" if y0 >= BOTTOM_ZONE * h else None
                lines.append({"text": text, "zone": zone})
            if lines:
                blocks.append(lines)
        if blocks:
            # Some books print the number just inside the margins (Tripp's footer).
            blocks[0][0]["zone"] = blocks[0][0]["zone"] or "top"
            blocks[-1][-1]["zone"] = blocks[-1][-1]["zone"] or "bottom"
        chars = sum(len(l["text"]) for b in blocks for l in b)
        if chars >= MIN_TEXT_CHARS or not page_has_scan(page):
            return blocks, False

    blocks = []
    for para in re.split(r"\n\s*\n", ocr_page(page, ocr_cache)):
        lines = [{"text": t, "zone": None} for t in map(clean_line, para.splitlines()) if t]
        if lines:
            blocks.append(lines)
    flat = [l for b in blocks for l in b]
    for l in flat[:2]:
        l["zone"] = "top"
    for l in flat[-2:]:
        l["zone"] = l["zone"] or "bottom"
    if sum(len(l["text"]) for l in flat) < MIN_TEXT_CHARS:
        blocks = []
    return blocks, True


def candidate_number(line):
    """A printed page number on a header/footer line, or None."""
    t = SPACED_DIGITS.sub("", line["text"])  # letter-spaced "4 6" -> "46"
    m = STANDALONE_NUM.match(t)
    if m:
        return int(m.group(1))
    if line["zone"] == "top" and len(t) <= 100:
        m = LEADING_NUM.match(t) or TRAILING_NUM.search(t)
        if m:
            return int(m.group(1))
    return None


def header_page_numbers(cands, n):
    """Assign printed page numbers from the numbers found in headers/footers.

    The OCR layers of these scans misread page numbers (a dropped leading digit turns
    "105" into "5"), so each page's offset (printed number minus PDF index) is decided
    by a vote of the pages around it, and a page is an anchor only when its own number
    agrees with that vote. Pages between anchors inherit the offset when both
    neighbours agree; where the offset changes (unnumbered plates, missing pages) the
    pages in between are left unnumbered rather than guessed.
    """
    cands = {i: cs for i, cs in cands.items() if cs}
    totals = collections.Counter(o for i, cs in cands.items() for o in {c - i for c in cs})
    common = [o for o, c in totals.items() if c >= max(5, n // 50)]

    def supports(j, o):
        return j + o > 0 and any(number_matches(str(j + o), c) for c in cands.get(j, []))

    anchors = {}
    for i in cands:
        # A garbled "19" on p. 119 still supports the true offset, since 119 ends in 19.
        votes = {o: sum(1 / (1 + abs(i - j)) for j in range(i - 8, i + 9) if supports(j, o))
                 for o in common}
        if votes and max(votes.values()) > 0:
            best = max(votes, key=votes.get)
            if supports(i, best):
                anchors[i] = best
            else:
                # A full-length number that disagrees is a real break in the sequence
                # (a duplicated scan page), not a dropped digit.
                full = [c for c in cands[i] if len(str(c)) == len(str(i + best))]
                if len(full) == 1:
                    anchors[i] = full[0] - i
    idx = sorted(anchors)
    pages, method = [None] * n, ["none"] * n
    for i in range(n):
        if i in anchors:
            pages[i], method[i] = str(i + anchors[i]), "printed"
            continue
        prev = max((a for a in idx if a < i), default=None)
        nxt = min((a for a in idx if a > i), default=None)
        if prev is not None and nxt is not None and anchors[prev] == anchors[nxt]:
            pages[i], method[i] = str(i + anchors[prev]), "inferred"
        elif prev is not None and nxt is None:
            pages[i], method[i] = str(i + anchors[prev]), "inferred"
    return pages, method


def label_page_numbers(labels):
    """Drop numeric labels that are out of sequence with the labels around them
    (e.g. Hadas's contents pages labelled 360-361 just before p. 1)."""
    offs = {i: int(l) - i for i, l in enumerate(labels) if l and l.isdigit()}
    out = list(labels)
    for i, o in offs.items():
        near = collections.Counter(offs[j] for j in range(i - 5, i + 6) if j in offs)
        if near[o] < max(near.values()):
            out[i] = None
    return out


def number_matches(page_num, printed):
    """True when a printed number is this page's, allowing OCR-dropped leading digits."""
    return page_num.isdigit() and page_num.endswith(str(printed))


def join_lines(lines):
    out = ""
    for t in lines:
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and t[:1].islower():
            out = out[:-1] + t  # rejoin a word hyphenated across lines
        elif out:
            out += " " + t
        else:
            out = t
    return out


def render_text(blocks, page_num, running_heads):
    kept = []
    for b in blocks:
        removed_num = False
        lines = []
        for l in b:
            t, zone = l["text"], l["zone"]
            if zone:
                n = candidate_number(l)
                if (STANDALONE_NUM.match(SPACED_DIGITS.sub("", t)) or ROMAN.match(t)
                        or head_key(t) in running_heads
                        or (n is not None and page_num is not None and number_matches(page_num, n))):
                    removed_num = removed_num or n is not None
                    continue
            lines.append(l)
        # Guide words left in the same header block as the page number (e.g. "MYCENEUS").
        if removed_num and lines and all(l["zone"] == "top" for l in lines) and len(lines) <= 2:
            lines = []
        if lines:
            kept.append([l["text"] for l in lines])
    paras = []
    for b in kept:
        text = join_lines(b)
        # A paragraph split across columns: "sena-" / "tors loathed..." or "conceived an" / "incestuous..."
        if paras and paras[-1].endswith("-") and paras[-1][-2:-1].isalpha() and text[:1].islower():
            paras[-1] = paras[-1][:-1] + text
        elif paras and paras[-1][-1:].isalpha() and text[:1].islower():
            paras[-1] += " " + text
        else:
            paras.append(text)
    return "\n\n".join(paras)


def quality(text):
    toks = text.split()
    if not toks:
        return None
    return round(sum(1 for t in toks if WORDLIKE.match(t)) / len(toks), 3)


def find_pdf(match):
    hits = [p for p in TEXTBOOKS.glob("*.pdf") if match in p.name]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one PDF matching {match!r}, found {[h.name for h in hits]}")
    return hits[0]


def process_book(book):
    pdf = find_pdf(book["match"])
    doc = fitz.open(pdf)
    n = doc.page_count
    raw, ocr_used = [], []
    for i in range(n):
        blocks, used = read_page(doc[i], book["ocr"], OUT / "ocr-cache" / book["id"])
        raw.append(blocks)
        ocr_used.append(used)

    zone_lines = [[l for b in blocks for l in b if l["zone"]] for blocks in raw]
    head_counts = collections.Counter()
    for zl in zone_lines:
        head_counts.update({head_key(l["text"]) for l in zl if len(head_key(l["text"])) >= 3})
    running_heads = {k for k, c in head_counts.items() if c >= 5}
    cands = {i: [c for c in map(candidate_number, zl) if c is not None] for i, zl in enumerate(zone_lines)}

    if book["pages"] == "labels":
        pages = label_page_numbers([doc[i].get_label() or None for i in range(n)])
        method = ["label" if p else "none" for p in pages]
    else:
        pages, method = header_page_numbers(cands, n)

    # How often the printed number on the page agrees with the page number we assigned.
    checked = agreed = 0
    for i in range(n):
        if cands[i] and pages[i] and pages[i].isdigit():
            checked += 1
            agreed += any(number_matches(pages[i], c) for c in cands[i])

    records = []
    for i in range(n):
        text = render_text(raw[i], pages[i], running_heads)
        records.append({
            "source_id": book["id"], "author": book["author"], "title": book["title"], "cite": book["cite"],
            "pdf_file": pdf.name, "pdf_page": i + 1, "page": pages[i], "page_method": method[i],
            "ocr": ocr_used[i], "quality": quality(text), "text": text,
        })
    return book, records, {"checked": checked, "agreed": agreed, "running_heads": len(running_heads)}


def page_ref(r):
    if r["page"]:
        return f"p. {r['page']}"
    return f"unnumbered, pdf page {r['pdf_page']}"


def write_outputs(book, records):
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    (OUT / "txt").mkdir(parents=True, exist_ok=True)
    with open(OUT / "pages" / f"{book['id']}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "txt" / f"{book['id']}.txt", "w") as f:
        f.write(f"SOURCE: {book['author']}, {book['title']}\nCITE AS: {book['cite']}\n"
                f"FILE: {records[0]['pdf_file']}\n\n")
        for r in records:
            if not r["text"]:
                continue
            flags = " [OCR]" if r["ocr"] else ""
            f.write(f"===== [{book['cite']}, {page_ref(r)}] (pdf page {r['pdf_page']}){flags} =====\n\n")
            f.write(r["text"] + "\n\n")


def report_section(book, records, stats):
    n = len(records)
    blank = sum(1 for r in records if not r["text"])
    ocr = sum(1 for r in records if r["ocr"])
    counts = collections.Counter(r["page_method"] for r in records)
    low = [r for r in records if r["quality"] is not None and r["quality"] < 0.6]
    unnumbered = [r["pdf_page"] for r in records if not r["page"] and r["text"]]
    numbered = [r for r in records if r["page"] and r["page"].isdigit() and r["text"]]
    first = numbered[len(numbered) // 2] if numbered else None
    agree = f"{stats['agreed']}/{stats['checked']}" if stats["checked"] else "n/a"
    lines = [
        f"## {book['cite']} (`{book['id']}`)",
        f"- File: {records[0]['pdf_file']}",
        f"- PDF pages: {n}; blank: {blank}; OCR'd: {ocr}",
        f"- Page numbers from: {book['pages']} — " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        f"- Printed number on page agrees with assigned page: {agree}",
    ]
    if first:
        lines.append(f"- Spot check: pdf page {first['pdf_page']} should be printed p. {first['page']}")
    if unnumbered:
        lines.append(f"- Pages with text but no page number ({len(unnumbered)}): {compress(unnumbered)}")
    if low:
        lines.append(f"- Low-quality text (<60% word-like tokens, {len(low)} pages): "
                     + compress([r["pdf_page"] for r in low]))
    return "\n".join(lines)


def compress(nums):
    """[1,2,3,7,9,10] -> '1–3, 7, 9–10' (pdf page numbers)."""
    out, start, prev = [], None, None
    for x in nums + [None]:
        if start is None:
            start = prev = x
        elif x is not None and x == prev + 1:
            prev = x
        else:
            out.append(f"{start}–{prev}" if prev != start else str(start))
            start = prev = x
    return ", ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="book ids to extract (default: all)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    books = [b for b in BOOKS if not args.ids or b["id"] in args.ids]
    unknown = set(args.ids) - {b["id"] for b in BOOKS}
    if unknown:
        raise SystemExit(f"unknown ids: {sorted(unknown)}")

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_book, b): b["id"] for b in books}
        for fut in as_completed(futures):
            book, records, stats = fut.result()
            write_outputs(book, records)
            results[book["id"]] = (book, records, stats)
            print(f"done {book['id']}: {len(records)} pages", flush=True)

    sections = [report_section(*results[b["id"]]) for b in BOOKS if b["id"] in results]
    report = OUT / ("report.md" if not args.ids else f"report-{'-'.join(args.ids)}.md")
    report.write_text("# Extraction report\n\nPage numbers: `label` = embedded PDF page label, "
                      "`printed` = read from the page's header/footer, `inferred` = filled in between "
                      "printed pages with the same offset.\n\n" + "\n\n".join(sections) + "\n")
    print(f"wrote {report}")


if __name__ == "__main__":
    sys.exit(main())
