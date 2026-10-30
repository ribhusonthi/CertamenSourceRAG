#!/usr/bin/env python3
"""Ask questions of the textbooks; every claim in the answer cites a source page.

Retrieval is hybrid: BM25 keyword search (good at rare names like "Myceneus")
plus embedding search (good at paraphrase), merged with reciprocal rank fusion.
The top passages go to Gemini, which may answer only from them.

Usage:
  python3 scripts/chat.py                       # interactive chat
  python3 scripts/chat.py "Who founded Mycenae?"
  python3 scripts/chat.py --passages "..."      # also print the retrieved passages
"""
import argparse
import re
import sqlite3
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_config import (  # noqa: E402
    CHAT_MODEL, EMBED_DIM, EMBED_MODEL, INDEX_DB, TIER_NAMES, TIER_WEIGHTS, genai_client, retry, source_tier)

CANDIDATES = 40   # hits taken from each of keyword and vector search
TOP_K = 15        # passages given to the model
RRF_K = 60
# Query rewriting and embedding normally take 1–3 s; give up on a stuck call fast.
QUICK_TIMEOUT_MS = 15_000
# Writing a cited answer can take ~30 s; the site waits 90 s in all, so cap it below that.
ANSWER_TIMEOUT_MS = 55_000
RARE_TERM_DOCS = 150        # a query word in at most this many passages gets its own search
RARE_TERM_STEM_BELOW = 5    # fewer exact hits than this: also match by stem ("contubern*")
RARE_TERM_WEIGHT = 2        # an exact rare-word hit counts double when the lists are merged

STOPWORDS = set("""
a an and are as at be been but by did do does for from had has have he her his how i if in into is it its
me my no not of on or our she so than that the their them then there these they this those to was we were
what when where which who whom whose why will with would you your also about after before during under
over between than more most some any each other such only same very can could may might shall should
name identify give tell according said says say called known one two three first second
""".split())

# Wording that marks a statement as one version among several (common in the mythology books).
VARIANT_WORDS = re.compile(
    r"\b(?:another|other|some|one|later|earlier|different|alternative|variant|rival)\s+"
    r"(?:tradition|version|account|story|legend|myth|authors?|writers?|sources?)s?\b"
    r"|\bsome (?:say|said|claim|relate|tell|make)\b|\bothers (?:say|said|claim|relate|make|tell)\b"
    r"|\baccording to (?:some|others|one|another)\b|\balternatively\b|\bit (?:is|was) also said\b"
    r"|\bin (?:one|another|some|a later|an earlier) version\b|\bless commonly\b", re.I)
# Wording that marks the standard version.
MAIN_WORDS = re.compile(
    r"\b(?:the )?(?:usual|common|commonest|most common|best[- ]known|standard|traditional|canonical)"
    r" (?:version|account|story|tradition|legend)\b|\bmost (?:authors|writers|accounts|sources|versions)\b"
    r"|\busually (?:said|called|regarded|made|named)\b|\bgenerally (?:said|agreed|accepted)\b", re.I)

SYSTEM = """You are a fact-checking assistant for Latin and Classics certamen (Roman history, \
mythology, Latin literature, Roman life). You answer ONLY from the numbered source passages \
provided with each question, which come from standard certamen textbooks.

Rules:
- Cite every factual claim (the citation format is given at the end).
- Use only what the passages say. Never fill gaps from your own knowledge. If they answer only \
part of the question, give that part and say plainly what they don't cover. Say "The sources \
here don't cover this." only when nothing in the passages bears on the question.
- If the user gives a claim, a question with its answer, or a certamen tossup, start with \
exactly one verdict: Supported, Contradicted, Partly supported, or Not found in the sources. \
Then explain with citations, checking each clue separately when there are several. A clue \
that no passage mentions is "not found", not contradicted.
- If sources disagree (different dates, parentage, numbers), say so and cite each.
- Quote a short phrase from a passage when the exact wording decides the matter.
- Be concise.

Source priority. Each passage is labeled as a primary, secondary or semi-official source:
- Base each claim on primary sources when they cover it. Use secondary sources only for what \
the primary ones don't cover, and semi-official sources only when neither does.
- When sources disagree, follow the highest-ranked one and mention the disagreement, saying which \
source says what.
- List citations to primary sources first.
- These labels are for you; don't write "primary source" or "semi-official source" in the \
answer. Name the book instead when it matters.

Myths often exist in several versions, and certamen questions usually follow the standard one:
- For every mythological fact, check how the passage frames it. A passage header may list \
"Alternate-version wording" or "Main-version wording" found in it; read the sentence around \
that phrase to see which fact it applies to.
- Treat a fact as an alternate version when the passage frames it that way ("some say", \
"another tradition", "according to some", "in a later version", "less commonly", or a single \
named author such as "according to Pausanias"), or when the sources simply disagree.
- Give the standard version first: the one the passages call usual, or the one most of them \
give. Then list each alternate under a bold **Alternate versions** heading, quoting the \
source's own framing phrase. Never present an alternate version as the standard story.
- When verifying a tossup, if a clue matches only an alternate version, the verdict must say \
so, e.g. "Supported (alternate version only)", and name the standard version.
"""

# The terminal shows passage numbers; the web UI needs the exact supporting sentence per citation.
TEXT_FORMAT = """
Citation format: after each factual sentence put the numbers of the passages it comes from in \
square brackets, like [3] or [2][7]."""

CITED_FORMAT = """
Citation format: reply with JSON. Write the answer as a list of "segments" that join, in \
order, into one Markdown answer. Each segment is one claim, usually one sentence or one list \
item; put any Markdown it needs in its "text" (e.g. "\\n\\n**Alternate versions**\\n\\n" as its \
own segment, or "\\n- " before a list item). Do not write citation numbers in the text. In \
"sources", list every passage that supports that segment's claim, each with "passage" (the \
passage number) and "quote": the single sentence from that passage that supports this exact \
claim, copied word for word. Headings and connecting text have no sources."""

CITED_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "sources": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"passage": {"type": "integer"}, "quote": {"type": "string"}},
                    "required": ["passage", "quote"]}},
            },
            "required": ["text", "sources"]}},
    },
    "required": ["segments"],
}

QUERIES = """You write search queries for a library of Classics textbooks (Roman history, Greek \
and Roman mythology, Latin literature, Roman daily life). Write up to 4 short search queries \
that would find the passages answering the user's latest message. If it has several clues (as \
certamen tossups do), write one query per clue, each naming that clue's key people, places or \
things, so every clue gets searched. If the message gives an answer to check, make one query \
the plain core fact with that answer (e.g. "Paris killed Achilles"). Resolve pronouns and \
references from the conversation. Use the key names and terms, and add alternate spellings or \
the Greek/Latin forms of names where they exist (e.g. Heracles / Hercules, Mycenae / Mycene, \
Odysseus / Ulysses). One query per line, no numbering, nothing else.

Conversation so far:
{history}

Latest message: {message}"""


class Index:
    def __init__(self, path=INDEX_DB):
        if not Path(path).exists():
            raise SystemExit(f"no index at {path}; run scripts/build_index.py first")
        self.path = path
        self._local = threading.local()
        model, dim = (dict(self.db.execute("SELECT key, value FROM meta"))[k] for k in ("embed_model", "embed_dim"))
        if model != EMBED_MODEL or int(dim) != EMBED_DIM:
            raise SystemExit(f"index was built with {model}/{dim}, config says {EMBED_MODEL}/{EMBED_DIM}")
        rows = self.db.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
        self.ids = np.array([r[0] for r in rows])
        self.vecs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        # Search weight per chunk, from its book's tier (primary books first).
        self.tier_weight = {cid: TIER_WEIGHTS[source_tier(sid)]
                            for cid, sid in self.db.execute("SELECT id, source_id FROM chunks")}
        self.client = genai_client()

    @property
    def db(self):
        """One read-only SQLite connection per thread, so web requests can run concurrently."""
        conn = getattr(self._local, "db", None)
        if conn is None:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            # Per-term document counts, so keyword_search can tell rare words from common ones.
            conn.execute("CREATE VIRTUAL TABLE temp.vocab USING fts5vocab(main, chunks_fts, row)")
            self._local.db = conn
        return conn

    def fts(self, match, k):
        rows = self.db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?", (match, k))
        return [r[0] for r in rows]

    def doc_freq(self, term):
        row = self.db.execute("SELECT doc FROM temp.vocab WHERE term = ?", (term,)).fetchone()
        return row[0] if row else 0

    def keyword_search(self, query, k=CANDIDATES):
        """(weight, ranked list) pairs: all terms OR'd together, plus one list per rare term.

        With everything OR'd, pages that repeat common words ("Latin", "author", "born")
        outrank the one page with the rare word that matters ("Brescia"), so each rare
        term also gets a search of its own. A long term with almost no exact hits (a Latin
        inflection like "contubernalis", or an OCR misspelling) is searched by its stem.
        """
        terms = []
        for t in re.findall(r"\w+", query.lower()):
            if len(t) >= 3 and t not in STOPWORDS and not t.isdigit() and t not in terms:
                terms.append(t)
        if not terms:
            return []
        lists = [(1, self.fts(" OR ".join(f'"{t}"' for t in terms), k))]
        for t in terms:
            df = self.doc_freq(t)
            if df < RARE_TERM_STEM_BELOW and len(t) >= 7:
                lists.append((RARE_TERM_WEIGHT, self.fts(f'"{t[:max(6, len(t) - 4)]}" *', k)))
            elif 0 < df <= RARE_TERM_DOCS:
                lists.append((RARE_TERM_WEIGHT, self.fts(f'"{t}"', k)))
        return [(w, r) for w, r in lists if r]

    def vector_search(self, queries, k=CANDIDATES):
        """One ranked list per query (embedded in a single call)."""
        from google.genai import types
        res = retry(lambda: self.client.models.embed_content(
            model=EMBED_MODEL, contents=queries,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=EMBED_DIM,
                                            http_options=types.HttpOptions(timeout=QUICK_TIMEOUT_MS))))
        out = []
        for e in res.embeddings:
            q = np.asarray(e.values, dtype=np.float32)
            scores = self.vecs @ (q / np.linalg.norm(q))
            top = np.argpartition(-scores, k)[:k]
            out.append([int(self.ids[i]) for i in top[np.argsort(-scores[top])]])
        return out

    def search(self, queries, k=TOP_K):
        """The k passages for these queries: each query's keyword and vector results are
        fused into one ranking, then the rankings take turns filling the slots. Taking turns
        matters for multi-clue tossups: with one combined ranking, the clue with the most
        matching text crowded out the others (Memnon's dew lost to the Heliades' amber)."""
        rankings = []
        for query, vector_ranked in zip(queries, self.vector_search(queries)):
            # General matches are scaled by the book's tier, so primary books come
            # first. Exact hits on a rare word (or its stem) are not: if a primary
            # book has that word it already leads that list, and if only a secondary
            # one does, that passage is what "look at the other books" means.
            general, exact = {}, {}
            for weight, ranked in self.keyword_search(query) + [(1, vector_ranked)]:
                target = exact if weight == RARE_TERM_WEIGHT else general
                for rank, cid in enumerate(ranked):
                    target[cid] = target.get(cid, 0) + weight / (RRF_K + rank + 1)
            fused = {cid: general.get(cid, 0) * self.tier_weight[cid] + exact.get(cid, 0)
                     for cid in general.keys() | exact.keys()}
            rankings.append(sorted(fused, key=fused.get, reverse=True))
        best = []
        for depth in range(max(map(len, rankings), default=0)):
            for ranking in rankings:
                if depth < len(ranking) and ranking[depth] not in best:
                    best.append(ranking[depth])
            if len(best) >= k:
                break
        best = best[:k]
        cols = "id, source_id, cite, page, pdf_page, text"
        rows = {r[0]: r for r in self.db.execute(
            f"SELECT {cols} FROM chunks WHERE id IN ({','.join('?' * len(best))})", best)}
        passages = [dict(zip(cols.split(", "), rows[i])) for i in best]
        for p in passages:
            p["tier"] = TIER_NAMES[source_tier(p["source_id"])]
        return passages


def page_ref(p):
    if p["page"]:
        return f"{'pp' if '–' in p['page'] else 'p'}. {p['page']}"
    return f"unnumbered page (pdf page {p['pdf_page']})"


def phrases(pattern, text, limit=4):
    found = []
    for m in pattern.finditer(text):
        s = m.group(0).lower()
        if s not in found:
            found.append(s)
    return found[:limit]


def version_flags(text):
    lines = []
    if alt := phrases(VARIANT_WORDS, text):
        lines.append("Alternate-version wording: " + ", ".join(f'"{s}"' for s in alt))
    if main := phrases(MAIN_WORDS, text):
        lines.append("Main-version wording: " + ", ".join(f'"{s}"' for s in main))
    return lines


def format_passages(passages):
    return "\n\n".join(
        "\n".join([f"[{n}] {p['cite']}, {page_ref(p)} ({p['tier']} source)", *version_flags(p["text"]), p["text"]])
        for n, p in enumerate(passages, 1))


def words(s):
    return re.findall(r"\w+", s.lower())


def locate_quote(quote, text):
    """Find the model's quote in the passage: (start, end, exact). Matching ignores spacing,
    punctuation and case; if the quote isn't there word for word, fall back to the passage
    sentence that shares the most words with it."""
    qw = words(quote)
    if qw:
        m = re.search(r"\W+".join(map(re.escape, qw)), text, re.I)
        if m:
            return m.start(), m.end(), True
    best, best_score = None, 0
    for m in re.finditer(r"[^.!?]+[.!?]*", text):
        score = len(set(qw) & set(words(m.group(0))))
        if score > best_score:
            best, best_score = m, score
    if not best:
        return 0, 0, False
    start = best.start() + len(best.group(0)) - len(best.group(0).lstrip())
    return start, best.end(), False


class Chat:
    def __init__(self, index, history=None):
        self.index = index
        self.history = list(history or [])  # (user message, answer) pairs

    def contents(self, message, passages):
        from google.genai import types
        contents = []
        for u, a in self.history[-3:]:
            # Old citation numbers would clash with this turn's passage numbers.
            contents += [types.Content(role="user", parts=[types.Part(text=u)]),
                         types.Content(role="model", parts=[types.Part(text=re.sub(r"\[\d+\]", "", a))])]
        prompt = f"Source passages:\n\n{format_passages(passages)}\n\nUser: {message}"
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
        return contents

    def ask_cited(self, message):
        """Answer for the web UI: Markdown with [n] markers, and per marker the exact sentence,
        book and page it came from."""
        import json
        from google.genai import types
        queries = self.search_queries(message)
        passages = self.index.search(queries)
        res = retry(lambda: self.index.client.models.generate_content(
            model=CHAT_MODEL, contents=self.contents(message, passages),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM + CITED_FORMAT, temperature=0.2,
                response_mime_type="application/json", response_schema=CITED_SCHEMA,
                http_options=types.HttpOptions(timeout=ANSWER_TIMEOUT_MS))))
        # Number citations in order of first use; the same sentence cited twice keeps its number.
        citations, numbers, answer = [], {}, ""
        for seg in json.loads(res.text).get("segments", []):
            text = seg.get("text", "")
            marks = []
            for s in seg.get("sources", []):
                if not 0 < s["passage"] <= len(passages):
                    continue
                p = passages[s["passage"] - 1]
                start, end, exact = locate_quote(s["quote"], p["text"])
                key = (p["id"], start, end)
                if key not in numbers:
                    numbers[key] = len(citations) + 1
                    citations.append({
                        "n": numbers[key], "cite": p["cite"], "source_id": p["source_id"], "page": p["page"],
                        "pdf_page": p["pdf_page"], "page_ref": page_ref(p), "text": p["text"],
                        "quote_start": start, "quote_end": end, "exact": exact,
                        "alternate": bool(VARIANT_WORDS.search(p["text"])), "tier": p["tier"]})
                if numbers[key] not in marks:
                    marks.append(numbers[key])
            # Markers go after the claim, before any trailing whitespace/newlines.
            # The model sometimes leaves out the space between one segment and the next.
            if answer and not answer[-1].isspace() and text[:1].isalnum():
                answer += " "
            body = text.rstrip()
            answer += body + "".join(f"[{n}]" for n in marks) + text[len(body):]
        self.history.append((message, answer))
        return {"answer": answer, "citations": citations, "queries": queries}

    def search_queries(self, message):
        """The message itself plus up to 3 rewritten queries with alternate name spellings."""
        from google.genai import types
        convo = "\n".join(f"User: {u}\nAssistant: {a[:600]}" for u, a in self.history[-3:]) or "(none)"
        try:
            res = self.index.client.models.generate_content(
                model=CHAT_MODEL, contents=QUERIES.format(history=convo, message=message),
                config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="low"), temperature=0,
                                                   http_options=types.HttpOptions(timeout=QUICK_TIMEOUT_MS)))
        except Exception as e:  # the rewrite only improves recall; search with the question as asked
            print(f"query rewrite failed, using the question as is: {e}", file=sys.stderr, flush=True)
            return [message]
        extra = [q.strip(" -*\"") for q in (res.text or "").splitlines() if q.strip()][:4]
        # A follow-up like "who was his father?" is useless as a query on its own.
        return ([message] if not self.history else []) + extra or [message]

    def ask(self, message, show_passages=False, out=sys.stdout):
        from google.genai import types
        queries = self.search_queries(message)
        passages = self.index.search(queries)
        if show_passages:
            print(f"--- search: {queries}\n{format_passages(passages)}\n---\n", file=out)
        answer = ""
        for part in self.index.client.models.generate_content_stream(
                model=CHAT_MODEL, contents=self.contents(message, passages),
                config=types.GenerateContentConfig(system_instruction=SYSTEM + TEXT_FORMAT, temperature=0.2)):
            if part.text:
                answer += part.text
                print(part.text, end="", flush=True, file=out)
        print("\n", file=out)
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(passages)})
        if cited:
            print("Sources:", file=out)
            for n in cited:
                p = passages[n - 1]
                note = " (mentions alternate versions)" if VARIANT_WORDS.search(p["text"]) else ""
                print(f"  [{n}] {p['cite']}, {page_ref(p)}{note}", file=out)
            print(file=out)
        self.history.append((message, answer))
        return answer, [passages[n - 1] for n in cited]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--passages", action="store_true", help="print the retrieved passages")
    args = ap.parse_args()
    chat = Chat(Index())
    if args.question:
        chat.ask(" ".join(args.question), args.passages)
        return
    print(f"Certamen source chat ({CHAT_MODEL}). Ask a question, paste a tossup to verify, "
          "'/reset' to clear the conversation, Ctrl-D to quit.\n")
    while True:
        try:
            msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if msg == "/reset":
            chat.history.clear()
            print("(conversation cleared)\n")
        elif msg:
            chat.ask(msg, args.passages)


if __name__ == "__main__":
    main()
