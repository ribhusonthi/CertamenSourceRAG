# CertamenSourceRAG

Search index and question-answering API over the certamen textbooks. It powers
**Source check** on certamenreader (`/db/sources`).

The textbooks (`Textbooks/`), the text extracted from them (`extracted/`) and the
index (`index/`) are copyrighted material or derived from it. They are gitignored
and must never be pushed to GitHub. The index reaches Cloud Run only inside the
container image, which `gcloud run deploy --source .` builds from this folder
(see `.gcloudignore`).

## Pipeline

```bash
python3 scripts/extract_pages.py   # PDFs -> extracted/pages/*.jsonl, one record per printed page
python3 scripts/build_index.py     # chunks + embeddings -> index/sources.db
python3 scripts/chat.py            # terminal chat
python3 scripts/web.py             # local UI with PDF links, http://127.0.0.1:8770
python3 scripts/api.py             # the API certamenreader calls, http://127.0.0.1:8771
```

The scripts use Vertex AI (Gemini 3.8 Flash, gemini-embedding-001) through your
gcloud application-default credentials, project `<your-project-id>`.

## Deploying (Cloud Run service `certamen-sources`)

The API returns only the cited sentence with one sentence of context on each
side. It never returns whole passages or PDFs. It accepts requests only when the
`X-Sources-Secret` header matches `SOURCES_SECRET`, which only certamenreader's
server knows.

```bash
# once: create the shared secret
openssl rand -hex 32 | gcloud secrets create SOURCES_SECRET --data-file=-

# build from this folder (includes index/sources.db) and deploy
gcloud run deploy certamen-sources --source . --region us-central1 \
  --memory 1Gi --timeout 120 --allow-unauthenticated \
  --set-secrets SOURCES_SECRET=SOURCES_SECRET:latest \
  --set-env-vars GOOGLE_CLOUD_PROJECT=<your-project-id>

# point certamenreader at it (env vars survive its normal deploys)
gcloud run services update certamenreader --region us-central1 \
  --update-env-vars SOURCES_API_URL=<certamen-sources URL> \
  --update-secrets SOURCES_SECRET=SOURCES_SECRET:latest
```

The service runs as the default compute service account, which already has
`roles/aiplatform.user` and Secret Manager access. After rebuilding the index,
re-run the `gcloud run deploy` command to ship it.
