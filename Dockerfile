# Source-check API for certamenreader. Built from this local folder with
# `gcloud run deploy --source .` so the search index (index/sources.db, which is
# gitignored) goes into the image without ever touching GitHub.
FROM python:3.13-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy==2.4.3 google-genai==1.73.1
COPY scripts/chat.py scripts/rag_config.py scripts/api.py ./scripts/
COPY index/sources.db ./index/sources.db
ENV PYTHONUNBUFFERED=1 PORT=8080
EXPOSE 8080
USER nobody
CMD ["python3", "scripts/api.py"]
