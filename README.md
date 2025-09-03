# PrivAI-Cloud-private

## Commands

```bash
uv venv .venv

uv sync

uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 75 --limit-concurrency 4
```

## Endpoints

- Docs [Docs](http://34.14.205.113:8000/docs)
- Transcribe [Transcribe](http://34.14.205.113:8000/transcribe)
