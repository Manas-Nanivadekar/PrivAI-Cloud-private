import os
import asyncio
import time
import uuid
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.cors import CORSMiddleware

from .audio_io import prepare_audio_for_model
from .inference import engine, SUPPORTED_LANGS


DECODER_DEFAULT = os.environ.get("ASR_DECODER_DEFAULT", "rnnt").lower()
MAX_UPLOAD_MB = int(os.environ.get("ASR_MAX_UPLOAD_MB", "200"))
MAX_DURATION_MIN = int(os.environ.get("ASR_MAX_DURATION_MIN", "60"))
ALLOWED_MIME = set(
    (
        os.environ.get("ASR_ALLOWED_MIME")
        or "audio/wav,audio/x-wav,audio/mpeg,audio/mp4,audio/x-m4a,audio/ogg,audio/opus"
    ).split(",")
)
NORMALIZE_DEFAULT = os.environ.get("ASR_NORMALIZE_DEFAULT", "false").lower() == "true"
CONCURRENCY = int(os.environ.get("ASR_MAX_CONCURRENCY", "2"))
API_KEY = os.environ.get("ASR_API_KEY")

MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_DURATION_SEC = MAX_DURATION_MIN * 60


app = FastAPI(title="IndicConformer ASR", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ASR_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.state.sem = asyncio.Semaphore(CONCURRENCY)
app.state.ready = False


@app.on_event("startup")
def _load_model():

    engine.load()
    app.state.ready = True


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"


@app.get("/readyz", response_class=PlainTextResponse)
def readyz():
    return "ready" if app.state.ready else "starting"


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(..., description="ISO language code like hi, mr, ta, ..."),
    decoder: str = Form(DECODER_DEFAULT, description="'rnnt"),
    normalize: bool = Form(NORMALIZE_DEFAULT),
    return_segments: bool = Form(False),
):
    req_id = str(uuid.uuid4())[:8]
    t0 = time.monotonic()

    if file.content_type and file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415, detail=f"Unsupported content-type {file.content_type}"
        )

    read_bytes = 0

    def _iter():
        nonlocal read_bytes
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            read_bytes += len(chunk)
            if read_bytes > MAX_BYTES:
                raise HTTPException(
                    status_code=413, detail=f"File too large. Max {MAX_UPLOAD_MB} MB"
                )
            yield chunk

    async with app.state.sem:

        try:
            wav_path, duration = prepare_audio_for_model(
                _iter(), max_duration_sec=MAX_DURATION_SEC, normalize=normalize
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Audio preprocessing failed: {e}"
            )

        try:
            text = engine.transcribe(
                wav16k_path=wav_path, language=language, decoder=decoder.lower()
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass

    t1 = time.monotonic()
    return JSONResponse(
        {
            "request_id": req_id,
            "text": text,
            "language": language,
            "decoder": decoder.lower(),
            "duration_ms": int(duration * 1000),
            "processing_ms": int((t1 - t0) * 1000),
            "model_version": os.environ.get("ASR_REVISION", "main"),
        }
    )
