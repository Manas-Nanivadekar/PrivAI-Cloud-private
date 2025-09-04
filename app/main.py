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
from .wer_calculator import compute_wer


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


@app.post("/transcribe_with_comparison")
async def transcribe_with_comparison(
    file: UploadFile = File(...),
    language: str = Form(..., description="ISO language code like hi, mr, ta, ..."),
    reference_transcription: str = Form(
        ..., description="Reference transcription to compare against"
    ),
    decoder: str = Form(DECODER_DEFAULT, description="'rnnt"),
    normalize: bool = Form(NORMALIZE_DEFAULT),
    return_segments: bool = Form(False),
):
    """
    Transcribe audio and compute WER against a provided reference transcription.
    This is useful when you have a known correct transcription and want to evaluate
    your ASR model's performance.
    """
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
            # Get transcription from your model
            model_transcription = engine.transcribe(
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

    # Compute WER between model output and reference transcription
    wer_result = None
    try:
        wer_result = compute_wer(
            hypothesis=model_transcription, reference=reference_transcription
        )
    except Exception:
        wer_result = {"wer": None}

    return JSONResponse(
        {
            "request_id": req_id,
            "model_transcription": model_transcription,
            "reference_transcription": reference_transcription,
            "language": language,
            "decoder": decoder.lower(),
            "duration_ms": int(duration * 1000),
            "processing_ms": int((t1 - t0) * 1000),
            "model_version": os.environ.get("ASR_REVISION", "main"),
            "wer": wer_result.get("wer") if isinstance(wer_result, dict) else None,
            "wer_details": wer_result if isinstance(wer_result, dict) else None,
        }
    )


@app.post("/compute_wer")
async def compute_wer_endpoint(
    hypothesis: str = Form(..., description="Hypothesis transcription (model output)"),
    reference: str = Form(..., description="Reference transcription (ground truth)"),
):
    """
    Compute WER between two transcriptions without processing any audio.
    Useful for evaluating transcriptions you already have.
    """
    try:
        wer_result = compute_wer(hypothesis=hypothesis, reference=reference)
        return JSONResponse(wer_result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"WER computation failed: {e}")


# Enhanced version of your existing endpoint with better error handling
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(..., description="ISO language code like hi, mr, ta, ..."),
    decoder: str = Form(DECODER_DEFAULT, description="'rnnt"),
    normalize: bool = Form(NORMALIZE_DEFAULT),
    return_segments: bool = Form(False),
    reference_text: Optional[str] = Form(
        None, description="Optional reference text for WER calculation"
    ),
):
    """
    Transcribe audio and optionally compute WER against reference text.
    """
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

    # Compute WER if a reference transcription is provided
    wer_result = None
    if reference_text is not None and reference_text.strip() != "":
        try:
            wer_result = compute_wer(hypothesis=text, reference=reference_text)
        except Exception as e:
            # Log the error but keep API resilient
            print(f"WER computation failed for request {req_id}: {e}")
            wer_result = {"wer": None, "error": str(e)}

    return JSONResponse(
        {
            "request_id": req_id,
            "text": text,
            "language": language,
            "decoder": decoder.lower(),
            "duration_ms": int(duration * 1000),
            "processing_ms": int((t1 - t0) * 1000),
            "model_version": os.environ.get("ASR_REVISION", "main"),
            "wer": (
                (wer_result.get("wer") if isinstance(wer_result, dict) else None)
                if wer_result is not None
                else None
            ),
            "wer_details": wer_result if isinstance(wer_result, dict) else None,
        }
    )
