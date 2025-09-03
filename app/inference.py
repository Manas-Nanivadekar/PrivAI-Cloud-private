import os
import asyncio
from typing import Optional, Literal, Dict, Any

import torch
import torchaudio
from transformers import AutoModel


REPO_ID = os.environ.get("ASR_REPO_ID", "ai4bharat/indic-conformer-600m-multilingual")
REVISION = os.environ.get("ASR_REVISION", "main")


SUPPORTED_LANGS = {
    "as",
    "bn",
    "brx",
    "doi",
    "gu",
    "hi",
    "kn",
    "kok",
    "ks",
    "mai",
    "ml",
    "mni",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "sat",
    "sd",
    "ta",
    "te",
    "ur",
}

Decoder = Literal["ctc", "rnnt"]


class IndicASR:
    def __init__(self) -> None:
        self.model = None
        self.ready = False

    def load(self) -> None:

        self.model = AutoModel.from_pretrained(
            REPO_ID, trust_remote_code=True, revision=REVISION
        )

        wav = torch.zeros(1, 8000, dtype=torch.float32)
        _ = self.model(wav, "hi", "ctc")
        self.ready = True

    def transcribe(
        self,
        wav16k_path: str,
        *,
        language: str = "hi",
        decoder: Decoder = "rnnt",
    ) -> str:
        if self.model is None or not self.ready:
            raise RuntimeError("Model not ready")

        if language not in SUPPORTED_LANGS:
            raise ValueError(f"Unsupported language '{language}'")

        wav, sr = torchaudio.load(wav16k_path)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)

        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)

        text = self.model(wav, language, decoder)
        if isinstance(text, (list, tuple)):
            text = " ".join(map(str, text))
        return str(text)


engine = IndicASR()
