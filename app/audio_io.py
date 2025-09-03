import os
import shlex
import subprocess
import tempfile
from typing import Tuple, Optional

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")


def probe_duration(input_path: str) -> float:
    cmd = (
        f"{shlex.quote(FFPROBE)} -v error -show_entries format=duration "
        f"-of default=noprint_wrappers=1:nokey=1 {shlex.quote(input_path)}"
    )
    out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    try:
        return float(out.decode().strip())
    except Exception:
        raise ValueError("Unable to detect audio duration")


def ensure_decodable(input_path: str) -> None:
    cmd = f"{shlex.quote(FFPROBE)} -v error -show_format -show_streams {shlex.quote(input_path)}"
    subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)


def to_wav16k_mono(input_path: str, output_path: str, normalize: bool = False) -> None:
    af = ["aformat=channel_layouts=mono", "aresample=16000"]
    if normalize:

        af.insert(0, "loudnorm=I=-23:TP=-2:LRA=11")
    af_str = ",".join(af)

    cmd = (
        f"{shlex.quote(FFMPEG)} -y -hide_banner -loglevel error "
        f"-i {shlex.quote(input_path)} "
        f"-vn -ac 1 -ar 16000 -sample_fmt s16 -af {shlex.quote(af_str)} "
        f"-c:a pcm_s16le {shlex.quote(output_path)}"
    )
    subprocess.check_call(cmd, shell=True)


def save_upload_to_tmp(data_iterable, suffix: str = "") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        for chunk in data_iterable:
            f.write(chunk)
    return path


def prepare_audio_for_model(
    uploaded_iterable,
    *,
    max_duration_sec: int,
    normalize: bool,
    tmpdir: Optional[str] = None,
) -> Tuple[str, float]:

    raw_path = save_upload_to_tmp(uploaded_iterable)
    try:

        ensure_decodable(raw_path)

        dur = probe_duration(raw_path)
        if dur > max_duration_sec:
            raise ValueError(f"Audio too long: {dur:.1f}s > {max_duration_sec}s")

        out_fd, out_path = tempfile.mkstemp(suffix=".wav", dir=tmpdir)
        os.close(out_fd)
        to_wav16k_mono(raw_path, out_path, normalize=normalize)

        return out_path, dur
    finally:
        try:
            os.remove(raw_path)
        except Exception:
            pass
