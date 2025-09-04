# app/wer_calculator.py

from typing import Dict, Any
from jiwer import (
    process_words,
    Compose,
    ToLowerCase,
    RemoveMultipleSpaces,
    Strip,
    RemovePunctuation,
)

# Normalization pipeline applied to both reference and hypothesis
_DEFAULT_TRANSFORM = Compose(
    [
        ToLowerCase(),
        RemovePunctuation(),
        RemoveMultipleSpaces(),
        Strip(),
    ]
)


def compute_wer(hypothesis: str, reference: str) -> Dict[str, Any]:
    """
    Compute Word Error Rate (WER) and breakdown using jiwer (v3+ API).
    - Applies a standard text normalization to both strings.
    - Handles empty-input edge cases explicitly.
    Returns a dict with: wer, hits, substitutions, deletions, insertions, reference_length, (optional) error.
    """

    # Guard against None
    hypothesis = hypothesis or ""
    reference = reference or ""

    # Pre-transform empty checks (as in your original behavior)
    if not reference.strip():
        return {
            "wer": None,
            "hits": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "reference_length": 0,
            "error": "Reference text is empty",
        }

    if not hypothesis.strip():
        ref_words = len(reference.strip().split())
        return {
            "wer": 1.0,
            "hits": 0,
            "substitutions": 0,
            "deletions": ref_words,
            "insertions": 0,
            "reference_length": ref_words,
        }

    try:
        # Apply the same transform to both sides (v3+ doesn't take transform kwargs)
        ref_proc = _DEFAULT_TRANSFORM(reference)
        hyp_proc = _DEFAULT_TRANSFORM(hypothesis)

        # If the transform wipes out the reference, treat as empty-reference condition
        if not ref_proc.strip():
            return {
                "wer": None,
                "hits": 0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "reference_length": 0,
                "error": "Reference text is empty after normalization",
            }

        # If hypothesis becomes empty after normalization, count as all deletions
        if not hyp_proc.strip():
            ref_words = len(ref_proc.split())
            return {
                "wer": 1.0,
                "hits": 0,
                "substitutions": 0,
                "deletions": ref_words,
                "insertions": 0,
                "reference_length": ref_words,
            }

        # Compute word-level alignment and error metrics
        out = process_words(ref_proc, hyp_proc)

        # Reference length = hits + substitutions + deletions
        ref_len = int(out.hits + out.substitutions + out.deletions)

        # Safety: avoid divide-by-zero if something odd happened
        if ref_len == 0:
            return {
                "wer": None,
                "hits": 0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": int(out.insertions),
                "reference_length": 0,
                "error": "Reference length computed as zero",
            }

        return {
            "wer": float(out.wer),  # jiwer computes (S + D + I) / N internally
            "hits": int(out.hits),
            "substitutions": int(out.substitutions),
            "deletions": int(out.deletions),
            "insertions": int(out.insertions),
            "reference_length": ref_len,
        }

    except Exception as e:
        return {"wer": None, "error": f"WER computation failed: {str(e)}"}
