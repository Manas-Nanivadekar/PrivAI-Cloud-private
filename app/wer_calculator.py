from typing import Optional, Dict, Any

from jiwer import (
    compute_measures,
    Compose,
    ToLowerCase,
    RemoveMultipleSpaces,
    Strip,
    RemovePunctuation,
)


_DEFAULT_TRANSFORM = Compose(
    [
        ToLowerCase(),
        RemovePunctuation(),
        RemoveMultipleSpaces(),
        Strip(),
    ]
)


def compute_wer(hypothesis: str, reference: str) -> Dict[str, Any]:
    """Compute Word Error Rate (WER) and breakdown using jiwer."""

    # Handle empty inputs
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
        # All words are deletions if hypothesis is empty
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
        measures = compute_measures(
            truth=reference,
            hypothesis=hypothesis,
            truth_transform=_DEFAULT_TRANSFORM,
            hypothesis_transform=_DEFAULT_TRANSFORM,
        )

        ref_len = measures.truth_words
        if ref_len == 0:
            return {
                "wer": None,
                "hits": 0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "reference_length": 0,
            }

        wer_value = (
            measures.substitutions + measures.deletions + measures.insertions
        ) / ref_len

        return {
            "wer": float(wer_value),
            "hits": int(measures.hits),
            "substitutions": int(measures.substitutions),
            "deletions": int(measures.deletions),
            "insertions": int(measures.insertions),
            "reference_length": int(ref_len),
        }
    except Exception as e:
        return {"wer": None, "error": f"WER computation failed: {str(e)}"}
