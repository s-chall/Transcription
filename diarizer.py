"""
diarizer.py
Speaker diarization using pyannote.audio.
Identifies WHO spoke and WHEN, returning labeled time segments.
"""

import os
from pyannote.audio import Pipeline
from pyannote.core import Annotation


def load_diarization_pipeline(hf_token: str) -> Pipeline:
    """
    Load the pyannote speaker diarization pipeline.
    Requires a HuggingFace token with access to pyannote/speaker-diarization-3.1.
    Request access at: https://huggingface.co/pyannote/speaker-diarization-3.1
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    return pipeline


def diarize(pipeline: Pipeline, audio_path: str, num_speakers: int = None, min_speakers: int = None, max_speakers: int = None) -> list[dict]:
    """
    Run diarization on the audio file.
    Optionally constrain speaker count for better accuracy:
      - num_speakers: exact number of speakers (most accurate when known)
      - min_speakers / max_speakers: loose bounds when count is uncertain
    Returns a list of segments:
        [{"start": float, "end": float, "speaker": "SPEAKER_00"}, ...]
    """
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    diarization: Annotation = pipeline(audio_path, **kwargs)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,  # e.g. "SPEAKER_00", "SPEAKER_01"
        })

    # Sort by start time
    segments.sort(key=lambda s: s["start"])
    return segments


def normalize_speaker_labels(segments: list[dict]) -> list[dict]:
    """
    Remap pyannote speaker IDs (SPEAKER_00, SPEAKER_01, ...) to
    human-friendly labels (Speaker 1, Speaker 2, ...) in order of first appearance.
    """
    label_map = {}
    counter = 1
    for seg in segments:
        raw = seg["speaker"]
        if raw not in label_map:
            label_map[raw] = f"Speaker {counter}"
            counter += 1
        seg["speaker"] = label_map[raw]
    return segments
