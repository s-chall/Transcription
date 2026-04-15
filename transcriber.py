"""
transcriber.py
Speech-to-text using OpenAI Whisper.
Returns word-level and segment-level transcripts with timestamps.
"""

import whisper
import numpy as np


def load_whisper_model(model_size: str = "base") -> whisper.Whisper:
    """
    Load a Whisper model.
    model_size options: tiny, base, small, medium, large
    Larger = more accurate but slower. 'base' is a good starting point.
    """
    print(f"Loading Whisper model: {model_size}")
    return whisper.load_model(model_size)


def transcribe(model: whisper.Whisper, audio_path: str) -> list[dict]:
    """
    Transcribe the audio file using Whisper with word-level timestamps.
    Returns a list of segments:
        [{"start": float, "end": float, "text": str}, ...]
    """
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        verbose=False,
    )

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
        })

    return segments
