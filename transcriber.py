import whisper
import numpy as np


def load_whisper_model(model_size: str = "base") -> whisper.Whisper:
    print(f"Loading Whisper model: {model_size}")
    return whisper.load_model(model_size)


def transcribe(model: whisper.Whisper, audio_path: str) -> list[dict]:
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
