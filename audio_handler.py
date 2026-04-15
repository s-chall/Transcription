"""
audio_handler.py
Handles recording audio from a microphone and loading existing audio files.
"""

import os
import wave
import queue
import numpy as np
import sounddevice as sd
import torchaudio
import torch


SAMPLE_RATE = 16000  # Hz — Whisper works best at 16kHz
CHANNELS = 1
CHUNK_SIZE = 1024   # frames per callback — keeps memory usage flat


def record_audio(output_path: str, duration: int) -> str:
    """
    Record audio from the default microphone for `duration` seconds using
    a streaming callback so memory stays flat regardless of duration.
    Saves as a 16kHz mono WAV to `output_path`.
    Returns the path to the saved file.
    """
    print(f"Recording for {duration} seconds... (press nothing, it stops automatically)")

    audio_queue: queue.Queue = queue.Queue()

    def callback(indata, frames, time, status):
        audio_queue.put(indata.copy())

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=callback,
        ):
            total_frames = int(duration * SAMPLE_RATE)
            recorded = 0
            while recorded < total_frames:
                chunk = audio_queue.get()
                remaining = total_frames - recorded
                chunk = chunk[:remaining]
                wf.writeframes(chunk.tobytes())
                recorded += len(chunk)

    print(f"Saved recording to: {output_path}")
    return output_path


def load_audio(input_path: str, output_dir: str = "audio") -> str:
    """
    Load an existing audio file (mp3, m4a, wav, etc.) and convert it to
    a 16kHz mono WAV suitable for Whisper + pyannote.
    Returns the path to the converted WAV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base}_converted.wav")

    waveform, sr = torchaudio.load(input_path)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    torchaudio.save(output_path, waveform, SAMPLE_RATE)
    print(f"Converted and saved to: {output_path}")
    return output_path
