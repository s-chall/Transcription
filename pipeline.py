"""
pipeline.py
Merges speaker diarization segments with Whisper transcription segments
to produce a timestamped, speaker-labeled transcript.
"""


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return the duration of overlap between two time intervals."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(
    transcription_segments: list[dict],
    diarization_segments: list[dict],
) -> list[dict]:
    """
    For each Whisper transcription segment, find the speaker with the
    most overlapping diarization time and assign them.

    Returns a list of merged entries:
        [{"start": float, "end": float, "speaker": str, "text": str}, ...]
    """
    result = []

    for tseg in transcription_segments:
        t_start, t_end, text = tseg["start"], tseg["end"], tseg["text"]

        # Find the diarization segment with the most overlap
        best_speaker = None
        best_overlap = 0.0

        for dseg in diarization_segments:
            ov = _overlap(t_start, t_end, dseg["start"], dseg["end"])
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = dseg["speaker"]

        # No overlap found — fall back to the nearest diarization segment by time
        if best_speaker is None and diarization_segments:
            t_mid = (t_start + t_end) / 2
            best_speaker = min(
                diarization_segments,
                key=lambda d: min(
                    abs(t_mid - d["start"]),
                    abs(t_mid - d["end"]),
                )
            )["speaker"]

        result.append({
            "start": t_start,
            "end": t_end,
            "speaker": best_speaker or "Unknown",
            "text": text,
        })

    return result


def merge_consecutive(entries: list[dict]) -> list[dict]:
    """
    Merge back-to-back entries from the same speaker into a single entry
    to produce a cleaner, more readable transcript.
    """
    if not entries:
        return []

    merged = [entries[0].copy()]
    for entry in entries[1:]:
        last = merged[-1]
        if entry["speaker"] == last["speaker"]:
            last["end"] = entry["end"]
            last["text"] = last["text"] + " " + entry["text"]
        else:
            merged.append(entry.copy())

    return merged
