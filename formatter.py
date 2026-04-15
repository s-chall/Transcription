"""
formatter.py
Formats the merged transcript entries for display and file output.
"""

import json
import os


def _fmt_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def format_transcript(entries: list[dict]) -> str:
    """
    Produce a human-readable transcript string.

    Example output:
        [00:00:01.200 --> 00:00:05.400]  Speaker 1:
        Hello, how are you doing today?

        [00:00:06.100 --> 00:00:09.800]  Speaker 2:
        I'm doing well, thanks for asking.
    """
    lines = []
    for entry in entries:
        timestamp = f"[{_fmt_time(entry['start'])} --> {_fmt_time(entry['end'])}]"
        lines.append(f"{timestamp}  {entry['speaker']}:")
        lines.append(f"  {entry['text']}")
        lines.append("")
    return "\n".join(lines)


def clear_transcripts(output_dir: str = "transcripts") -> None:
    """
    Delete all .txt and .json transcript files in the output directory.
    """
    if not os.path.exists(output_dir):
        print(f"No transcripts directory found at '{output_dir}', nothing to clear.")
        return

    removed = []
    for filename in os.listdir(output_dir):
        if filename.endswith(".txt") or filename.endswith(".json"):
            filepath = os.path.join(output_dir, filename)
            os.remove(filepath)
            removed.append(filename)

    if removed:
        print(f"Cleared {len(removed)} transcript file(s) from '{output_dir}':")
        for f in removed:
            print(f"  - {f}")
    else:
        print(f"No transcript files found in '{output_dir}'.")


def save_transcript(entries: list[dict], output_dir: str = "transcripts", base_name: str = "transcript"):
    """
    Save the transcript in two formats:
      - <base_name>.txt  — human-readable
      - <base_name>.json — machine-readable (includes raw start/end times)
    """
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    json_path = os.path.join(output_dir, f"{base_name}.json")

    readable = format_transcript(entries)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(readable)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"Transcript saved:\n  {txt_path}\n  {json_path}")
    return txt_path, json_path
