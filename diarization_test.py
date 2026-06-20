from pathlib import Path
import os
import sys

import torch
import whisper
from pyannote.audio import Pipeline


def extract_annotation(result):
    if hasattr(result, "itertracks"):
        return result
    if hasattr(result, "speaker_diarization"):
        return result.speaker_diarization
    if hasattr(result, "annotation"):
        return result.annotation
    if isinstance(result, dict):
        for key in ("speaker_diarization", "annotation", "diarization"):
            value = result.get(key)
            if value is not None and hasattr(value, "itertracks"):
                return value
    raise TypeError(f"Nieznany typ wyniku diarization: {type(result)}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Użycie: python diarization_test.py <plik_audio>")
        return 1

    audio_path = Path(sys.argv[1]).resolve()
    if not audio_path.exists():
        print(f"Brak pliku: {audio_path}")
        return 1

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("Brak HF_TOKEN w środowisku.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")
    print("Ładowanie pipeline diarization...")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )
    pipeline.to(device)

    print(f"Wczytywanie audio przez whisper/ffmpeg: {audio_path}")
    audio = whisper.load_audio(str(audio_path))
    waveform = torch.from_numpy(audio).unsqueeze(0)

    print(f"Analiza pliku: {audio_path}")
    result = pipeline({"waveform": waveform, "sample_rate": 16000})
    diarization = extract_annotation(result)

    out_path = audio_path.with_suffix(audio_path.suffix + ".speakers.txt")
    with out_path.open("w", encoding="utf-8") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            line = f"[{turn.start:08.2f} - {turn.end:08.2f}] {speaker}"
            print(line)
            f.write(line + "\n")

    print(f"Zapisano: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
