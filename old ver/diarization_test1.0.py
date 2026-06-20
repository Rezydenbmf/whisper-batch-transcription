from pathlib import Path
import os
import sys

import torch
from pyannote.audio import Pipeline


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

    print(f"Analiza pliku: {audio_path}")
    diarization = pipeline(str(audio_path))

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
