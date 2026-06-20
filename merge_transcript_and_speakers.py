from pathlib import Path
import argparse
import os

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


def build_speaker_segments(diarization):
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })
    return segments


def pick_speaker(start, end, speaker_segments):
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for seg in speaker_segments:
        overlap = min(end, seg["end"]) - max(start, seg["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]

    return best_speaker


def parse_time_to_seconds(value: str | None):
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Nieprawidłowy czas: {value}") from exc
    raise argparse.ArgumentTypeError(f"Nieprawidłowy format czasu: {value}")


def format_seconds(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Transkrypcja + diarization z opcjonalnym zakresem czasu."
    )
    parser.add_argument("audio_path", help="Ścieżka do pliku audio")
    parser.add_argument("--start", type=parse_time_to_seconds, default=None, help="Początek zakresu, np. 75, 12:30, 01:02:03")
    parser.add_argument("--end", type=parse_time_to_seconds, default=None, help="Koniec zakresu, np. 140, 18:10, 01:10:30")
    parser.add_argument("--model", default="medium", help="Model Whisper, np. small, medium, large, turbo")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    audio_path = Path(args.audio_path).resolve()
    if not audio_path.exists():
        print(f"Brak pliku: {audio_path}")
        return 1

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("Brak HF_TOKEN w środowisku.")
        return 1

    if args.start is not None and args.start < 0:
        print("Parametr --start nie może być ujemny.")
        return 1
    if args.end is not None and args.end < 0:
        print("Parametr --end nie może być ujemny.")
        return 1
    if args.start is not None and args.end is not None and args.end <= args.start:
        print("Parametr --end musi być większy niż --start.")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Urządzenie: {device}")

    print("Wczytywanie audio przez whisper/ffmpeg...")
    audio = whisper.load_audio(str(audio_path))

    offset = args.start or 0.0
    if args.start is not None or args.end is not None:
        start_sample = int((args.start or 0.0) * whisper.audio.SAMPLE_RATE)
        end_sample = int(args.end * whisper.audio.SAMPLE_RATE) if args.end is not None else len(audio)
        audio = audio[start_sample:end_sample]
        if audio.size == 0:
            print("Wybrany zakres czasu dał pusty fragment audio.")
            return 1
        print(
            f"Używam zakresu: {format_seconds(offset)} -> "
            f"{format_seconds((args.end if args.end is not None else offset + len(audio)/whisper.audio.SAMPLE_RATE))}"
        )

    waveform = torch.from_numpy(audio).unsqueeze(0)

    print("Ładowanie modelu diarization...")
    diar_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )
    diar_pipeline.to(torch.device(device))

    print("Uruchamianie diarization...")
    diar_result = diar_pipeline({"waveform": waveform, "sample_rate": 16000})
    diarization = extract_annotation(diar_result)
    speaker_segments = build_speaker_segments(diarization)

    print("Uruchamianie transkrypcji Whisper...")
    model = whisper.load_model(args.model, device=device)
    result = model.transcribe(audio, language="pl", task="transcribe", verbose=False)

    suffix_extra = ""
    if args.start is not None or args.end is not None:
        start_label = format_seconds(args.start or 0.0).replace(":", "-").replace(".", "_")
        end_value = args.end if args.end is not None else offset + len(audio) / whisper.audio.SAMPLE_RATE
        end_label = format_seconds(end_value).replace(":", "-").replace(".", "_")
        suffix_extra = f".clip_{start_label}_to_{end_label}"

    out_path = audio_path.with_suffix(audio_path.suffix + f"{suffix_extra}.merged.txt")
    with out_path.open("w", encoding="utf-8") as f:
        for seg in result.get("segments", []):
            start = float(seg["start"]) + offset
            end = float(seg["end"]) + offset
            text = seg.get("text", "").strip()
            if not text:
                continue
            speaker = pick_speaker(start, end, speaker_segments)
            line = f"[{start:08.2f} - {end:08.2f}] [{speaker}] {text}"
            print(line)
            f.write(line + "\n")

    print(f"Zapisano: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
