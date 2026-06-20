# whis — Whisper Batch Transcriber

A local Windows desktop tool for batch audio transcription with optional speaker diarization. Built around OpenAI Whisper (GPU-accelerated) and pyannote.audio, with a tkinter GUI for folder-based batch runs and standalone CLI scripts for single-file processing.

## What it does

The GUI app (`whisper_batch_gui.py`) scans one or more folders for audio files, transcribes each with Whisper, and optionally identifies who is speaking in each segment. Output is written next to the source audio file — no cloud, no internet required for transcription (only the initial HuggingFace model download needs a token).

Speaker diarization produces timestamped `.merged.txt` files in the format:

```
[00000.82 - 00014.31] [SPEAKER_00] Dzień dobry, mamy dzisiaj...
[00015.44 - 00028.06] [SPEAKER_01] Tak, zgadzam się w pełni.
```

## Key features

- **Batch folder processing** — add multiple input folders; app scans all of them in one run
- **7 audio formats** — wav, m4a, mp3, flac, ogg, aac, wma
- **Whisper medium model** — balanced quality/speed; supports small, large, turbo via dropdown
- **Polish language optimized** — language set to `pl` by default; any Whisper-supported language works
- **GPU support** — automatically uses CUDA if available, falls back to CPU
- **Speaker labelling** — optional pyannote diarization adds `[SPEAKER_00]` / `[SPEAKER_01]` labels with timestamps to a separate `.merged.txt` file alongside the plain transcript
- **Time-range clipping** — transcribe only a segment of a file (start/end in `ss`, `mm:ss`, or `hh:mm:ss`)
- **Skip-existing mode** — skips any file whose `.txt` output already exists, safe to re-run on growing folders
- **HuggingFace token via masked GUI field** — token is entered once in the app, saved only to the session environment variable, never written to disk or hardcoded

## Tech stack

| Component | Technology |
|-----------|-----------|
| GUI | Python, tkinter |
| Transcription | OpenAI Whisper (`openai-whisper`) |
| Speaker diarization | pyannote.audio 4.0.4 (`pyannote/speaker-diarization-community-1`) |
| GPU acceleration | PyTorch 2.5.1 + CUDA 12.1 |
| Audio I/O | FFmpeg (system PATH), torchaudio |
| Runtime | Python 3.11, local `.venv` |

## How to run

### Prerequisites

- Python 3.11+
- FFmpeg on system PATH
- NVIDIA GPU + CUDA drivers recommended (CPU fallback works but is slow)
- HuggingFace account with access to `pyannote/speaker-diarization-community-1`

### Setup

```bat
python -m venv .venv
.venv\Scripts\pip install openai-whisper pyannote.audio torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Launch GUI

```bat
start.bat
```

Or directly:

```bat
.venv\Scripts\python.exe whisper_batch_gui.py
```

### In the GUI

1. Click **Odśwież status** (Refresh status) — all indicators should show green
2. If using diarization: paste your HuggingFace token into the HF_TOKEN field and click **Zapisz token do sesji**
3. Click **Dodaj folder** (Add folder) to select your audio folder(s)
4. Choose model size and language
5. Optionally enable **Rozpoznawanie mówców** (Speaker diarization)
6. Click **Skanuj i transkrybuj** (Scan and transcribe)

### CLI variants

```bat
# Transcription + diarization, single file
.venv\Scripts\python.exe merge_transcript_and_speakers.py audio.wav --model medium --start 1:30 --end 5:00

# Diarization only (speaker segments, no transcript text)
.venv\Scripts\python.exe diarization_test.py audio.wav
```

## Project structure

```
whisper_batch_gui.py          Main GUI application
merge_transcript_and_speakers.py  CLI: transcription + diarization for one file
diarization_test.py           CLI: diarization only, for testing
start.bat                     Self-relative launcher
old ver/                      Earlier versions kept for reference
  whisper_batch_gui 1.1.py
  diarization_test1.0.py
```

## Notes on GGML binaries

The whisper.cpp C++ release binaries (`whisper-cli.exe`, `whisper-server.exe`, etc.) and GGML model files (`models/ggml-base.bin`, `models/ggml-small.bin`) are **not included** in this repository. The Python application uses the `openai-whisper` Python package independently of whisper.cpp. If you want the C++ CLI tools, download a whisper.cpp Windows release from the project's GitHub releases page and place it alongside these files.

## Portfolio note

Personal tool built for transcribing meeting recordings and interviews locally on Windows. Processes Polish-language audio with GPU acceleration and produces both plain transcripts and speaker-labelled merged files. No audio data, transcription outputs, or tokens are included in this repository.
