import os
import queue
import shutil
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import torch
import whisper
from pyannote.audio import Pipeline

AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac", ".wma"}
STATUS_OK = "OK"
STATUS_WARN = "BRAK / NIE GOTOWE"


def parse_time_to_seconds(value: str):
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
    except ValueError:
        return "ERROR"
    return "ERROR"


def format_seconds(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


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


class WhisperBatchApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Whisper Batch Transcriber (GPU)")
        self.root.geometry("1120x820")

        self.base_dir = Path(__file__).resolve().parent
        self.python_exe = self.base_dir / ".venv" / "Scripts" / "python.exe"
        self.stop_requested = False
        self.worker_thread = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.folders: list[str] = []
        self.status_labels: dict[str, ttk.Label] = {}
        self.whisper_model_cache = {}
        self.diarization_pipeline = None

        self.model_var = tk.StringVar(value="medium")
        self.language_var = tk.StringVar(value="Polish")
        self.skip_existing_var = tk.BooleanVar(value=True)
        self.recursive_var = tk.BooleanVar(value=True)
        self.extensions_var = tk.StringVar(value="wav,m4a,mp3,flac")
        self.enable_diarization_var = tk.BooleanVar(value=False)
        self.hf_token_var = tk.StringVar(value=os.environ.get("HF_TOKEN", ""))
        self.start_time_var = tk.StringVar(value="")
        self.end_time_var = tk.StringVar(value="")

        self._build_ui()
        self._poll_log_queue()
        self.refresh_statuses()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        info = (
            f"Folder aplikacji: {self.base_dir}\n"
            f"Python środowiska: {self.python_exe}"
        )
        ttk.Label(top, text=info, justify="left").pack(anchor="w")

        status_frame = ttk.LabelFrame(self.root, text="Status składników", padding=10)
        status_frame.pack(fill="x", padx=10, pady=8)
        grid = ttk.Frame(status_frame)
        grid.pack(fill="x")

        statuses = [
            ("python_env", "Python .venv"),
            ("ffmpeg", "FFmpeg w PATH"),
            ("whisper", "Pakiet whisper"),
            ("torch", "Pakiet torch"),
            ("cuda", "CUDA / GPU dostępne"),
            ("pyannote", "Pakiet pyannote.audio"),
            ("hf_token", "HF_TOKEN ustawiony"),
        ]

        for row, (key, title) in enumerate(statuses):
            ttk.Label(grid, text=title + ":", width=24).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            lbl = ttk.Label(grid, text="sprawdzam...")
            lbl.grid(row=row, column=1, sticky="w", pady=2)
            self.status_labels[key] = lbl

        ttk.Button(status_frame, text="Odśwież status", command=self.refresh_statuses).pack(anchor="w", pady=(8, 0))

        folders_frame = ttk.LabelFrame(self.root, text="Foldery do skanowania", padding=10)
        folders_frame.pack(fill="x", padx=10, pady=8)
        btn_row = ttk.Frame(folders_frame)
        btn_row.pack(fill="x", pady=(0, 8))
        ttk.Button(btn_row, text="Dodaj folder", command=self.add_folder).pack(side="left")
        ttk.Button(btn_row, text="Usuń zaznaczony", command=self.remove_selected_folder).pack(side="left", padx=6)
        ttk.Button(btn_row, text="Wyczyść listę", command=self.clear_folders).pack(side="left")
        self.folder_listbox = tk.Listbox(folders_frame, height=6, selectmode=tk.SINGLE)
        self.folder_listbox.pack(fill="x")

        options_frame = ttk.LabelFrame(self.root, text="Opcje", padding=10)
        options_frame.pack(fill="x", padx=10, pady=8)

        row1 = ttk.Frame(options_frame)
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="Model:").pack(side="left")
        ttk.Combobox(row1, textvariable=self.model_var, values=["small", "medium", "large", "turbo"], width=12, state="readonly").pack(side="left", padx=(6, 16))
        ttk.Label(row1, text="Język:").pack(side="left")
        ttk.Entry(row1, textvariable=self.language_var, width=18).pack(side="left", padx=(6, 16))
        ttk.Label(row1, text="Rozszerzenia:").pack(side="left")
        ttk.Entry(row1, textvariable=self.extensions_var, width=28).pack(side="left", padx=(6, 0))

        row2 = ttk.Frame(options_frame)
        row2.pack(fill="x", pady=4)
        ttk.Checkbutton(row2, text="Skanuj podfoldery", variable=self.recursive_var).pack(side="left")
        ttk.Checkbutton(row2, text="Pomiń plik, jeśli TXT już istnieje", variable=self.skip_existing_var).pack(side="left", padx=16)

        row3 = ttk.Frame(options_frame)
        row3.pack(fill="x", pady=4)
        ttk.Label(row3, text="Start zakresu:").pack(side="left")
        ttk.Entry(row3, textvariable=self.start_time_var, width=14).pack(side="left", padx=(6, 12))
        ttk.Label(row3, text="Koniec zakresu:").pack(side="left")
        ttk.Entry(row3, textvariable=self.end_time_var, width=14).pack(side="left", padx=(6, 12))
        ttk.Label(row3, text="Format: ss lub mm:ss lub hh:mm:ss").pack(side="left")

        diar_frame = ttk.LabelFrame(self.root, text="Rozpoznawanie mówców", padding=10)
        diar_frame.pack(fill="x", padx=10, pady=8)
        ttk.Checkbutton(
            diar_frame,
            text="Włącz diarization (etykiety SPEAKER_00 / SPEAKER_01)",
            variable=self.enable_diarization_var,
        ).pack(anchor="w")

        hf_row = ttk.Frame(diar_frame)
        hf_row.pack(fill="x", pady=(8, 0))
        ttk.Label(hf_row, text="HF_TOKEN:").pack(side="left")
        ttk.Entry(hf_row, textvariable=self.hf_token_var, width=70, show="*").pack(side="left", padx=(6, 8), fill="x", expand=True)
        ttk.Button(hf_row, text="Zapisz token do sesji", command=self.apply_hf_token).pack(side="left")

        ttk.Label(
            diar_frame,
            text=(
                "Po włączeniu diarization aplikacja zapisze dodatkowy plik *.merged.txt z czasem, etykietą SPEAKER_xx i tekstem. "
                "Jeśli ustawisz Start/Koniec, przetwarzany będzie tylko wskazany fragment."
            ),
            wraplength=980,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        actions_frame = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        actions_frame.pack(fill="x", pady=8)
        self.start_button = ttk.Button(actions_frame, text="Skanuj i transkrybuj", command=self.start_processing)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions_frame, text="Zatrzymaj po bieżącym pliku", command=self.request_stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.scan_button = ttk.Button(actions_frame, text="Podgląd listy plików", command=self.preview_files)
        self.scan_button.pack(side="left")
        self.progress_label = ttk.Label(actions_frame, text="Gotowy.")
        self.progress_label.pack(side="left", padx=16)

        log_frame = ttk.LabelFrame(self.root, text="Log", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_box = ScrolledText(log_frame, wrap="word", height=20)
        self.log_box.pack(fill="both", expand=True)
        self.log_box.configure(state="disabled")

    def apply_hf_token(self):
        token = self.hf_token_var.get().strip()
        if token:
            os.environ["HF_TOKEN"] = token
            self.log("HF_TOKEN zapisany do bieżącej sesji aplikacji.")
        else:
            os.environ.pop("HF_TOKEN", None)
            self.log("HF_TOKEN usunięty z bieżącej sesji aplikacji.")
        self.refresh_statuses()

    def set_status(self, key: str, ok: bool, details: str = ""):
        label = self.status_labels[key]
        text = STATUS_OK if ok else STATUS_WARN
        if details:
            text += f" | {details}"
        color = "#0a7a22" if ok else "#b35a00"
        label.config(text=text, foreground=color)

    def _run_python_check(self, code: str):
        if not self.python_exe.exists():
            return False, "python.exe nie istnieje"
        try:
            import subprocess
            result = subprocess.run(
                [str(self.python_exe), "-c", code],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(self.base_dir),
                env=os.environ.copy(),
            )
        except Exception as exc:
            return False, str(exc)
        output = (result.stdout or "").strip() or (result.stderr or "").strip()
        return result.returncode == 0, output

    def refresh_statuses(self):
        self.set_status("python_env", self.python_exe.exists(), str(self.python_exe.name) if self.python_exe.exists() else "brak .venv")
        ffmpeg_path = shutil.which("ffmpeg")
        self.set_status("ffmpeg", bool(ffmpeg_path), ffmpeg_path or "ffmpeg nie znaleziony")
        ok, out = self._run_python_check("import whisper; print(getattr(whisper, '__file__', 'whisper ok'))")
        self.set_status("whisper", ok, out if out else "")
        ok, out = self._run_python_check("import torch; print(torch.__version__)")
        self.set_status("torch", ok, out if out else "")
        ok, out = self._run_python_check("import torch; print('GPU:' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA niedostępne')")
        self.set_status("cuda", ok and out.startswith("GPU:"), out if out else "")
        ok, out = self._run_python_check("import pyannote.audio; print(getattr(pyannote.audio, '__version__', 'pyannote ok'))")
        self.set_status("pyannote", ok, out if out else "")
        token = self.hf_token_var.get().strip() or os.environ.get("HF_TOKEN", "")
        self.set_status("hf_token", bool(token), "ustawiony" if token else "brak tokenu")

    def add_folder(self):
        folder = filedialog.askdirectory(title="Wybierz folder z plikami audio")
        if folder and folder not in self.folders:
            self.folders.append(folder)
            self.folder_listbox.insert(tk.END, folder)

    def remove_selected_folder(self):
        selection = self.folder_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        self.folder_listbox.delete(index)
        del self.folders[index]

    def clear_folders(self):
        self.folder_listbox.delete(0, tk.END)
        self.folders.clear()

    def log(self, message: str):
        self.log_queue.put(message)

    def _poll_log_queue(self):
        while not self.log_queue.empty():
            message = self.log_queue.get()
            self.log_box.configure(state="normal")
            self.log_box.insert(tk.END, message + "\n")
            self.log_box.see(tk.END)
            self.log_box.configure(state="disabled")
        self.root.after(150, self._poll_log_queue)

    def get_extensions(self):
        raw = self.extensions_var.get().strip()
        if not raw:
            return AUDIO_EXTENSIONS
        result = set()
        for item in raw.split(","):
            ext = item.strip().lower()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = "." + ext
            result.add(ext)
        return result or AUDIO_EXTENSIONS

    def get_time_range(self):
        start_raw = self.start_time_var.get().strip()
        end_raw = self.end_time_var.get().strip()
        start = parse_time_to_seconds(start_raw)
        end = parse_time_to_seconds(end_raw)
        if start == "ERROR" or end == "ERROR":
            raise ValueError("Nieprawidłowy format czasu. Użyj ss, mm:ss lub hh:mm:ss")
        if start is not None and start < 0:
            raise ValueError("Start zakresu nie może być ujemny")
        if end is not None and end < 0:
            raise ValueError("Koniec zakresu nie może być ujemny")
        if start is not None and end is not None and end <= start:
            raise ValueError("Koniec zakresu musi być większy niż początek")
        return start, end

    def discover_files(self):
        exts = self.get_extensions()
        files = []
        for folder in self.folders:
            base = Path(folder)
            iterator = base.rglob("*") if self.recursive_var.get() else base.glob("*")
            for path in iterator:
                if path.is_file() and path.suffix.lower() in exts:
                    files.append(path)
        files.sort()
        return files

    def preview_files(self):
        if not self.folders:
            messagebox.showwarning("Brak folderów", "Najpierw dodaj co najmniej jeden folder.")
            return
        files = self.discover_files()
        self.log(f"Znaleziono {len(files)} plików audio.")
        for path in files[:100]:
            self.log(f"  {path}")
        if len(files) > 100:
            self.log("  ... i więcej")
        self.progress_label.config(text=f"Podgląd: {len(files)} plików")

    def request_stop(self):
        self.stop_requested = True
        self.log("Żądanie zatrzymania przyjęte. Zatrzymam po bieżącym pliku.")

    def start_processing(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Praca w toku", "Transkrypcja już trwa.")
            return
        if not self.python_exe.exists():
            messagebox.showerror("Brak środowiska", f"Nie znaleziono: {self.python_exe}")
            return
        if not self.folders:
            messagebox.showwarning("Brak folderów", "Dodaj co najmniej jeden folder.")
            return
        try:
            self.get_time_range()
        except ValueError as exc:
            messagebox.showerror("Błędny zakres czasu", str(exc))
            return

        files = self.discover_files()
        if not files:
            messagebox.showwarning("Brak plików", "Nie znaleziono plików audio w wybranych folderach.")
            return

        if self.enable_diarization_var.get():
            token = self.hf_token_var.get().strip() or os.environ.get("HF_TOKEN", "")
            if not token:
                messagebox.showerror("Brak HF_TOKEN", "Włączono diarization, ale nie ustawiono HF_TOKEN.")
                return
            self.log("Diarization włączone: będzie zapisany dodatkowy plik *.merged.txt.")

        self.stop_requested = False
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.progress_label.config(text=f"Start: {len(files)} plików")
        self.log(f"Start transkrypcji. Liczba plików: {len(files)}")
        self.log(f"Model: {self.model_var.get()}, język: {self.language_var.get()}")

        self.worker_thread = threading.Thread(target=self._process_files, args=(files,), daemon=True)
        self.worker_thread.start()

    def _process_files(self, files):
        total = len(files)
        completed = 0
        skipped = 0
        failed = 0

        for index, audio_path in enumerate(files, start=1):
            if self.stop_requested:
                break

            txt_path = audio_path.parent / f"{audio_path.stem}.txt"
            merged_path = audio_path.with_suffix(audio_path.suffix + ".merged.txt")
            if self.skip_existing_var.get() and txt_path.exists() and (not self.enable_diarization_var.get() or merged_path.exists()):
                skipped += 1
                self.log(f"[{index}/{total}] POMINIĘTO (wynik istnieje): {audio_path}")
                self._update_progress_safe(f"Pominięto: {index}/{total}")
                continue

            self.log(f"[{index}/{total}] START: {audio_path}")
            ok, details = self.process_audio(audio_path)
            if ok:
                completed += 1
                self.log(f"[{index}/{total}] OK: {audio_path}")
            else:
                failed += 1
                self.log(f"[{index}/{total}] BŁĄD: {audio_path}")
                self.log(details)

            self._update_progress_safe(f"Gotowe: {index}/{total}")

        summary = f"KONIEC | OK: {completed} | POMINIĘTE: {skipped} | BŁĘDY: {failed}"
        if self.stop_requested:
            summary = "ZATRZYMANO | " + summary
        self.log(summary)
        self.root.after(0, self._finish_ui, summary)

    def _update_progress_safe(self, text: str):
        self.root.after(0, lambda: self.progress_label.config(text=text))

    def _finish_ui(self, summary: str):
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress_label.config(text=summary)

    def get_whisper_model(self):
        model_name = self.model_var.get()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        key = (model_name, device)
        if key not in self.whisper_model_cache:
            self.log(f"Ładowanie modelu Whisper: {model_name} na {device}")
            self.whisper_model_cache[key] = whisper.load_model(model_name, device=device)
        return self.whisper_model_cache[key], device

    def get_diarization_pipeline(self):
        if self.diarization_pipeline is not None:
            return self.diarization_pipeline
        token = self.hf_token_var.get().strip() or os.environ.get("HF_TOKEN", "")
        if not token:
            raise RuntimeError("Brak HF_TOKEN dla diarization.")
        self.log("Ładowanie pipeline diarization...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=token,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        self.diarization_pipeline = pipeline
        return pipeline

    def process_audio(self, audio_path: Path):
        try:
            start_time, end_time = self.get_time_range()
            model, device = self.get_whisper_model()
            language_value = self.language_var.get().strip()
            language_arg = "pl" if language_value.lower() in {"polish", "pl", "polski"} else language_value

            self.log(f"Wczytywanie audio: {audio_path}")
            audio = whisper.load_audio(str(audio_path))
            offset = start_time or 0.0
            clip_suffix = ""
            if start_time is not None or end_time is not None:
                start_sample = int((start_time or 0.0) * whisper.audio.SAMPLE_RATE)
                end_sample = int(end_time * whisper.audio.SAMPLE_RATE) if end_time is not None else len(audio)
                audio = audio[start_sample:end_sample]
                if audio.size == 0:
                    raise RuntimeError("Wybrany zakres czasu dał pusty fragment audio.")
                end_value = end_time if end_time is not None else offset + len(audio) / whisper.audio.SAMPLE_RATE
                self.log(f"Zakres: {format_seconds(offset)} -> {format_seconds(end_value)}")
                clip_suffix = f".clip_{format_seconds(offset).replace(':', '-').replace('.', '_')}_to_{format_seconds(end_value).replace(':', '-').replace('.', '_')}"

            self.log(f"Transkrypcja: {audio_path} | device={device}")
            transcript_result = model.transcribe(
                audio,
                language=language_arg,
                task="transcribe",
                verbose=False,
            )

            plain_txt_path = audio_path.parent / f"{audio_path.stem}{clip_suffix}.txt"
            plain_text = (transcript_result.get("text") or "").strip()
            plain_txt_path.write_text(plain_text + ("\n" if plain_text else ""), encoding="utf-8")

            if self.enable_diarization_var.get():
                self.log(f"Diarization: {audio_path}")
                waveform = torch.from_numpy(audio).unsqueeze(0)
                diar_pipeline = self.get_diarization_pipeline()
                diar_result = diar_pipeline({"waveform": waveform, "sample_rate": 16000})
                diarization = extract_annotation(diar_result)
                speaker_segments = build_speaker_segments(diarization)

                merged_path = audio_path.with_suffix(audio_path.suffix + f"{clip_suffix}.merged.txt")
                with merged_path.open("w", encoding="utf-8") as f:
                    for seg in transcript_result.get("segments", []):
                        start = float(seg["start"]) + offset
                        end = float(seg["end"]) + offset
                        text = seg.get("text", "").strip()
                        if not text:
                            continue
                        speaker = pick_speaker(start - offset, end - offset, speaker_segments)
                        line = f"[{start:08.2f} - {end:08.2f}] [{speaker}] {text}"
                        f.write(line + "\n")

            return True, "OK"
        except Exception as exc:
            return False, f"Wyjątek uruchomienia: {exc}"


if __name__ == "__main__":
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    app = WhisperBatchApp(root)
    root.mainloop()
