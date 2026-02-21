"""Video/Audio Transcriber GUI — transcribe media files using faster-whisper."""

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

# Add NVIDIA CUDA DLL directories to search path (for GPU support)
for _nvidia_pkg in ("nvidia.cublas", "nvidia.cudnn"):
    try:
        _pkg = __import__(_nvidia_pkg, fromlist=[""])
        _bin = os.path.join(_pkg.__spec__.submodule_search_locations[0], "bin")
        if os.path.isdir(_bin):
            os.add_dll_directory(_bin)
            os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


class TranscriberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Transcriber")
        self.root.geometry("720x620")
        self.root.minsize(650, 540)

        self.msg_queue = queue.Queue()
        self.transcribing = False
        self._stop_requested = False
        self.cuda_available = self._detect_cuda()
        self._file_duration = None

        # Remember last-used directories
        self._last_input_dir = None
        self._last_output_dir = None

        self._build_ui()
        self._poll_queue()

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # File picker
        file_frame = ttk.Frame(self.root)
        file_frame.pack(fill="x", **pad)
        ttk.Label(file_frame, text="File:").pack(side="left")
        self.file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var).pack(
            side="left", fill="x", expand=True, padx=(4, 4)
        )
        ttk.Button(file_frame, text="Browse", command=self._browse_file).pack(side="left")

        # File info row: Load, duration display, Open in player
        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill="x", **pad)
        self.load_btn = ttk.Button(info_frame, text="Load info", command=self._load_file_info)
        self.load_btn.pack(side="left")
        self.duration_var = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self.duration_var, font=("Consolas", 10)).pack(
            side="left", padx=(8, 0)
        )
        self.open_btn = ttk.Button(
            info_frame, text="Open in player", command=self._open_in_player, state="disabled"
        )
        self.open_btn.pack(side="right")

        # Output directory picker
        outdir_frame = ttk.Frame(self.root)
        outdir_frame.pack(fill="x", **pad)
        ttk.Label(outdir_frame, text="Output dir:").pack(side="left")
        self.outdir_var = tk.StringVar()
        ttk.Entry(outdir_frame, textvariable=self.outdir_var).pack(
            side="left", fill="x", expand=True, padx=(4, 4)
        )
        ttk.Button(outdir_frame, text="Browse", command=self._browse_outdir).pack(side="left")

        # Time range row
        time_frame = ttk.Frame(self.root)
        time_frame.pack(fill="x", **pad)
        ttk.Label(time_frame, text="Start time:").pack(side="left")
        self.start_var = tk.StringVar()
        ttk.Entry(time_frame, textvariable=self.start_var, width=8).pack(
            side="left", padx=(4, 4)
        )
        ttk.Label(time_frame, text="End time:").pack(side="left", padx=(8, 0))
        self.end_var = tk.StringVar()
        ttk.Entry(time_frame, textvariable=self.end_var, width=8).pack(
            side="left", padx=(4, 4)
        )
        ttk.Label(time_frame, text="(mm:ss or m — empty = full file)").pack(
            side="left", padx=(8, 0)
        )

        # Model row (+ Device if GPU available)
        model_frame = ttk.Frame(self.root)
        model_frame.pack(fill="x", **pad)
        ttk.Label(model_frame, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value="base.en")
        ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=["tiny.en", "base.en", "small.en", "medium.en"],
            state="readonly",
            width=12,
        ).pack(side="left", padx=(4, 0))

        if self.cuda_available:
            ttk.Label(model_frame, text="Device:").pack(side="left", padx=(16, 0))
            self.device_var = tk.StringVar(value="Auto (GPU)")
            ttk.Combobox(
                model_frame,
                textvariable=self.device_var,
                values=["Auto (GPU)", "GPU", "CPU"],
                state="readonly",
                width=12,
            ).pack(side="left", padx=(4, 0))
        else:
            self.device_var = tk.StringVar(value="CPU")

        # Output format
        fmt_frame = ttk.LabelFrame(self.root, text="Output format")
        fmt_frame.pack(fill="x", **pad)
        self.format_var = tk.StringVar(value="both")
        for text, val in [
            ("Timestamped", "timestamped"),
            ("Plain text", "plain"),
            ("Both", "both"),
        ]:
            ttk.Radiobutton(fmt_frame, text=text, variable=self.format_var, value=val).pack(
                side="left", padx=8, pady=4
            )

        # Button row
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", **pad)
        self.transcribe_btn = ttk.Button(
            btn_frame, text="Transcribe", command=self._start_transcription
        )
        self.transcribe_btn.pack(side="left", padx=(0, 4))
        self.stop_btn = ttk.Button(
            btn_frame, text="Stop (save progress)", command=self._stop_transcription,
            state="disabled"
        )
        self.stop_btn.pack(side="left")

        # Progress bar + ETA label
        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill="x", **pad)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100, length=400
        )
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.eta_var = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.eta_var, font=("Consolas", 9)).pack(
            side="left"
        )

        # Log area
        log_frame = ttk.Frame(self.root)
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(log_frame, wrap="word", state="disabled", font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(
            fill="x", side="bottom"
        )

    # ── Device detection ────────────────────────────────────────────

    @staticmethod
    def _detect_cuda():
        try:
            import nvidia.cublas  # noqa: F401 — only present in GPU builds
            import nvidia.cudnn  # noqa: F401
            return True
        except Exception:
            return False

    # ── Helpers ───────────────────────────────────────────────────────

    def _browse_file(self):
        initial = self._last_input_dir
        if not initial:
            current = self.file_var.get().strip()
            if current and os.path.isfile(current):
                initial = str(Path(current).parent)

        path = filedialog.askopenfilename(
            initialdir=initial,
            filetypes=[
                ("Media files", "*.mp4 *.mp3 *.wav *.m4a *.mkv *.webm *.ogg *.flac *.aac *.wma"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.file_var.set(path)
            self._last_input_dir = str(Path(path).parent)
            if not self.outdir_var.get().strip():
                self.outdir_var.set(self._last_input_dir)
            self._load_file_info()

    def _load_file_info(self):
        filepath = self.file_var.get().strip()
        if not filepath or not os.path.isfile(filepath):
            self.duration_var.set("No file selected")
            return
        self.duration_var.set("Loading...")
        self.load_btn.configure(state="disabled")

        def _worker():
            duration = self._get_duration(filepath)
            def _update():
                self.load_btn.configure(state="normal")
                self.open_btn.configure(state="normal")
                if duration is not None:
                    self._file_duration = duration
                    self.duration_var.set(
                        f"Duration: {self._format_time(duration)}  "
                        f"({duration / 60:.1f} min)"
                    )
                else:
                    self._file_duration = None
                    self.duration_var.set("Could not read duration")
            self.root.after(0, _update)

        threading.Thread(target=_worker, daemon=True).start()

    def _open_in_player(self):
        filepath = self.file_var.get().strip()
        if not filepath or not os.path.isfile(filepath):
            return
        if os.name == "nt":
            os.startfile(filepath)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", filepath])
        else:
            subprocess.Popen(["xdg-open", filepath])

    def _browse_outdir(self):
        initial = self._last_output_dir or self.outdir_var.get().strip() or None
        path = filedialog.askdirectory(initialdir=initial)
        if path:
            self.outdir_var.set(path)
            self._last_output_dir = path

    @staticmethod
    def _parse_time(text):
        """Parse a time string into seconds. Accepts: mm:ss, m:ss, or just minutes (e.g. '5')."""
        text = text.strip()
        if not text:
            return None
        if ":" in text:
            parts = text.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid time format: '{text}' — use mm:ss")
            minutes, seconds = parts
            return int(minutes) * 60 + int(seconds)
        else:
            return float(text) * 60

    def _log(self, text):
        self.msg_queue.put(text)

    def _set_status(self, text):
        self.msg_queue.put(("__STATUS__", text))

    def _set_progress(self, pct, eta_text):
        self.msg_queue.put(("__PROGRESS__", pct, eta_text))

    def _poll_queue(self):
        while not self.msg_queue.empty():
            item = self.msg_queue.get_nowait()
            if isinstance(item, tuple) and item[0] == "__STATUS__":
                self.status_var.set(item[1])
            elif isinstance(item, tuple) and item[0] == "__PROGRESS__":
                self.progress_var.set(item[1])
                self.eta_var.set(item[2])
            else:
                self.log.configure(state="normal")
                self.log.insert("end", item)
                self.log.see("end")
                self.log.configure(state="disabled")
        self.root.after(100, self._poll_queue)

    # ── Transcription ─────────────────────────────────────────────────

    def _start_transcription(self):
        filepath = self.file_var.get().strip()
        if not filepath or not os.path.isfile(filepath):
            self._log("Error: Please select a valid file.\n")
            return
        if self.transcribing:
            return

        self.transcribing = True
        self._stop_requested = False
        self.transcribe_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_var.set(0)
        self.eta_var.set("")

        # Clear log
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

        thread = threading.Thread(target=self._transcribe_worker, args=(filepath,), daemon=True)
        thread.start()

    def _stop_transcription(self):
        if self.transcribing:
            self._stop_requested = True
            self.stop_btn.configure(state="disabled")
            self._set_status("Stopping... saving progress")

    def _transcribe_worker(self, filepath):
        try:
            self._run_transcription(filepath)
        except Exception as e:
            self._log(f"\nError: {e}\n")
            self._set_status("Error")
        finally:
            self.transcribing = False
            self._stop_requested = False
            self.root.after(0, lambda: self.transcribe_btn.configure(state="normal"))
            self.root.after(0, lambda: self.stop_btn.configure(state="disabled"))

    def _run_transcription(self, filepath):
        from faster_whisper import WhisperModel

        model_name = self.model_var.get()
        fmt = self.format_var.get()

        # Parse start/end times
        try:
            start_sec = self._parse_time(self.start_var.get())
            end_sec = self._parse_time(self.end_var.get())
        except ValueError as e:
            self._log(f"Error: {e}\n")
            return

        # Determine output directory
        outdir_text = self.outdir_var.get().strip()
        if outdir_text:
            outdir = Path(outdir_text)
            if not outdir.is_dir():
                self._log(f"Error: Output directory does not exist: {outdir}\n")
                return
        else:
            outdir = Path(filepath).parent

        # Resolve device
        device_choice = self.device_var.get()
        if device_choice == "GPU":
            device, compute_type = "cuda", "float16"
        elif device_choice == "CPU":
            device, compute_type = "cpu", "int8"
        else:  # Auto
            if self.cuda_available:
                device, compute_type = "cuda", "float16"
            else:
                device, compute_type = "cpu", "int8"

        self._set_status("Loading model...")
        self._log(f"Loading model '{model_name}' on {device.upper()}...\n")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self._log(f"Model loaded ({device.upper()}, {compute_type}).\n\n")

        # Build clip_timestamps from start/end
        clip_timestamps = None
        time_label = ""
        audio_start = 0
        audio_end = None

        if start_sec is not None or end_sec is not None:
            self._set_status("Reading file duration...")
            self._log("Reading file duration...\n")
            duration = self._get_duration(filepath)

            audio_start = start_sec if start_sec is not None else 0
            audio_end = end_sec if end_sec is not None else (duration or 0)

            if duration:
                self._log(f"Duration: {self._format_time(duration)}\n")
                if audio_end > duration:
                    audio_end = duration

            clip_timestamps = f"{audio_start},{audio_end}"
            self._log(
                f"Transcribing: {self._format_time(audio_start)} -> "
                f"{self._format_time(audio_end)}\n\n"
            )
            time_label = f" ({self._format_time_safe(audio_start)}-{self._format_time_safe(audio_end)})"
        else:
            duration = self._file_duration
            if duration is None:
                duration = self._get_duration(filepath)
            if duration:
                audio_end = duration

        total_audio = (audio_end - audio_start) if audio_end else None

        # Transcribe
        self._set_status("Transcribing...")
        self._set_progress(0, "Estimating...")
        self._log("Transcribing...\n" + "-" * 50 + "\n")

        kwargs = {"beam_size": 5, "language": "en"}
        if clip_timestamps:
            kwargs["clip_timestamps"] = clip_timestamps

        segments_iter, info = model.transcribe(filepath, **kwargs)

        timestamped_lines = []
        plain_lines = []
        wall_start = time.time()
        stopped = False

        for seg in segments_iter:
            if self._stop_requested:
                stopped = True
                break

            ts = self._format_time(seg.start)
            te = self._format_time(seg.end)
            text = seg.text.strip()

            timestamped_lines.append(f"[{ts} -> {te}]  {text}")
            plain_lines.append(text)
            self._log(f"[{ts} -> {te}]  {text}\n")

            # Update progress bar and ETA
            if total_audio and total_audio > 0:
                audio_done = seg.end - audio_start
                elapsed = time.time() - wall_start
                pct = min(100, audio_done / total_audio * 100)

                if audio_done > 0 and elapsed > 1:
                    speed = audio_done / elapsed
                    remaining_audio = total_audio - audio_done
                    eta_sec = max(0, remaining_audio / speed)
                    self._set_progress(
                        pct,
                        f"{pct:.0f}%  ~{self._format_time(eta_sec)} left"
                    )
                else:
                    self._set_progress(pct, f"{pct:.0f}%")

        elapsed_total = time.time() - wall_start

        if stopped:
            self._log("\n" + "-" * 50 + "\n")
            self._log("Stopped by user.\n")
        else:
            self._log("-" * 50 + "\n")
            self._set_progress(100, "Complete")

        # Save whatever we have
        if not plain_lines:
            self._log("Nothing to save.\n")
            self._set_status("Stopped — nothing saved")
            return

        src = Path(filepath)
        partial = " (partial)" if stopped else ""
        base_name = f"{src.stem}{time_label}{partial}"

        if fmt in ("timestamped", "both"):
            out_ts = outdir / f"{base_name} (timestamped).txt"
            out_ts.write_text("\n".join(timestamped_lines), encoding="utf-8")
            self._log(f"Saved: {out_ts}\n")

        if fmt in ("plain", "both"):
            out_plain = outdir / f"{base_name}.txt"
            out_plain.write_text("\n".join(plain_lines), encoding="utf-8")
            self._log(f"Saved: {out_plain}\n")

        total_segments = len(plain_lines)
        status_word = "Stopped" if stopped else "Done"
        self._log(
            f"\n{status_word} — {total_segments} segments transcribed "
            f"in {self._format_time(elapsed_total)}.\n"
        )
        self._set_status(status_word)

    def _get_duration(self, filepath):
        try:
            import av
            with av.open(filepath) as container:
                return float(container.duration) / 1_000_000
        except Exception:
            return None

    @staticmethod
    def _format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _format_time_safe(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}.{m:02d}.{s:02d}"


if __name__ == "__main__":
    root = tk.Tk()
    TranscriberApp(root)
    root.mainloop()
