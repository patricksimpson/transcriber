# Transcriber

Transcribe video and audio files to text using AI speech recognition. Runs entirely on your computer — no audio is sent anywhere.

<img width="718" height="648" alt="image" src="https://github.com/user-attachments/assets/643e5641-33d7-4fce-b194-ad0647f895d8" />

## Download

| Platform | Link | Size |
|----------|------|------|
| Windows (x64, GPU) | [Transcriber-windows-x64.zip](../../releases/latest/download/Transcriber-windows-x64.zip) | ~1.2 GB |
| Windows (x64, CPU-only) | [Transcriber-windows-x64-light.zip](../../releases/latest/download/Transcriber-windows-x64-light.zip) | ~30 MB |
| macOS (Apple Silicon) | [Transcriber-macos-arm64.zip](../../releases/latest/download/Transcriber-macos-arm64.zip) | ~127 MB |
| Linux (x64) | [Transcriber-linux-x64.zip](../../releases/latest/download/Transcriber-linux-x64.zip) | ~138 MB |

> **Note:** File sizes are approximate. The Windows GPU build is larger because it includes NVIDIA CUDA libraries.

No install required — download, unzip, and run.

### Windows

1. Download and unzip the folder
2. Double-click `Transcriber.exe`
3. If Windows shows "Windows protected your PC" — click **More info** → **Run anyway**

### macOS

1. Download and unzip
2. Double-click `Transcriber`
3. If macOS blocks it: System Settings → Privacy & Security → click **Open Anyway**

### Linux

1. Download and unzip
2. `chmod +x Transcriber && ./Transcriber`

> **First run:** The app downloads a speech recognition model (~150 MB). This only happens once — after that it works offline.

## How to Use

1. **Browse** for a video or audio file (MP4, MP3, WAV, M4A, MKV, etc.)
2. The file duration loads automatically — click **Open in player** to preview it
3. Optionally set a **Start time** and **End time** (`mm:ss` format, e.g. `35:00`, or just minutes like `35`)
4. Choose an **Output directory** (defaults to the file's folder)
5. Pick a **Model**:

   | Model | Speed | Accuracy |
   |-------|-------|----------|
   | tiny.en | Fastest | Lower |
   | **base.en** | **Fast** | **Good (recommended)** |
   | small.en | Medium | Better |
   | medium.en | Slow | Best |

6. Pick an **Output format**:
   - **Timestamped** — each line has `[HH:MM:SS -> HH:MM:SS]` timestamps
   - **Plain text** — just the spoken words
   - **Both** — saves two separate files
7. Click **Transcribe** — text appears line by line as it works

## Output Files

Files are saved to your chosen output directory:

```
sermon.txt
sermon (timestamped).txt
sermon (00.35.00-01.16.56).txt                    # with time range
sermon (00.35.00-01.16.56) (timestamped).txt
```

## Run from Source

Requires Python 3.10+.

```bash
git clone https://github.com/patricksimpson/transcriber.git
cd transcriber
pip install .
transcriber
```

Or without installing:

```bash
pip install -r requirements.txt
python transcriber.py
```

### GPU Acceleration

On Windows with an NVIDIA GPU, install with GPU support for significantly faster transcription:

```bash
pip install ".[gpu]"
```

Or: `pip install -r requirements-gpu.txt`

The app will automatically detect and use your GPU when available. GPU mode uses `float16` precision for best performance.

## Customizing Style (`sermon-style.json`)

The app automatically capitalizes proper nouns (names, places, scripture references) in transcription output. These rules are loaded from a `sermon-style.json` file.

**Where to put it:** Place `sermon-style.json` next to the executable (or script), or in your current working directory. The app checks both locations and uses the first one found.

**What's inside:** The file contains capitalization rules organized by category, plus a paragraph gap setting:

```json
{
  "description": "Scripture and sermon transcription style guide",
  "paragraph_gap": 1.5,
  "proper_nouns": {
    "deity_titles": [["holy spirit", "Holy Spirit"], ["god", "God"], ...],
    "biblical_persons": [["moses", "Moses"], ["abraham", "Abraham"], ...],
    "biblical_places": [["jerusalem", "Jerusalem"], ...],
    "scripture_references": [["old testament", "Old Testament"], ...],
    "religious_terms": [["baptism", "Baptism"], ...]
  }
}
```

Each rule is a pair: `["lowercase pattern", "Correct Capitalization"]`. Multi-word patterns are automatically matched before shorter ones (longest first), so "Holy Spirit" matches before "Spirit" alone.

- **`paragraph_gap`** — seconds of silence between segments that triggers a paragraph break (default: 1.5)
- **Categories** — organize however you like; all categories are flattened into one list at load time

To customize: edit `sermon-style.json` to add, remove, or change any rules. If the file is missing, built-in defaults are used. The status bar shows which style file was loaded on startup.

## Tips

- A 1-hour file takes roughly 5–15 minutes with "base.en" on a modern computer
- Start with **base.en** — only move up if accuracy isn't good enough
- You can transcribe while the file plays in another app
- Everything runs locally on your CPU (or GPU if available) — nothing is uploaded

## Troubleshooting

- **App won't start:** Make sure you extracted the entire zip, not just the executable.
- **Model download hangs:** Check your internet connection. After the first download, no internet is needed.
- **Poor accuracy:** Try a larger model (`small.en` or `medium.en`).
- **macOS "damaged" warning:** Run `xattr -cr Transcriber` in Terminal, then try again.
