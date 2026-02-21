# Transcriber

Transcribe video and audio files to text using AI speech recognition. Runs entirely on your computer — no audio is sent anywhere.

<img width="718" height="648" alt="image" src="https://github.com/user-attachments/assets/643e5641-33d7-4fce-b194-ad0647f895d8" />


## Download

**[Download Transcriber.zip from Releases](../../releases/latest)**
~100 MB — Windows 10/11, no install required.

1. Download and unzip the folder
2. Double-click `Transcriber.exe`
3. If Windows shows "Windows protected your PC" — click **More info** → **Run anyway**

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

## Tips

- A 1-hour file takes roughly 5–15 minutes with "base.en" on a modern computer
- Start with **base.en** — only move up if accuracy isn't good enough
- You can transcribe while the file plays in another app
- Everything runs locally on your CPU — nothing is uploaded

## For Developers

If you want to run from source instead of the exe:

```bash
pip install -r requirements.txt
python transcriber.py
```

Requires Python 3.9+.

## Troubleshooting

- **App won't start:** Make sure you extracted the entire zip, not just the `.exe`. All files must stay together.
- **Model download hangs:** Check your internet connection. After the first download, no internet is needed.
- **Poor accuracy:** Try a larger model (`small.en` or `medium.en`).
