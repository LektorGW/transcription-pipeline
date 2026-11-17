"""Audio I/O through FFmpeg: decoding, compression for API upload, clipping, resilient downloads."""
import os
import subprocess
import time

SR = 16000
AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".mp4", ".webm", ".mkv")


def load_audio(path, limit_s=0, sr=SR):
    """Decode any FFmpeg-readable file to mono float32 PCM at 16 kHz (what Whisper expects)."""
    import numpy as np
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path]
    if limit_s:
        cmd += ["-t", str(limit_s)]
    cmd += ["-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def compress(src, dst, bitrate="32k", start=None, end=None):
    """Mono 16 kHz MP3, optionally only [start, end). One hour at 32 kb/s is ~14 MB."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.2f}"]
    if end is not None:
        cmd += ["-to", f"{end:.2f}"]
    cmd += ["-i", src, "-ac", "1", "-ar", str(SR), "-b:a", bitrate, dst]
    subprocess.run(cmd, check=True)
    return dst


def download(url, dst, user_agent="transcription-pipeline/1.0", retries=4, timeout=180):
    """Stream to `<dst>.part`, rename when complete; retry with back-off on flaky connections."""
    import requests
    last = None
    for attempt in range(retries):
        try:
            with requests.get(url, headers={"User-Agent": user_agent}, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(dst + ".part", "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(dst + ".part", dst)
            return dst
        except Exception as ex:
            last = ex
            time.sleep(5 * (attempt + 1))
    raise last


def local_copy(ep, audio_dir):
    """Path of an already downloaded file for this episode, if any."""
    if ep.get("audio") and not str(ep["audio"]).startswith(("http://", "https://")) and os.path.exists(ep["audio"]):
        return ep["audio"]
    if audio_dir:
        for ext in AUDIO_EXTS:
            p = os.path.join(audio_dir, ep["id"] + ext)
            if os.path.exists(p):
                return p
    return None


def fetch(ep, tmp_dir, audio_dir="", download_cfg=None):
    """Return (path, is_temporary) for the episode audio, downloading it when needed."""
    p = local_copy(ep, audio_dir)
    if p:
        return p, False
    url = ep.get("audio", "")
    if not url.startswith(("http://", "https://")):
        raise FileNotFoundError(f'audio not found for {ep["id"]}: {url!r}')
    ext = os.path.splitext(url.split("?")[0])[1].lower()
    dst = os.path.join(tmp_dir, ep["id"] + (ext if ext in AUDIO_EXTS else ".mp3"))
    d = download_cfg or {}
    return download(url, dst, d.get("user_agent", "transcription-pipeline/1.0"), d.get("retries", 4),
                    d.get("timeout_s", 180)), True
