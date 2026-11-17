"""Configuration: built-in defaults, deep-merged with an optional YAML/JSON file and environment variables."""
import copy
import json
import os

DEFAULTS = {
    "output_dir": "output",          # transcripts/, run logs, reports
    "state_dir": "state",            # claims/ (lock files), failed/ (tracebacks)
    "audio_dir": "",                 # if <audio_dir>/<id>.<ext> exists, it is used instead of downloading
    "tmp_dir": "",                   # temporary downloads (default: system temp)
    "language": "en",
    # Domain vocabulary passed to Whisper as initial prompt (names, places, jargon). Keep it short.
    "vocabulary": "",
    # Regex fixes applied to segment text before saving: [[pattern, replacement], ...]
    "replacements": [],
    # Extra regexes for phrases Whisper invents on music or silence (added to the built-in list).
    "extra_hallucinations": [],
    "repair": {
        "enabled": True,
        "gap_s": 6.0,          # silence between segments longer than this may hide lost speech
        "edge_gap_s": 20.0,    # at the start/end of a file (intros, music) only gaps longer than this
        "long_word_s": 4.0,    # a single word lasting longer than this is flagged for QA
        "include_edges": True,
    },
    "local": {
        "model": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "batch_size": 16,
        "beam_size": 5,
        "vad_filter": True,
        "cpu_threads": 2,
    },
    "groq": {
        "model": "whisper-large-v3",
        "api_url": "https://api.groq.com/openai/v1/audio/transcriptions",
        "bitrate": "32k",            # mono 16 kHz at 32 kb/s keeps ~1 h of audio under the 25 MB upload limit
        "max_wait_s": 3600,
    },
    "download": {
        "user_agent": "transcription-pipeline/1.0 (+https://github.com/LektorGW/transcription-pipeline)",
        "retries": 4,
        "timeout_s": 180,
    },
    "sync": {
        "remote": "",                # optional rclone remote, e.g. "myremote:archive"
        "every_s": 120,
    },
    "site": {
        "title": "Podcast Archive",
        "subtitle": "Every episode fully transcribed. Search any word and jump straight to that moment in the audio.",
        "footer": "",
        "meta_dir": "",              # optional per-episode metadata (summary, keywords, key moments)
    },
}


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None):
    """Return the effective configuration. `path` may be a .yaml/.yml (needs PyYAML) or .json file."""
    data = {}
    if path:
        with open(path, encoding="utf-8") as f:
            if path.endswith((".yaml", ".yml")):
                import yaml
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)
    cfg = deep_merge(DEFAULTS, data)
    if os.environ.get("TP_OUTPUT_DIR"):
        cfg["output_dir"] = os.environ["TP_OUTPUT_DIR"]
    if os.environ.get("TP_SYNC_REMOTE"):
        cfg["sync"]["remote"] = os.environ["TP_SYNC_REMOTE"]
    return cfg


def load_dotenv(path=".env"):
    """Minimal .env reader (KEY=VALUE lines); existing environment variables win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
