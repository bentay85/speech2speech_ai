"""Process bootstrap: paths, logging, environment, warning filters, constants.

Import this module before anything heavy (numpy, torch, sounddevice, ...):
it truncates the log file, configures the standard ``logging`` root logger
and sets the cache directory env vars that the heavy stacks pick up at
their import time.

Log level comes from the ``LOG_LEVEL`` environment variable (default INFO);
set it to DEBUG for per-fragment/per-request diagnostics.
"""
import logging
import os
import sys
import threading
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_log_dir = ROOT / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
# Always truncate on startup: the file only ever holds the most recent run.
_log_stream = open(_log_dir / "app.log", "w", buffering=1, encoding="utf-8")
if sys.stdout is None:
    sys.stdout = _log_stream
if sys.stderr is None:
    sys.stderr = _log_stream

_CACHE_ROOT = ROOT / ".cache"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_CACHE_ROOT / "torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_CACHE_ROOT / "triton"))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*should be kept in float32.*")
warnings.filterwarnings("ignore", message=r"Some weights of.*")
warnings.filterwarnings("ignore", message=r".*dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=r".*weight_norm.*")

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
_root = logging.getLogger()
_root.setLevel(_level)
# StreamHandler over the line-buffered log file: records are flushed as they
# are emitted, so the tail of logs/app.log always shows live progress.
_file_handler = logging.StreamHandler(_log_stream)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
_root.addHandler(_file_handler)
if sys.stderr is not _log_stream:
    _console_handler = logging.StreamHandler(sys.stderr)
    _console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    _root.addHandler(_console_handler)
# Third-party chatter: keep only errors from the model stacks.
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.image_utils").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("silero_vad").setLevel(logging.WARNING)
logging.captureWarnings(True)

# The heavy model stacks (torch, kokoro, onnxruntime, flash_head) are imported
# lazily inside the loader threads so the window appears almost immediately.
# They must not import concurrently: transformers' lazy module machinery is
# not thread-safe and a concurrent import can fail with "cannot import name".
HEAVY_IMPORT_LOCK = threading.Lock()

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SETTINGS_FILE = ROOT / "settings.json"
NAME_TOKEN = "<Name>"
COMPONENTS = [
    ("llm", "LLM server (Gemma 4)"),
    ("avatar", "Avatar model"),
    ("vad", "Voice activity detector"),
    ("tts", "Text-to-speech (Kokoro)"),
]
COLORS = {
    "pending": "#9aa0a6",
    "loading": "#f5a623",
    "ready": "#2ecc71",
    "failed": "#e74c3c",
}
STATE_TEXT = {
    "idle": "Idle",
    "listening": "Listening",
    "thinking": "Thinking",
    "speaking": "Speaking",
}
AVATAR_SIZES = [256, 320, 384, 448, 512]
