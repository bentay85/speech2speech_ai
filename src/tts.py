"""Kokoro text-to-speech on CUDA. Yields one float32 24 kHz array per sentence
so the avatar and speakers can start on the first sentence while later
sentences are still being synthesized."""
import logging
import time
from pathlib import Path

import numpy as np
from kokoro import KModel, KPipeline

logger = logging.getLogger(__name__)

KOKORO_DIR = Path(__file__).resolve().parent.parent / "models" / "Kokoro-82M"


def to_16k(audio):
    from scipy.signal import resample_poly

    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)
    return resample_poly(np.asarray(audio, dtype=np.float32), 2, 3).astype(np.float32)


from voices import VOICE_LABELS, VOICE_META, VOICES, voice_name  # noqa: F401


class TTS:
    def __init__(self, voice="af_heart", device="cuda", speed=1.0):
        self.voice = voice
        self.device = device
        self.speed = speed
        self._model = None
        self._pipe = None
        self._lang = None

    def _pipeline(self):
        if self._model is None:
            config = KOKORO_DIR / "config.json"
            weights = KOKORO_DIR / "kokoro-v1_0.pth"
            if not config.is_file() or not weights.is_file():
                raise FileNotFoundError(
                    f"Kokoro weights missing in {KOKORO_DIR}; "
                    "run setup.bat or see the README section on models."
                )
            logger.info("Loading Kokoro TTS weights on %s...", self.device)
            t_load = time.perf_counter()
            self._model = KModel(
                repo_id="hexgrad/Kokoro-82M",
                config=str(config),
                model=str(weights),
            ).to(self.device).eval()
            logger.info("Kokoro weights loaded in %.1fs.", time.perf_counter() - t_load)
        lang = self.voice[0]
        if self._pipe is None or self._lang != lang:
            logger.info("Building Kokoro pipeline for lang_code=%r.", lang)
            self._pipe = KPipeline(
                lang_code=lang,
                repo_id="hexgrad/Kokoro-82M",
                model=self._model,
                device=self.device,
            )
            self._lang = lang
        return self._pipe

    def _voice_path(self):
        path = KOKORO_DIR / "voices" / f"{self.voice}.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Kokoro voice '{self.voice}' missing at {path}; "
                "run setup.bat or re-download the voices/ folder from "
                "hexgrad/Kokoro-82M."
            )
        return str(path)

    def warmup(self):
        t_warm = time.perf_counter()
        for _ in self.stream("Hello."):
            pass
        logger.info("TTS warmup complete in %.1fs.", time.perf_counter() - t_warm)

    def stream(self, text):
        text = (text or "").strip()
        if not text:
            return
        yield from self.stream_raw(text)

    def stream_raw(self, text):
        voice_path = self._voice_path()
        logger.debug("TTS synthesis: voice=%s text=%r", self.voice, text[:60])
        for _, _, audio in self._pipeline()(
            text, voice=voice_path, speed=self.speed
        ):
            if audio is None or len(audio) == 0:
                continue
            yield np.asarray(audio, dtype=np.float32)
