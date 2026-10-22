"""Continuous avatar streaming session.

Instead of a fixed wav, this keeps a rolling 8 s audio window and decodes a new
1 s video chunk every time 15360 new 16 kHz samples arrive (24 frames at 25 fps).
Audio can be appended at any time (``feed_audio``), padded with silence to flush
the final partial chunk (``flush_audio``) or dropped for barge-in
(``clear_audio``). Decoded chunks are queued for the UI, which presents one
frame per tick, so decode speed never causes frame drops.
"""
import ctypes
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import logging

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_CACHE_ROOT = _ROOT / ".cache"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_CACHE_ROOT / "torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_CACHE_ROOT / "triton"))
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

import flash_head.inference as _inference
from flash_head.inference import (
    get_audio_embedding,
    get_base_data,
    get_infer_params,
    get_pipeline,
    run_pipeline,
)

AUDIO = "audio"
FLUSH = "flush"
CLEAR = "clear"
STOP = "stop"


def _timer_resolution(begin):
    try:
        fn = ctypes.windll.winmm.timeBeginPeriod if begin else ctypes.windll.winmm.timeEndPeriod
        fn(1)
    except Exception:
        pass


class AvatarSession:
    def __init__(
        self,
        ckpt_dir,
        wav2vec_dir,
        cond_image,
        size=384,
        base_seed=42,
        compile_model=True,
        compile_vae_decode=True,
        compile_vae_encode=True,
    ):
        self.cfg = {
            "ckpt_dir": ckpt_dir,
            "wav2vec_dir": wav2vec_dir,
            "cond_image": cond_image,
            "size": size,
            "base_seed": base_seed,
            "compile_model": compile_model,
            "compile_vae_decode": compile_vae_decode,
            "compile_vae_encode": compile_vae_encode,
        }
        self.cmd_q = queue.Queue()
        self.frame_q = queue.Queue(maxsize=2)
        self.ready = threading.Event()
        self.failed = None
        self.info = {}
        self._stop = threading.Event()
        self._thread = None
        self._gen = 0
        self._gen_lock = threading.Lock()
        self._chunks = 0
        self._chunk_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._paused = False
        self._hold_until = 0.0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def generation(self):
        with self._gen_lock:
            return self._gen

    @property
    def chunks_produced(self):
        with self._chunk_lock:
            return self._chunks

    def feed_audio(self, audio):
        self.cmd_q.put(
            (AUDIO, (self.generation, np.asarray(audio, dtype=np.float32)))
        )

    def flush_audio(self):
        self.cmd_q.put((FLUSH, None))

    def clear_audio(self):
        with self._gen_lock:
            self._gen += 1
        with self._frame_lock:
            self._paused = False
            self._hold_until = 0.0
        self.cmd_q.put((CLEAR, None))
        while True:
            try:
                self.frame_q.get_nowait()
            except queue.Empty:
                break

    def pause_frames(self):
        with self._frame_lock:
            self._paused = True

    def resume_after(self, seconds):
        with self._frame_lock:
            self._hold_until = time.perf_counter() + max(0.0, seconds)
            self._paused = False

    def next_frames(self):
        with self._frame_lock:
            if self._paused or time.perf_counter() < self._hold_until:
                return None
        current = self.generation
        while True:
            try:
                gen, frames = self.frame_q.get_nowait()
            except queue.Empty:
                return None
            if gen == current:
                return frames

    def stop(self):
        self._stop.set()
        self.cmd_q.put((STOP, None))
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _pad(self, pending):
        slice_samples = self.info["slice"]
        remainder = len(pending) % slice_samples
        if remainder == 0:
            return pending
        pad = slice_samples - remainder
        return np.concatenate([pending, np.zeros(pad, dtype=np.float32)])

    def _decode(self, pipeline, ring, audio_start_idx, audio_end_idx, motion):
        audio_array = np.asarray(ring, dtype=np.float32)
        embedding = get_audio_embedding(
            pipeline, audio_array, audio_start_idx, audio_end_idx
        )
        video = run_pipeline(pipeline, embedding)
        return video[motion:].cpu().numpy().astype(np.uint8)

    def _run(self):
        from flash_head.src.pipeline import flash_head_pipeline as _pipeline

        try:
            _pipeline.COMPILE_MODEL = self.cfg["compile_model"]
            _pipeline.COMPILE_VAE_DECODE = self.cfg["compile_vae_decode"]
            _pipeline.COMPILE_VAE_ENCODE = self.cfg["compile_vae_encode"]
            _inference.infer_params["height"] = self.cfg["size"]
            _inference.infer_params["width"] = self.cfg["size"]

            _timer_resolution(True)
            logger.info("Loading avatar model...")
            pipeline = get_pipeline(
                ckpt_dir=self.cfg["ckpt_dir"], wav2vec_dir=self.cfg["wav2vec_dir"]
            )
            get_base_data(
                pipeline,
                cond_image_path_or_dir=self.cfg["cond_image"],
                base_seed=self.cfg["base_seed"],
                use_face_crop=False,
            )
            params = get_infer_params()
            sample_rate = params["sample_rate"]
            tgt_fps = params["tgt_fps"]
            cached_duration = params["cached_audio_duration"]
            frame_num = params["frame_num"]
            motion = params["motion_frames_num"]
            slice_samples = (frame_num - motion) * sample_rate // tgt_fps
            cached_len = sample_rate * cached_duration
            audio_end_idx = cached_duration * tgt_fps
            audio_start_idx = audio_end_idx - frame_num
            self.info = {
                "sr": sample_rate,
                "fps": tgt_fps,
                "slice": slice_samples,
                "cached_len": cached_len,
            }
            ring = deque([0.0] * cached_len, maxlen=cached_len)
            pending = np.zeros(0, dtype=np.float32)
            pending_gen = None

            logger.info("Warming up avatar decode...")
            t_warm = time.perf_counter()
            ring.extend([0.0] * slice_samples)
            self._decode(pipeline, ring, audio_start_idx, audio_end_idx, motion)
            logger.info(f"Avatar warmup complete in {time.perf_counter() - t_warm:.1f}s.")
            logger.info(
                f"Avatar ready at {self.cfg['size']}px ({slice_samples} samples/chunk, "
                f"{tgt_fps} fps)."
            )
            self.ready.set()

            while not self._stop.is_set():
                try:
                    op, payload = self.cmd_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if op == STOP:
                    break
                if op == CLEAR:
                    pending = np.zeros(0, dtype=np.float32)
                    pending_gen = None
                    flush = False
                    while True:
                        try:
                            next_op, _ = self.cmd_q.get_nowait()
                        except queue.Empty:
                            break
                        if next_op == STOP:
                            self._stop.set()
                        elif next_op == FLUSH:
                            flush = True
                    if flush and not self._stop.is_set():
                        pending = self._pad(pending)
                elif op == FLUSH:
                    pending = self._pad(pending)
                elif op == AUDIO:
                    gen, chunk = payload
                    if len(pending) == 0:
                        pending_gen = gen
                    elif gen != pending_gen:
                        pending = np.zeros(0, dtype=np.float32)
                        pending_gen = gen
                    pending = np.concatenate([pending, chunk])

                while len(pending) >= slice_samples and not self._stop.is_set():
                    gen = pending_gen
                    piece = pending[:slice_samples]
                    pending = pending[slice_samples:]
                    ring.extend(piece.tolist())
                    frames = self._decode(
                        pipeline, ring, audio_start_idx, audio_end_idx, motion
                    )
                    self.frame_q.put((gen, frames))
                    with self._chunk_lock:
                        if gen == self.generation:
                            self._chunks += 1
        except Exception as exc:
            logger.exception("Avatar session failed")
            self.failed = str(exc)
        finally:
            _timer_resolution(False)
            self.ready.set()
