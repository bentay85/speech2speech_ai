"""Voice conversation loop: Silero VAD -> Gemma 4 (audio in, streaming out)
-> Kokoro fragments -> avatar.

Three background threads:
  * VAD thread drains the microphone queue, detects utterances and pushes them
    to the pipeline queue. Speech detected while the assistant is thinking or
    speaking triggers barge-in.
  * Pipeline thread streams the LLM reply, splits it into speakable fragments
    as the tokens arrive, and starts audio + video as soon as the avatar has
    decoded its first chunk (both streams then run unbuffered; the avatar
    decodes faster than realtime so neither starves).
  * Synth thread synthesizes queued fragments (Kokoro) and feeds the avatar
    while the LLM is still generating.

Gemma consumes the audio directly, so the "STT" and "LLM" stages are one call.
"""
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)
from PySide6.QtCore import QObject, Signal
from silero_vad import VADIterator
from silero_vad.utils_vad import OnnxWrapper

import llm as llm_client
from audio_io import MIC_BLOCK, MIC_SR
from tts import TTS, to_16k

MAX_UTTERANCE_S = 20.0
PREROLL_CHUNKS = 10
SILENCE_PEAK = 0.02
SPEAK_TIMEOUT_S = 300.0


class Fragmenter:
    """Splits streamed reply text for Kokoro prosody: the first fragment is the
    first complete sentence (a full prosodic unit for TTS to start on), the
    remaining text is sent as one final fragment once the LLM is done. Kokoro
    yields audio sentence-by-sentence internally for that long fragment, so
    speech still streams while only prosody grouping changes."""

    _HARD = ".!?"

    def __init__(self):
        self._buf = ""
        self._first = True

    @staticmethod
    def _break_at(buf, seps, minimum):
        for i, ch in enumerate(buf):
            if ch in seps and i + 1 >= minimum:
                head, tail = buf[: i + 1], buf[i + 1:]
                if tail and tail[0] == " ":
                    tail = tail[1:]
                return head, tail
        return None

    def feed(self, text):
        self._buf = (self._buf + text).lstrip()
        if not self._first:
            return []
        br = self._break_at(self._buf, self._HARD, 10)
        if br is None and len(self._buf) >= 160:
            br = (self._buf[:160], self._buf[160:])
        if br is None:
            return []
        head, self._buf = br
        head = head.strip()
        if not head:
            return []
        self._first = False
        return [head]

    def flush(self):
        head, self._buf = self._buf.strip(), ""
        return [head] if head else []


class VoiceLoop(QObject):
    level = Signal(float)
    state_changed = Signal(str)
    user_text = Signal(str)
    reply_text = Signal(str)
    interrupted = Signal()
    failed = Signal(str)

    def __init__(
        self,
        engine,
        avatar,
        server_url,
        voice="af_heart",
        tts=None,
        vad_model=None,
        input_name=None,
        av_delay=0.0,
        capture=None,
        system_prompt=None,
        sampling=None,
        vad_opts=None,
        pending=None,
        sent=None,
        image_megapixels=0.35,
    ):
        super().__init__()
        self.engine = engine
        self.avatar = avatar
        self.capture = capture
        self.server_url = server_url
        self.input_name = input_name
        self.av_delay = av_delay
        self.system_prompt = system_prompt
        self.sampling = sampling if sampling is not None else {}
        self.vad_opts = vad_opts if vad_opts is not None else {}
        self.image_megapixels = float(image_megapixels or 0.0)
        self._pending = pending if pending is not None else {"image": None}
        self._sent = sent if sent is not None else {"image": None}
        self.tts = tts if tts is not None else TTS(voice=voice)
        self.history = []
        self.state = "idle"
        self._turn = 0
        self._utt_q = queue.Queue()
        self._frag_q = queue.Queue()
        self._audio_q = queue.Queue()
        self._synths = {}
        self._synth_cv = threading.Condition()
        self._stop = threading.Event()
        self._model = vad_model
        self._threads = []

    def start(self):
        if not llm_client.health(self.server_url):
            raise RuntimeError(
                f"llama-server not reachable at {self.server_url}."
            )
        self.tts.warmup()
        if self._model is None:
            model_path = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "silero-vad"
                / "silero_vad.onnx"
            )
            self._model = OnnxWrapper(str(model_path), force_onnx_cpu=True)
        for target in (self._vad_loop, self._pipeline_loop, self._synth_worker):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)
        self._set_state("listening")

    def set_voice(self, voice):
        self.tts.voice = voice

    def stop(self):
        self._stop.set()
        with self._synth_cv:
            self._synth_cv.notify_all()

    def _take_pending(self):
        """Image dropped into the chat wins over the live capture, once."""
        image = self._pending.get("image")
        if image is not None:
            self._pending["image"] = None
            logger.info(
                f"Dropped image attached: {image.shape[1]}x{image.shape[0]} px"
            )
        return image

    def submit_text(self, text, image=None):
        """Type-to-assistant turn: skips VAD/ASR, reuses the rest of the flow."""
        if self.state == "idle":
            return False
        if self.state in ("thinking", "speaking"):
            self._barge_in()
        if image is None:
            image = self._take_pending()
        elif image is not None:
            logger.info(
                f"Typed image attached: {image.shape[1]}x{image.shape[0]} px"
            )
        self._utt_q.put((None, image, text or None))
        self._set_state("thinking")
        return True

    def _set_state(self, state):
        if state != self.state:
            self.state = state
            self.state_changed.emit(state)

    def _barge_in(self):
        self._turn += 1
        self._drain(self._frag_q)
        self._drain(self._audio_q)
        for key in [k for k in self._synths if k != self._turn]:
            self._synths.pop(key, None)
        self.engine.flush_output()
        self.avatar.clear_audio()
        self.interrupted.emit()
        logger.info("Barge-in: assistant interrupted")

    def _vad_loop(self):
        vad = VADIterator(
            self._model,
            threshold=float(self.vad_opts.get("threshold", 0.5)),
            sampling_rate=MIC_SR,
            min_silence_duration_ms=int(self.vad_opts.get("min_silence_ms", 500)),
            speech_pad_ms=int(self.vad_opts.get("speech_pad_ms", 100)),
        )
        preroll = []
        utterance = []
        in_speech = False
        try:
            while not self._stop.is_set():
                try:
                    chunk = self.engine.mic_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                self.level.emit(min(1.0, float(np.sqrt(np.mean(chunk ** 2))) * 8.0))

                if not in_speech:
                    preroll.append(chunk)
                    preroll = preroll[-PREROLL_CHUNKS:]

                event = vad(torch.from_numpy(chunk))

                if event and "start" in event:
                    preroll_peak = (
                        max(float(np.max(np.abs(item))) for item in preroll)
                        if preroll
                        else 0.0
                    )
                    if preroll_peak < SILENCE_PEAK:
                        vad.reset_states()
                        preroll = []
                        continue
                    if self.state in ("thinking", "speaking"):
                        self._barge_in()
                    in_speech = True
                    utterance = list(preroll)
                    self._set_state("listening")
                elif in_speech and event and "end" in event:
                    self._finalize(utterance)
                    utterance = []
                    in_speech = False
                    preroll = []
                    vad.reset_states()
                elif in_speech:
                    utterance.append(chunk)

                if in_speech and len(utterance) * MIC_BLOCK / MIC_SR > MAX_UTTERANCE_S:
                    self._finalize(utterance)
                    utterance = []
                    in_speech = False
                    preroll = []
                    vad.reset_states()
        except Exception as exc:
            logger.exception("VAD loop failed")
            self.failed.emit(str(exc))

    def _finalize(self, utterance):
        if not utterance:
            return
        audio = np.concatenate(utterance).astype(np.float32)
        peak = float(np.max(np.abs(audio)))
        duration = len(audio) / MIC_SR
        if duration < 0.3 or peak < SILENCE_PEAK:
            source = f" ({self.input_name})" if self.input_name else ""
            logger.warning(
                f"Ignoring silent utterance ({duration:.1f}s, peak {peak:.3f}); "
                f"check the selected microphone{source} and its level."
            )
            self._set_state("listening")
            return
        image = self._take_pending()
        if image is None and self.capture is not None:
            image = self.capture.snapshot()
            if image is not None:
                logger.info(
                    f"Capture attached: {image.shape[1]}x{image.shape[0]} px"
                )
        logger.info(f"Utterance captured: {duration:.1f}s, peak {peak:.2f}")
        self._utt_q.put((audio, image, None))
        self._set_state("thinking")

    def _pipeline_loop(self):
        while not self._stop.is_set():
            try:
                audio, image, text = self._utt_q.get(timeout=0.1)
            except queue.Empty:
                continue
            turn = self._turn
            self._set_state("thinking")
            if image is not None:
                self._sent["image"] = image
            t0 = time.perf_counter()
            start_chunk = self.avatar.chunks_produced
            self._synths[turn] = {"queued": 0, "done": 0, "sent": False}
            self._drain(self._audio_q)
            self.avatar.pause_frames()
            fragmenter = Fragmenter()
            first_token = {"at": None, "frag": False, "streamed_reply": False}

            def emit_frag(frag):
                if not first_token["frag"]:
                    first_token["frag"] = True
                    logger.info(
                        f"turn {turn}: first fragment after "
                        f"{time.perf_counter() - t0:.2f}s: {frag[:40]!r}"
                    )
                self._queue_synth(turn, frag)

            def on_pieces(pieces, got):
                if turn != self._turn or self._stop.is_set():
                    return False
                if first_token["at"] is None and got["reply"]:
                    first_token["at"] = time.perf_counter()
                    logger.info(
                        f"turn {turn}: first reply token in "
                        f"{first_token['at'] - t0:.2f}s"
                    )
                for key, piece in pieces:
                    if key != "reply":
                        continue
                    first_token["streamed_reply"] = True
                    for frag in fragmenter.feed(piece):
                        emit_frag(frag)
                return None

            try:
                result = llm_client.ask(
                    self.server_url, audio, MIC_SR, self.history,
                    image=image, text=text, system_prompt=self.system_prompt,
                    image_max_px=(int(self.image_megapixels * 1e6)
                                  if self.image_megapixels > 0 else None),
                    stream_events=on_pieces,
                    **self.sampling,
                )
            except Exception as exc:
                logger.exception("LLM request failed")
                self._finish_synth(turn)
                self.failed.emit(str(exc))
                self._set_state("listening")
                continue
            if result is None or turn != self._turn:
                logger.info(
                    f"turn {turn}: superseded by a newer utterance "
                    "(barge-in or abort)."
                )
                self._finish_synth(turn)
                continue
            transcript, reply = result
            transcript, reply = transcript.strip(), reply.strip()
            if text is not None:
                transcript = text.strip() or "[image]"
            elif not transcript or transcript.lower().startswith(
                "answer the request in this audio"
            ):
                logger.warning(
                    f"turn {turn}: LLM returned no speech transcript; the audio "
                    "may not have reached the model."
                )
                transcript = ""
            logger.info(
                f"turn {turn}: LLM stream done in {time.perf_counter() - t0:.2f}s"
            )
            if not reply:
                self._finish_synth(turn)
                self._set_state("listening")
                continue
            self.user_text.emit(transcript)
            self.reply_text.emit(reply)
            if transcript:
                self.history.append({"transcript": transcript, "reply": reply})
            if not first_token["streamed_reply"] and reply:
                # The reply never streamed as JSON "reply" pieces (the model
                # ignored the JSON format and ask() fell back to raw text):
                # fragment the whole reply now so it still gets spoken.
                logger.warning(
                    f"turn {turn}: reply did not stream as JSON; speaking the "
                    f"fallback text instead: {reply[:60]!r}"
                )
                for frag in fragmenter.feed(reply):
                    emit_frag(frag)
            for frag in fragmenter.flush():
                emit_frag(frag)
            self._finish_synth(turn)
            self._speak_streaming(turn, t0, start_chunk)

    def _queue_synth(self, turn, fragment):
        state = self._synths.get(turn)
        if state is None:
            return
        state["queued"] += 1
        self._frag_q.put((turn, fragment))

    def _finish_synth(self, turn):
        self._frag_q.put((turn, None))

    def _drain(self, q):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    def _synth_worker(self):
        while not self._stop.is_set():
            try:
                turn, frag = self._frag_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if turn != self._turn:
                self._synths.pop(turn, None)
                continue
            if frag is None:
                if turn == self._turn:
                    self.avatar.flush_audio()
                    self._audio_q.put((turn, None))
                with self._synth_cv:
                    state = self._synths.get(turn)
                    if state is not None:
                        state["sent"] = True
                    self._synth_cv.notify_all()
                continue
            try:
                t_synth = time.perf_counter()
                seconds = 0.0
                for audio24k in self.tts.stream_raw(frag):
                    if turn != self._turn or self._stop.is_set():
                        break
                    audio24k = np.asarray(audio24k, dtype=np.float32)
                    seconds += len(audio24k) / 24000
                    self.avatar.feed_audio(to_16k(audio24k))
                    self._audio_q.put((turn, audio24k))
                logger.debug(
                    f"turn {turn}: synthesized {frag[:30]!r} in "
                    f"{time.perf_counter() - t_synth:.2f}s "
                    f"({seconds:.1f}s audio)"
                )
            except Exception:
                logger.exception(f"turn {turn}: TTS fragment failed: {frag[:40]!r}")
            with self._synth_cv:
                state = self._synths.get(turn)
                if state is not None:
                    state["done"] += 1
                    self._synth_cv.notify_all()

    def _speak_streaming(self, turn, t0, start_chunk):
        # Both streams start on the decoded first chunk; after that nothing is
        # buffered: the avatar decodes faster than realtime, and the LLM/TTS
        # produce far faster than speech plays, so neither side starves.
        deadline = time.perf_counter() + 10.0
        while (
            not self._stop.is_set()
            and turn == self._turn
            and self.avatar.chunks_produced <= start_chunk
            and time.perf_counter() < deadline
        ):
            time.sleep(0.02)
        if self._stop.is_set() or turn != self._turn:
            return
        if self.avatar.chunks_produced <= start_chunk:
            self.avatar.clear_audio()
            if turn == self._turn:
                logger.warning(
                    f"turn {turn}: avatar produced no chunk within 10s; "
                    "no speech or video played (was any fragment synthesized?)."
                )
                self._set_state("listening")
            return
        logger.info(
            f"turn {turn}: A/V start {(time.perf_counter() - t0) * 1000:.0f} ms "
            "after end-of-utterance (avatar first chunk ready)"
        )
        self._set_state("speaking")
        self.avatar.resume_after(
            self.engine.out_latency + 0.03 + self.av_delay
        )
        finished = False
        deadline = time.perf_counter() + SPEAK_TIMEOUT_S
        while (
            not finished
            and not self._stop.is_set()
            and turn == self._turn
            and time.perf_counter() < deadline
        ):
            try:
                item_turn, piece = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item_turn != turn:
                continue
            if piece is None:
                finished = True
                continue
            self.engine.play(piece)
        while (
            turn == self._turn
            and not self._stop.is_set()
            and self.engine.queued_seconds() > 0.0
        ):
            time.sleep(0.05)
        if turn == self._turn and not self._stop.is_set():
            self._set_state("listening")
