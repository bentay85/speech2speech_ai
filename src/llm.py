"""Thin client for the local llama.cpp server running Gemma 4 with the audio
projector. One call returns both the spoken transcript and the reply, since
Gemma consumes the microphone audio directly (no separate STT model)."""
import base64
import io
import json
import logging
import re
import urllib.request
import wave

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant. You receive a short spoken "
    "request as audio. Reply with JSON only, no markdown, with exactly two "
    'keys in this order: "reply" (one or two short spoken sentences, ALWAYS '
    'FIRST so speech can start early) and "transcript" (the verbatim words '
    "you heard)."
)

# Defaults applied to every request unless the UI overrides them.
# temperature 0 keeps replies deterministic, which matters for the JSON format.
# max_tokens caps generation length only; llama.cpp cannot cap image tokens.
DEFAULT_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "max_tokens": 300,
}


def health(url, timeout=3.0):
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as response:
            return json.load(response).get("status") == "ok"
    except Exception:
        return False


def props(url, timeout=3.0):
    with urllib.request.urlopen(f"{url}/props", timeout=timeout) as response:
        return json.load(response)


def has_audio(url, timeout=3.0):
    try:
        modalities = props(url, timeout).get("modalities") or {}
        return bool(modalities.get("audio"))
    except Exception:
        return False


def to_wav_bytes(audio, sample_rate=16000):
    samples = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def _parse(content):
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
        else:
            data = None
    if isinstance(data, dict):
        transcript = str(data.get("transcript", "")).strip()
        reply = str(data.get("reply", "")).strip()
        if reply:
            return transcript, reply
    if text:
        logger.warning(
            "Model output was not valid JSON; falling back to raw text: %r",
            text[:200],
        )
    return "", text


def image_data_url(rgb, max_px=None):
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    if max_px and image.width * image.height > max_px:
        scale = (float(max_px) / (image.width * image.height)) ** 0.5
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        logger.debug(
            "Downscaling image %dx%d -> %dx%d for LLM.",
            image.width, image.height, size[0], size[1],
        )
        image = image.resize(size, Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


class _StringFieldStream:
    """Incrementally extracts the string values of given keys from a streamed
    JSON object, emitting decoded pieces as they arrive. Tolerates markdown
    fences and unusual whitespace between tokens."""

    _ESCAPES = {
        '"': '"', "\\": "\\", "/": "/", "b": "\b",
        "f": "\f", "n": "\n", "r": "\r", "t": "\t",
    }

    def __init__(self, keys):
        self._keys = set(keys)
        self._buf = ""
        self._state = "key"
        self._key = None
        self._unicode = None
        self._char_code = 0
        self._partial = ""

    def _decode_partial(self):
        out = []
        i = 0
        n = len(self._partial)
        while i < n:
            ch = self._partial[i]
            if self._unicode is not None:
                want = 4 - self._unicode
                have = min(want, n - i)
                digits = self._partial[i:i + have]
                try:
                    value = int(digits, 16)
                except ValueError:
                    return "".join(out), False
                self._char_code = (self._char_code << (4 * have)) | value
                self._unicode += have
                i += have
                if self._unicode == 4:
                    out.append(chr(self._char_code))
                    self._unicode = None
                continue
            if ch == "\\":
                if i + 1 >= n:
                    i += 1
                    break
                esc = self._partial[i + 1]
                if esc == "u":
                    self._unicode = 0
                    self._char_code = 0
                    i += 2
                    continue
                out.append(self._ESCAPES.get(esc, esc))
                i += 2
                continue
            out.append(ch)
            i += 1
        done = i >= n
        if done and self._unicode is not None:
            done = False
        self._partial = "" if done else self._partial[max(0, i):]
        return "".join(out), done

    def feed(self, text):
        out = []
        self._buf += text
        while True:
            if self._state == "key":
                q = self._buf.find('"')
                if q < 0:
                    self._buf = ""
                    break
                close = self._buf.find('"', q + 1)
                if close < 0:
                    self._buf = self._buf[q:]
                    break
                name = self._buf[q + 1:close]
                if name in self._keys:
                    self._key = name
                    self._buf = self._buf[close + 1:]
                    self._state = "open"
                else:
                    self._buf = self._buf[close + 1:]
            elif self._state == "open":
                q = self._buf.find('"')
                if q < 0:
                    self._buf = ""
                    break
                self._buf = self._buf[q + 1:]
                self._partial = ""
                self._state = "value"
            elif self._state == "value":
                q = self._find_value_end()
                if q < 0:
                    piece = self._decode_all_safe()
                    if piece:
                        out.append((self._key, piece))
                    break
                raw = self._buf[:q]
                self._partial += raw
                self._buf = self._buf[q + 1:]
                piece = self._decode_all()
                if piece:
                    out.append((self._key, piece))
                self._key = None
                self._state = "key"
        return out

    def _find_value_end(self):
        i = 0
        n = len(self._buf)
        while i < n:
            ch = self._buf[i]
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                return i
            i += 1
        return -1

    def _decode_all_safe(self):
        # Keep a possibly-escaped tail (and trailing backslash / partial
        # \uXXXX) in _partial for the next feed.
        n = len(self._buf)
        cut = n
        slashes = 0
        i = n - 1
        while i >= 0 and self._buf[i] == "\\":
            slashes += 1
            i -= 1
        if slashes % 2 == 1:
            cut = i + 1
        else:
            tail = self._buf[max(0, n - 6):]
            u = tail.rfind("\\u")
            if u >= 0:
                pos = max(0, n - 6) + u
                if n - pos < 6:
                    cut = pos
        safe = self._buf[:cut]
        self._buf = self._buf[cut:]
        self._partial += safe
        piece, _ = self._decode_partial()
        return piece

    def _decode_all(self):
        piece, _ = self._decode_partial()
        return piece

    def finish(self):
        piece = self._decode_all()
        if not piece:
            return None
        key = self._key if self._key in self._keys else "reply"
        return (key, piece)


def _sse_delta(line):
    line = line.decode("utf-8", "replace").strip()
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return False
    try:
        obj = json.loads(data)
    except ValueError:
        return None
    choices = obj.get("choices") or [{}]
    return (choices[0].get("delta") or {}).get("content") or ""


def ask(url, audio, sample_rate=16000, history=None, image=None, text=None,
        system_prompt=None, max_tokens=300, timeout=120, stream_events=None,
        image_max_px=None,
        temperature=DEFAULT_SAMPLING["temperature"],
        top_p=DEFAULT_SAMPLING["top_p"],
        top_k=DEFAULT_SAMPLING["top_k"],
        min_p=DEFAULT_SAMPLING["min_p"],
        repeat_penalty=DEFAULT_SAMPLING["repeat_penalty"]):
    messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:
        messages.append({"role": "user", "content": turn["transcript"]})
        messages.append({"role": "assistant", "content": turn["reply"]})
    content = []
    if image is not None:
        content.append({
            "type": "text",
            "text": (
                "A screenshot of what the user is looking at is attached. "
                "Use it when answering; mention what you see if relevant."
            ),
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": image_data_url(image, max_px=image_max_px)},
        })
    if audio is not None:
        content.append({
            "type": "text",
            "text": (
                "Answer the request in this audio, using the attached screenshot."
                if image is not None
                else "Answer the request in this audio."
            ),
        })
        content.append({
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(to_wav_bytes(audio, sample_rate)).decode(),
                "format": "wav",
            },
        })
    else:
        if not text and image is not None:
            text = (
                "The user sent the attached image with no text. "
                "Reply about what you see in it."
            )
        if text:
            content.append({"type": "text", "text": text})
    if not content:
        return "", ""
    messages.append({"role": "user", "content": content})
    payload = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repeat_penalty": repeat_penalty,
        "stream": bool(stream_events),
    }).encode()
    logger.debug(
        "LLM request: history=%d image=%s audio=%s payload=%dKB",
        len(messages) - 1,
        image is not None,
        "-" if audio is None else f"{len(audio) / sample_rate:.1f}s",
        len(payload) // 1024,
    )
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    if not stream_events:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        content = result["choices"][0]["message"].get("content", "")
        return _parse(content)
    fields = _StringFieldStream(("transcript", "reply"))
    raw_parts = []
    got = {"transcript": [], "reply": []}
    aborted = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            delta = _sse_delta(line)
            if delta is False:
                break
            if not delta:
                continue
            raw_parts.append(delta)
            pieces = fields.feed(delta)
            for key, piece in pieces:
                got[key].append(piece)
            if stream_events(pieces, got) is False:
                aborted = True
                break
    final = fields.finish()
    if final:
        key, piece = final
        got[key].append(piece)
        logger.warning(
            "JSON stream ended mid-value under %r (max tokens or cut-off?): %r",
            key, piece[:80],
        )
    if aborted:
        return None
    transcript = "".join(got["transcript"]).strip()
    reply = "".join(got["reply"]).strip()
    if reply:
        return transcript, reply
    return _parse("".join(raw_parts))
