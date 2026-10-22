"""Microphone capture and speaker playback via PortAudio (sounddevice).

Mic audio is delivered as 16 kHz mono float32 blocks of 512 samples (Silero VAD
chunk size) into ``mic_q``. Playback accepts arbitrary-length float32 audio and
is resampled to the output device's native rate; ``flush_output`` drops queued
audio immediately so barge-in stops the assistant mid-word.
"""
import collections
import ctypes
import os
import queue
import re
import threading
import time
import uuid
from ctypes import addressof, byref, c_int32, c_ubyte, c_uint32, c_ushort, c_void_p
from fractions import Fraction

import numpy as np
import sounddevice as sd
import logging

logger = logging.getLogger(__name__)

MIC_SR = 16000
MIC_BLOCK = 512
TTS_SR = 24000

# WDM-KS reports raw endpoint IDs for some devices, e.g.
#   Headset (@System32\drivers\bthhfenum.sys,#2;%1 Hands-Free%0\;(EarFun Air Pro 4))
# Collapse those to "<friendly name> (<role>)".
_DRIVER_NAME_RE = re.compile(
    r"^(?P<desc>.+?)\s*\(@[^,]+,[^;]+;%1\s*(?P<role>[^%]+)%0\\?\s*;\s*"
    r"\((?P<name>[^()]+)\)\)$"
)


def clean_device_name(name):
    m = _DRIVER_NAME_RE.match(name.strip())
    if m:
        real = m.group("name").strip()
        role = m.group("role").strip()
        return f"{real} ({role})" if role else real
    return re.sub(r"\s*\(\s*\)\s*$", "", name).strip()


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", c_uint32),
        ("Data2", c_ushort),
        ("Data3", c_ushort),
        ("Data4", c_ubyte * 8),
    ]


class _PropVariant(ctypes.Structure):
    _fields_ = [
        ("vt", c_ushort),
        ("r1", c_ushort),
        ("r2", c_ushort),
        ("r3", c_ushort),
        ("p1", c_void_p),
        ("p2", c_void_p),
    ]


class _PropertyKey(ctypes.Structure):
    _fields_ = [
        ("fmtid", _GUID),
        ("pid", c_uint32),
    ]


def _guid(text):
    return _GUID.from_buffer_copy(uuid.UUID(text.strip("{}")).bytes_le)


def _com_method(this, index, argtypes, restype=c_int32):
    vtable = ctypes.cast(this, ctypes.POINTER(ctypes.POINTER(c_void_p))).contents
    proto = ctypes.CFUNCTYPE(restype, c_void_p, *argtypes)
    return proto(vtable[index])


_DEVICE_STATE_ACTIVE = 1
_DEVICE_STATE_UNAVAILABLE = 2 | 4 | 8  # DISABLED | NOTPRESENT | UNPLUGGED


def _endpoint_names(flow, state_mask):
    """Friendly names of MMDevice endpoints in the given states (ctypes COM).

    flow: 0 = render (speakers), 1 = capture (microphones). Returns None when
    the query is unavailable, in which case no filtering is applied.
    """
    if os.name != "nt":
        return None
    try:
        ole32 = ctypes.windll.ole32
        ole32.CoInitializeEx(None, 2)  # already-initialized returns a warning hr
        ole32.CoCreateInstance.restype = c_int32
        ole32.CoCreateInstance.argtypes = [
            ctypes.c_void_p,
            c_void_p,
            c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(c_void_p),
        ]
        enumerator = c_void_p()
        clsid = _guid("BCDE0395-E52F-467C-8E3D-C4579291692E")
        iid_unk = _guid("00000000-0000-0000-C000-000000000046")
        hr = ole32.CoCreateInstance(
            addressof(clsid),
            None,
            1,  # CLSCTX_INPROC_SERVER
            addressof(iid_unk),
            byref(enumerator),
        )
        if hr < 0:
            return None
        # IMMDeviceEnumerator::EnumAudioEndpoints(dataFlow, stateMask, **)
        collection = c_void_p()
        hr = _com_method(enumerator, 3, [c_uint32, c_uint32, ctypes.POINTER(c_void_p)])(
            enumerator, flow, state_mask, byref(collection)
        )
        names = set()
        if hr >= 0:
            count = c_uint32()
            _com_method(collection, 3, [ctypes.POINTER(c_uint32)])(
                collection, byref(count)
            )
            fmtid = _guid("a45c254e-df1c-4efd-8020-67d146a850e0")  # FMTID_Device
            key = _PropertyKey()
            key.fmtid = fmtid
            key.pid = 14  # PKEY_Device_FriendlyName
            for i in range(count.value):
                device = c_void_p()
                if _com_method(collection, 4, [c_uint32, ctypes.POINTER(c_void_p)])(
                    collection, i, byref(device)
                ) < 0:
                    continue
                store = c_void_p()
                ok = _com_method(device, 4, [c_uint32, ctypes.POINTER(c_void_p)])(
                    device, 0, byref(store)
                )  # OpenPropertyStore(STGM_READ)
                if ok >= 0:
                    value = _PropVariant()
                    hr = _com_method(
                        store,
                        5,
                        [ctypes.c_void_p, ctypes.POINTER(_PropVariant)],
                    )(store, addressof(key), byref(value))  # IPropertyStore::Get
                    if hr >= 0 and value.vt == 31 and value.p1:
                        name = ctypes.wstring_at(value.p1)
                        if name.strip():
                            names.add(name.strip())
                    if hr >= 0:
                        ole32.PropVariantClear.restype = c_int32
                        ole32.PropVariantClear.argtypes = [ctypes.POINTER(_PropVariant)]
                        ole32.PropVariantClear(byref(value))
                    _com_method(store, 2, [])(store)  # Release
                _com_method(device, 2, [])(device)
            _com_method(collection, 2, [])(collection)
        _com_method(enumerator, 2, [])(enumerator)
        return names
    except Exception as exc:
        logger.debug(f"MMDevice endpoint query failed: {exc}")
        return None


def _name_tokens(name):
    return set(re.findall(r"[a-z0-9]+", name.lower()))


_GENERIC_TOKENS = {
    "speakers", "microphone", "microphones", "headphones", "headset",
    "output", "input", "audio", "device",
}


def _token_match(ks, other):
    """True when a PortAudio name and an MMDevice friendly name plausibly
    describe the same device.  Driver-reported and PnP-reported names differ
    ("Realtek HD Audio output" vs "Realtek(R) Audio"), so we allow a shared
    token overlap rather than exact equality, but require at least one
    non-generic shared token so unrelated devices don't collide."""
    if not ks or not other:
        return False
    if ks <= other or other <= ks:
        return True
    shared = ks & other
    return len(shared) >= 2 and bool(shared - _GENERIC_TOKENS)


def _connected_only(devices, flow):
    """Drop WDM-KS entries whose Windows endpoint is disabled or unplugged
    (e.g. a powered-off Bluetooth device still enumerated by PortAudio).

    A device is kept when it matches an ACTIVE endpoint, or when it matches
    no endpoint at all (naming can differ between PortAudio and MMDevice, so
    we only hide names we can positively confirm as unavailable)."""
    active = _endpoint_names(flow, _DEVICE_STATE_ACTIVE)
    unavailable = _endpoint_names(flow, _DEVICE_STATE_UNAVAILABLE)
    if active is None and unavailable is None:
        return devices
    active = {frozenset(_name_tokens(n)) for n in (active or ())}
    unavailable = {frozenset(_name_tokens(n)) for n in (unavailable or ())}

    def matches(names, key):
        return any(_token_match(key, other) for other in names)

    kept = []
    for index, name in devices:
        key = frozenset(_name_tokens(clean_device_name(name)))
        if matches(active, key) or not matches(unavailable, key):
            kept.append((index, name))
        else:
            logger.info(f"Skipping unavailable audio device: {clean_device_name(name)}")
    return kept


def input_devices():
    """Microphones on the WDM-KS host API: lowest latency on Windows."""
    return _connected_only(
        [
            (index, device["name"])
            for index, device in enumerate(sd.query_devices())
            if device["max_input_channels"] > 0
            and sd.query_hostapis(device["hostapi"])["name"] == "Windows WDM-KS"
        ],
        flow=1,
    )


def output_devices():
    """Speakers on the WDM-KS host API: lowest latency on Windows."""
    return _connected_only(
        [
            (index, device["name"])
            for index, device in enumerate(sd.query_devices())
            if device["max_output_channels"] > 0
            and sd.query_hostapis(device["hostapi"])["name"] == "Windows WDM-KS"
        ],
        flow=0,
    )


def default_devices():
    return sd.default.device


def device_name(index):
    try:
        return sd.query_devices(index)["name"].strip()
    except Exception:
        return f"device {index}"


def device_label(index):
    try:
        return clean_device_name(sd.query_devices(index)["name"])
    except Exception:
        return f"device {index}"


def refresh_devices():
    """Re-scan PortAudio's device list so newly connected devices show up."""
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        pass


def _resample(audio, orig_sr, target_sr):
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32)
    from scipy.signal import resample_poly

    ratio = Fraction(int(target_sr), int(orig_sr)).limit_denominator(1000)
    return resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)


class AudioEngine:
    def __init__(self, in_device=None, out_device=None):
        self.in_device = in_device
        self.out_device = out_device
        self.mic_q = queue.Queue(maxsize=400)
        self.out_sr = TTS_SR
        self.out_latency = 0.0
        self.in_sr = MIC_SR
        self._out_buf = collections.deque()
        self._in_buf = collections.deque()
        self._in_resampler = None
        self._last_status_log = 0.0
        self._out_lock = threading.Lock()
        self._in_stream = None
        self._out_stream = None
        self._file_thread = None
        self._stop = threading.Event()

    def open_output(self):
        if self.out_device is None:
            self.out_sr = TTS_SR
        else:
            info = sd.query_devices(self.out_device)
            self.out_sr = int(info["default_samplerate"])
        self._out_stream = sd.OutputStream(
            samplerate=self.out_sr,
            channels=1,
            dtype="float32",
            blocksize=1024,
            device=self.out_device,
            callback=self._out_cb,
        )
        self._out_stream.start()
        self.out_latency = float(getattr(self._out_stream, "latency", 0.0) or 0.0)
        logger.info(
            f"Speaker open at {self.out_sr} Hz, output latency "
            f"{self.out_latency * 1000:.0f} ms."
        )

    def open_input(self, wav_path=None):
        if wav_path:
            self._file_thread = threading.Thread(
                target=self._stream_file, args=(wav_path,), daemon=True
            )
            self._file_thread.start()
            return
        try:
            self._in_stream = sd.InputStream(
                samplerate=MIC_SR,
                channels=1,
                dtype="float32",
                blocksize=MIC_BLOCK,
                device=self.in_device,
                callback=self._in_cb,
            )
            self._in_stream.start()
        except Exception as exc:
            self._open_input_native(exc)

    def _open_input_native(self, error):
        import soxr

        if self.in_device is None:
            info = sd.query_devices(sd.default.device[0])
        else:
            info = sd.query_devices(self.in_device)
        self.in_sr = int(info["default_samplerate"])
        if self.in_sr == MIC_SR:
            raise error
        blocksize = max(1, round(MIC_BLOCK * self.in_sr / MIC_SR))
        self._in_resampler = soxr.ResampleStream(
            self.in_sr, MIC_SR, 1, dtype="float32"
        )
        self._in_buf.clear()
        self._in_stream = sd.InputStream(
            samplerate=self.in_sr,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=self.in_device,
            callback=self._in_cb,
        )
        self._in_stream.start()
        logger.info(
            f"Microphone opened at {self.in_sr} Hz; resampling to {MIC_SR} Hz."
        )

    def _in_cb(self, indata, frames, time_info, status):
        if status:
            now = time.perf_counter()
            if now - self._last_status_log > 5.0:
                self._last_status_log = now
                logger.warning(f"Microphone stream status: {status}")
        try:
            block = np.array(indata[:, 0], dtype=np.float32, copy=True)
            if self._in_resampler is not None:
                block = self._in_resampler.resample_chunk(block)
                self._in_buf.extend(block.tolist())
                while len(self._in_buf) >= MIC_BLOCK:
                    out = np.fromiter(
                        (self._in_buf.popleft() for _ in range(MIC_BLOCK)),
                        dtype=np.float32,
                        count=MIC_BLOCK,
                    )
                    try:
                        self.mic_q.put_nowait(out)
                    except queue.Full:
                        pass
            else:
                self.mic_q.put_nowait(block)
        except queue.Full:
            pass

    def _stream_file(self, wav_path):
        import soundfile as sf

        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _resample(audio, sr, MIC_SR)
        step = MIC_BLOCK
        start = time.perf_counter()
        for i in range(0, len(audio), step):
            if self._stop.is_set():
                break
            block = audio[i:i + step]
            if len(block) < step:
                block = np.pad(block, (0, step - len(block)))
            target = start + (i + step) / MIC_SR
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            try:
                self.mic_q.put_nowait(block.astype(np.float32))
            except queue.Full:
                pass
        silence = np.zeros(step, dtype=np.float32)
        tail_end = time.perf_counter() + 1.2
        while not self._stop.is_set() and time.perf_counter() < tail_end:
            try:
                self.mic_q.put_nowait(silence)
            except queue.Full:
                pass
            time.sleep(step / MIC_SR)

    def _out_cb(self, outdata, frames, time_info, status):
        with self._out_lock:
            need = frames
            while need > 0 and self._out_buf:
                take = min(need, len(self._out_buf))
                for i in range(take):
                    outdata[len(outdata) - need + i, 0] = self._out_buf.popleft()
                need -= take
            if need > 0:
                outdata[frames - need:, 0] = 0.0

    def play(self, audio):
        if len(audio) == 0:
            return
        audio = _resample(np.asarray(audio, dtype=np.float32), TTS_SR, self.out_sr)
        with self._out_lock:
            self._out_buf.extend(audio.tolist())

    def flush_output(self):
        with self._out_lock:
            self._out_buf.clear()

    def queued_seconds(self):
        with self._out_lock:
            return len(self._out_buf) / max(self.out_sr, 1)

    def stop(self):
        self._stop.set()
        if self._in_stream is not None:
            self._in_stream.stop()
            self._in_stream.close()
            self._in_stream = None
        if self._out_stream is not None:
            self._out_stream.stop()
            self._out_stream.close()
            self._out_stream = None
        if self._file_thread is not None:
            self._file_thread.join(timeout=2.0)
            self._file_thread = None
        self.flush_output()
