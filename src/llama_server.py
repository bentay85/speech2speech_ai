"""Start and supervise the local llama.cpp server for Gemma 4 audio input."""
import ctypes
import os
import socket
import subprocess
import threading
import time
import urllib.parse
from ctypes import wintypes
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

import llm as llm_client

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OPTS = {
    "context": 8192,
    "gpu_layers": 99,
    "flash_attn": "auto",
    "kv_cache": "f16",
    "batch": 1024,
}


def _running_exe_path(pid):
    if os.name != "nt":
        return None
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        wintypes.LPDWORD,
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        buf = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(32768)
        if not kernel32.QueryFullProcessImageNameW(
            handle, 0, buf, ctypes.byref(size)
        ):
            return None
        return buf.value
    finally:
        kernel32.CloseHandle(handle)


class _BasicLimitInfo(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _ExtLimitInfo(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInfo),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _taskkill_tree(pid):
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class LlamaServer:
    def __init__(
        self,
        url="http://127.0.0.1:8080",
        bin_path=str(_ROOT / "bin" / "llama-server.exe"),
        model=str(
            _ROOT / "models" / "Gemma4-E4B" / "gemma-4-E4B-it-UD-Q4_K_XL.gguf"
        ),
        mmproj=str(_ROOT / "models" / "Gemma4-E4B" / "mmproj-BF16.gguf"),
        log_path=str(_ROOT / ".tmp" / "llama-server.log"),
        opts=None,
    ):
        self.url = url.rstrip("/")
        self._base_url = self.url
        self.bin_path = bin_path
        self.model = model
        self.mmproj = mmproj
        self.log_path = log_path
        self.opts = dict(DEFAULT_OPTS)
        if isinstance(opts, dict):
            for key, value in opts.items():
                if key in self.opts:
                    self.opts[key] = value
        self.process = None
        self.owned = False
        self._log_file = None
        self._shutdown = threading.Event()
        self._job = None
        self.pid_path = str(Path(self.log_path).parent / "llama-server.pid")
        self.command_parts = None

    def opts_args(self):
        """Build the full command from self.opts (ignoring any saved override)."""
        parts = urllib.parse.urlparse(self.url)
        host = parts.hostname or "127.0.0.1"
        port = str(parts.port or 8080)
        batch = str(int(self.opts["batch"]))
        args = [
            self.bin_path,
            "-m", self.model,
            "--mmproj", self.mmproj,
            "-ngl", str(int(self.opts["gpu_layers"])),
            "-c", str(int(self.opts["context"])),
            "-b", batch,
            "-ub", batch,
            "-fa", str(self.opts["flash_attn"]),
        ]
        kv = str(self.opts["kv_cache"])
        if kv != "f16":
            args += ["-ctk", kv, "-ctv", kv]
        args += [
            "--reasoning", "off",
            "--reasoning-budget", "0",
            "--host", host,
            "--port", port,
        ]
        return args

    def _args(self):
        if self.command_parts:
            return list(self.command_parts)
        return self.opts_args()

    def set_command(self, parts):
        """Override the whole server command line (None/empty restores defaults)."""
        self.command_parts = list(parts) if parts else None
        if not self.command_parts:
            self.url = self._base_url
            return
        self.bin_path = self.command_parts[0]
        host = port = None
        for i, part in enumerate(self.command_parts):
            nxt = self.command_parts[i + 1] if i + 1 < len(self.command_parts) else None
            if part in ("--host", "-h", "-a") and nxt is not None:
                host = nxt
            elif part in ("--port", "-p") and nxt is not None:
                port = nxt
        if host or port:
            parts = urllib.parse.urlparse(self.url)
            self.url = f"{parts.scheme or 'http'}://{host or parts.hostname}:{port or parts.port or 8080}"

    def restart(self, timeout=180.0):
        """Kill the owned server (if any) and start it again, e.g. after the
        command line changed. Returns True when the server is healthy after."""
        self._shutdown.clear()
        self.stop()
        time.sleep(0.5)
        return self.ensure_started(timeout=timeout)

    def health(self, timeout=1.5):
        return llm_client.health(self.url, timeout=timeout)

    def has_audio(self, timeout=3.0):
        return llm_client.has_audio(self.url, timeout=timeout)

    def _alternate_url(self):
        parts = urllib.parse.urlparse(self.url)
        host = parts.hostname or "127.0.0.1"
        bind_host = "127.0.0.1" if host in ("localhost", "0.0.0.0") else host
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, 0))
            port = sock.getsockname()[1]
        return f"http://{host}:{port}"

    def _reap_stale(self):
        """Kill an llama-server orphaned by a previous (crashed) app session."""
        try:
            pid = int(Path(self.pid_path).read_text().strip())
            Path(self.pid_path).unlink()
        except (OSError, ValueError):
            return
        exe = _running_exe_path(pid)
        if exe and Path(exe).resolve() == Path(self.bin_path).resolve():
            logger.info(f"Killing stale llama-server (pid {pid})...")
            _taskkill_tree(pid)
            time.sleep(0.5)

    def _assign_kill_job(self):
        """Put the server in a job that the OS kills when this app exits,
        even if it is killed or crashes without running closeEvent."""
        if os.name != "nt":
            return
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            ctypes.c_size_t,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return
        info = _ExtLimitInfo()
        info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            ctypes.c_void_p(job),
            9,  # JobObjectExtendedLimitInformation
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if ok and kernel32.AssignProcessToJobObject(
            ctypes.c_void_p(job), ctypes.c_void_p(int(self.process._handle))
        ):
            self._job = job
            return
        kernel32.CloseHandle(ctypes.c_void_p(job))

    def _clear_pid(self):
        try:
            Path(self.pid_path).unlink()
        except OSError:
            pass

    def ensure_started(self, timeout=180.0):
        if self._shutdown.is_set():
            return False
        self._reap_stale()
        if self.health(2.0):
            if self.has_audio():
                logger.info("llama-server already running with audio input.")
                self.owned = False
                return True
            logger.warning(
                "Existing llama-server has no audio input; starting a separate instance."
            )
            self.url = self._alternate_url()
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        if self._shutdown.is_set():
            return False
        paths = (
            (self.command_parts[0],)
            if self.command_parts
            else (self.bin_path, self.model, self.mmproj)
        )
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        self._log_file = open(self.log_path, "wb")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        logger.info("Starting llama-server: %s", " ".join(self._args()))
        t_start = time.time()
        self.process = subprocess.Popen(
            self._args(),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        self.owned = True
        Path(self.pid_path).write_text(str(self.process.pid))
        self._assign_kill_job()
        if self._shutdown.is_set():
            self.stop()
            return False
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._shutdown.is_set():
                self.stop()
                return False
            if self.health(2.0):
                for _ in range(10):
                    if self.has_audio():
                        break
                    time.sleep(0.2)
                else:
                    raise RuntimeError(
                        "llama-server started without audio input; "
                        f"check --mmproj and see {self.log_path}"
                    )
                logger.info(
                    f"llama-server ready in {time.time() - t_start:.1f}s "
                    "(audio input enabled)."
                )
                return True
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self.process.returncode}; "
                    f"see {self.log_path}"
                )
            time.sleep(0.5)
        raise TimeoutError(f"llama-server did not become healthy; see {self.log_path}")

    def stop(self):
        self._shutdown.set()
        if self.owned and self.process is not None and self.process.poll() is None:
            if os.name == "nt":
                _taskkill_tree(self.process.pid)
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        self.owned = False
        self._clear_pid()
        if self._job is not None:
            # KILL_ON_JOB_CLOSE: closing the handle kills anything still in the job.
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(self._job))
            self._job = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
