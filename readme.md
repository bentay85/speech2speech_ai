# Real-time speech-to-speech avatar (Windows)

Talk to a lip-synced AI avatar: Silero VAD gates the microphone, Gemma 4
(audio in, via llama.cpp) answers, Kokoro TTS speaks the reply, and the
FlashHead avatar animates in real time.

```text
mic -> Silero VAD -> llama.cpp (Gemma 4 E4B + mmproj audio) -> Kokoro TTS -> speakers
                                     |                          |
                              optional screenshot/webcam        +-> FlashHead avatar (lip sync)
```

Tested on Windows 11, NVIDIA RTX 4060 Ti (16 GB), Python 3.10,
torch 2.7.1 + CUDA 12.8, llama.cpp b11748. Run every command from the
repository root.

## 1. Before running setup

`setup.bat` is the one-shot installer: it creates `.venv`, installs
`requirements.txt`, downloads every model into `models\`, fetches the
llama-server binaries into `bin\`, and tries to install eSpeak NG.
`run-app.bat` starts the app afterwards.

What must exist first:

| Requirement | Notes |
| :--- | :--- |
| Windows 10 1803+ or Windows 11 (x64) | `setup.bat` uses `tar` to unpack the llama.cpp zip. |
| NVIDIA GPU with a CUDA 12.x driver | ~12 GB VRAM needed at runtime (LLM ~5.7 GB, avatar ~5 GB, TTS ~0.4 GB). No CUDA Toolkit install is needed: the CUDA runtime ships in the torch wheels and in `bin\`. |
| Python 3.10 | Only needed if `uv` is absent; `uv venv --python 3.10` downloads its own CPython. 3.11+ is not supported by the pinned stack. |
| `uv` (recommended) | Not required, but `setup.bat` prefers it and it removes the Python-version headache: `winget install astral-sh.uv` or <https://docs.astral.sh/uv/>. Must be on `PATH` (`where uv`). |
| ~12 GB free disk | ~7 GB models, ~3 GB venv, ~1 GB `bin\`, plus compile caches in `.cache\`. |
| Internet access | Downloads models from Hugging Face and llama.cpp/eSpeak NG from GitHub. |
| eSpeak NG 1.52.0 | Required by Kokoro TTS for English fallback. `setup.bat` downloads and silently installs it (admin prompt not needed for the MSI, but a new terminal is needed for `PATH`). Manual: <https://github.com/espeak-ng/espeak-ng/releases>. |
| Administrator rights (optional) | Only for the eSpeak NG silent install; otherwise the script prints manual steps. |

Optional, recommended for `torch.compile`: enable long paths
(`LongPathsEnabled = 1` under `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`).

Quick preflight:

```powershell
nvidia-smi                 # driver + GPU visible
uv --version               # if you use the uv path
where espeak-ng            # only if already installed
```

## 2. Install (quick start)

```bat
setup.bat
```

The script is re-runnable and skips work that is already done. When it
finishes, start the app:

```bat
run-app.bat
```

`run-app.bat` uses `python.exe`, so a console window shows live logs, and the
same output also goes to `logs\app.log`. For a window with no console run
`.venv\Scripts\pythonw.exe src\app.py` instead.

### Manual install

If you prefer not to use the batch files:

```powershell
uv venv .venv --python 3.10
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Then download the models (section 3), unpack the llama.cpp CUDA build into
`bin\`, and install eSpeak NG 1.52.0. `requirements.txt` installs the CUDA
12.8 PyTorch build (pinned as `torch==2.7.1+cu128`, which only exists on
PyTorch's wheel index), `triton-windows`, the SageAttention wheel and the
`hf` CLI, so no separate `huggingface_hub` step is needed.

Sanity-check the GPU build after installing:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It must print a `+cu128` version and `True`. If it prints `2.7.1+cpu` or
`False`, PyTorch was resolved from PyPI instead of the CUDA index.

## 3. Models

`setup.bat` places everything in `models\`; all paths are resolved from the
repository root, not the current working directory.

```text
models\
  SoulX-FlashHead-1_3B\   avatar DiT + VAE (only Model_Lite\ and VAE_LTX\ are used)
  wav2vec2-base-960h\     audio encoder for the avatar
  Gemma4-E4B\             LLM: gemma-4-E4B-it-UD-Q4_K_XL.gguf + mmproj-BF16.gguf
  Kokoro-82M\             TTS: config.json, kokoro-v1_0.pth, voices\*.pt
  silero-vad\             silero_vad.onnx
```

| Folder | Download from | License | Size |
| :--- | :--- | :--- | ---: |
| `SoulX-FlashHead-1_3B` | [Soul-AILab/SoulX-FlashHead-1_3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B) · [code](https://github.com/Soul-AILab/SoulX-FlashHead) | Apache-2.0 (DiT/code); `VAE_LTX` is OpenRAIL-M | ~7.2 GB |
| `wav2vec2-base-960h` | [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) | Apache-2.0 | ~1.1 GB |
| `Gemma4-E4B` | [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) · base [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | Apache-2.0 | ~5.8 GB |
| `Kokoro-82M` | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) · [code](https://github.com/hexgrad/kokoro) | Apache-2.0 | ~0.3 GB |
| `silero-vad` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) (copied out of the pip package) | MIT | ~2 MB |

Manual download commands:

```powershell
.venv\Scripts\hf.exe download Soul-AILab/SoulX-FlashHead-1_3B --local-dir models\SoulX-FlashHead-1_3B
.venv\Scripts\hf.exe download facebook/wav2vec2-base-960h --local-dir models\wav2vec2-base-960h
.venv\Scripts\hf.exe download unsloth/gemma-4-E4B-it-GGUF --include "gemma-4-E4B-it-UD-Q4_K_XL.gguf" "mmproj-BF16.gguf" --local-dir models\Gemma4-E4B
.venv\Scripts\hf.exe download hexgrad/Kokoro-82M --include "config.json" "kokoro-v1_0.pth" "voices/*.pt" --local-dir models\Kokoro-82M
New-Item -ItemType Directory -Force models\silero-vad | Out-Null
Copy-Item .venv\Lib\site-packages\silero_vad\data\silero_vad.onnx models\silero-vad\
```

`mmproj-BF16.gguf` is required: it is what gives the LLM audio input.
In mainland China set `$env:HF_ENDPOINT = "https://hf-mirror.com"` first.
`Model_Pro\` and `VAE_Wan\` inside `SoulX-FlashHead-1_3B` are unused and can
be deleted.

`bin\` holds the tested `llama-server.exe` (llama.cpp CUDA build b11748) plus
its CUDA runtime DLLs; `src\app.py` starts and stops it automatically.

## 4. Running the app

```powershell
.venv\Scripts\python.exe src\app.py     # console + window
.venv\Scripts\pythonw.exe src\app.py    # window only, logs to logs\app.log
run-app.bat                             # same as python.exe
```

Wait for the four status dots — LLM server (Gemma 4), Avatar model
(SoulX FlashHead), Voice activity detector (Silero) and Text-to-speech
(Kokoro) — to turn green, then press **Start conversation**. Pick a WDM-KS
microphone, speak, and the avatar replies live; interrupt any time by talking
over it.

> **Use headphones.** No acoustic echo cancellation (AEC) is implemented, so
> the avatar's own speech played through speakers is captured by the
> microphone and can interrupt or derail the conversation. Use headphones or
> a headset instead of speakers.

On the first launch, and the first time you select a new avatar resolution,
the avatar is compiled with `torch.compile`. Its status dot stays on
`Avatar model (SoulX FlashHead) - compiling <size>px` for a long time — 10+
minutes is not unusual — before it turns green, and **Start conversation**
stays disabled meanwhile. This is expected; progress is logged to
`logs\app.log`. The compiled graphs are cached in `.cache\`, so later launches
at the same resolution are much faster. Deleting `.cache\` forces a full
recompile.

In-app controls: avatar condition image (`avatar_images\`, default
`avatar_images\female.jpg`), avatar name (replaces `<Name>` in the system
prompt), avatar resolution, voice (28 US/UK Kokoro voices), desktop/webcam
capture (the current screen or a webcam frame is attached to the LLM on each
turn; a region can be given by pixel coordinates), an editable system prompt,
Llama.cpp command, sampling, Silero, Kokoro and FlashHead settings, and a text
box for typing messages or dropping images instead of speaking. All settings
are stored in `settings.json` and reloaded on the next launch.

`settings.json` is local to each machine and gitignored (it may contain
absolute paths such as the llama.cpp command). On a fresh clone the built-in
defaults apply: avatar name `Heart`, `avatar_images\female.jpg`, 384px, seed
42, voice `af_heart`, a built-in system prompt that introduces the avatar by
name (see `DEFAULT_PROMPT_TEMPLATE` in `src\app.py`), and an empty
`llama_command`, so the server command is rebuilt from `bin\`, `models\` and
the sampling options.

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--size` | `384` | Initial avatar resolution. |
| `--voice` | `af_heart` | Initial Kokoro voice. |
| `--cond_image` | saved setting | Initial avatar image. |
| `--base_seed` | saved setting | Initial generator seed. |
| `--av_delay_ms` | `0` | Positive delays video, negative advances it. |
| `--input_wav` | off | Debug: stream a WAV through the mic path. |
| `--server` | `http://127.0.0.1:8080` | Reuse an already-running llama-server. |
| `--ckpt_dir` | `models\SoulX-FlashHead-1_3B` | Avatar checkpoint directory. |
| `--wav2vec_dir` | `models\wav2vec2-base-960h` | Audio encoder directory. |

Changing avatar resolution recompiles the model for the new shape (first use
of a size can take minutes; graphs are cached in `.cache\`).

## 5. Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| Status dot for LLM is red, `.tmp\llama-server.log` mentions `mmproj` | `models\Gemma4-E4B\mmproj-BF16.gguf` is missing or not the BF16 projector. |
| `llama-server exited with code ...` and a CUDA error | Driver too old for the CUDA 12.4 build, or `bin\` lacks the `cudart`/`cublas` DLLs; re-run `setup.bat`. |
| `torch.cuda.is_available()` is `False` | CPU wheel installed; reinstall with `uv pip install --python .venv\Scripts\python.exe --reinstall torch torchvision`. |
| Kokoro output is silent on unknown words | eSpeak NG not installed / not on `PATH`; open a new terminal after installing. |
| Avatar interrupts itself or replies to its own voice | No AEC is implemented; use headphones or a headset instead of speakers. |
| `ModuleNotFoundError: flash_head.ltx_video.models...` | The checkout is missing the vendored LTX VAE sources; they are part of this repository — re-clone/pull. |
| Windows path errors during `torch.compile` | Enable long paths (see section 1). |

Logs: `logs\app.log` (truncated on every start, set `LOG_LEVEL=DEBUG` before
launching) and `.tmp\llama-server.log`.

## 6. Open-source models and licences

This project orchestrates several separately licensed models. None of the
weights are stored in this repository; `setup.bat` downloads them at install
time, so each model remains under its own licence.

| Model / part | Used for | Upstream | Licence |
| :--- | :--- | :--- | :--- |
| Gemma 4 E4B IT (GGUF Q4_K_XL + BF16 mmproj) | LLM that answers, audio-in | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it), quantised by [unsloth](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) | Apache-2.0 |
| SoulX-FlashHead-1.3B (DiT) | Avatar video generation | [Soul-AILab/SoulX-FlashHead-1_3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B), [code](https://github.com/Soul-AILab/SoulX-FlashHead) | Apache-2.0 |
| LTX-Video 0.9 VAE (`VAE_LTX`) | Avatar VAE decode (Lite variant) | [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video) via SoulX-FlashHead | **OpenRAIL-M** |
| Wan 2.1 components (`Model_Pro`, optional) | Pro avatar variant, not used here | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | Apache-2.0 |
| wav2vec2-base-960h | Audio features for the avatar | [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) · code in [transformers](https://github.com/huggingface/transformers) | Apache-2.0 |
| Kokoro-82M | Text-to-speech | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), [hexgrad/kokoro](https://github.com/hexgrad/kokoro) | Apache-2.0 |
| Silero VAD v6 | Voice activity detection | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | MIT |
| Self-Forcing / DMD (training lineage of FlashHead) | Method only; no code vendored from these | [Self-Forcing](https://github.com/guandeh17/Self-Forcing), [DMD2](https://github.com/tianweiy/DMD2) | see upstream repos |

Runtime software bundled or required:

| Component | Role | Licence |
| :--- | :--- | :--- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) b11748 (CUDA 12.4 Windows build) | Runs the Gemma 4 GGUF and the audio projector; started as a subprocess | MIT |
| [PyTorch](https://github.com/pytorch/pytorch) 2.7.1+cu128 / torchvision 0.22.1+cu128 | Avatar and Kokoro inference | BSD-3-Clause |
| [NVIDIA CUDA runtime](https://docs.nvidia.com/cuda/eula/index.html) (`cudart`, `cublas`, shipped in `bin\`) | GPU runtime for llama.cpp | NVIDIA CUDA Toolkit EULA (redistributable runtime) |
| [SageAttention](https://github.com/thu-ml/SageAttention) 2.2.0 (Windows wheels from [woct0rdho](https://github.com/woct0rdho/SageAttention)) | Fast attention for FlashHead | Apache-2.0 |
| [triton-windows](https://github.com/woct0rdho/triton-windows) | `torch.compile` backend on Windows | MIT |
| [diffusers](https://github.com/huggingface/diffusers), [transformers](https://github.com/huggingface/transformers), [accelerate](https://github.com/huggingface/accelerate) | Model plumbing | Apache-2.0 |
| [eSpeak NG](https://github.com/espeak-ng/espeak-ng) 1.52.0 | English G2P fallback for Kokoro (installed system-wide, loaded at runtime) | **GPL-3.0** |
| [misaki](https://github.com/hexgrad/misaki), [spaCy](https://github.com/explosion/spaCy) + `en_core_web_sm` | G2P for English | MIT |
| [PySide6-Essentials](https://doc.qt.io/qtforpython-6/) | Desktop UI (Qt Widgets) | LGPL-3.0 |
| [OpenCV](https://github.com/opencv/opencv), [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Webcam / screen capture helpers | Apache-2.0 |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) + PortAudio, [soxr](https://github.com/dofuuz/python-soxr) | Audio I/O and resampling | MIT / MIT / LGPL-2.1 |

Vendored source under `src/flash_head/` is derived from the Wan and
SoulX-FlashHead code bases and keeps its original Apache-2.0 copyright
headers (`# Copyright 2024-2025 The Alibaba Wan Team Authors`, and the
diffusers-derived attention/VAE code).

### Licence of this repository

**Apache License 2.0** — see [LICENSE](LICENSE). This is the right choice
here because:

- every piece of source code in the repository is Apache-2.0 or
  Apache-2.0-compatible (vendored Wan/SoulX/diffusers code, plus MIT
  dependencies), so no other code licence imposes a different term;
- it keeps the explicit patent grant of the upstream code while staying
  permissive enough for people to reuse the app;
- the model weights are downloaded separately, so their licences do not
  change the repository licence.

Caveats to keep in mind before publishing or redistributing:

1. **eSpeak NG is GPL-3.0.** It is installed as a separate system component
   and only invoked at runtime; it is not distributed in this repository.
   Do not bundle its DLLs into a release without complying with the GPL
   (shipping the corresponding source and the GPL text). Keeping it a
   user-installed dependency, as `setup.bat` does, avoids that.
2. **`VAE_LTX` weights are OpenRAIL-M**, not Apache-2.0. They are downloaded
   from Hugging Face and must keep the OpenRAIL-M terms if you redistribute
   them; you cannot relicense them under Apache-2.0.
3. **`avatar_images/female.jpg` and `male.jpg`** are synthetic images generated
   with Gemini, not photos of real people. If you replace them, use images you
   have the right to distribute (or a CC0/stock source) before making the
   repository public.
4. If you publish a release that includes compiled binaries of the vendored
   code, add a `NOTICE` file with the Wan/SoulX attribution; Apache-2.0
   requires preserving the existing copyright headers (they are already kept
   in the source files).

## 7. Notes

- Single NVIDIA GPU: LLM + projector ~5.7 GB, avatar at 384p ~5 GB, Kokoro on
  GPU (~11.5 GB of 16 GB total).
- Compiled graphs cache in `.cache\`; delete to force a rebuild (the next
  launch then spends a long time recompiling, see section 4). `.tmp\` is
  gitignored scratch.
- Voice-loop audio I/O uses PortAudio; only `Windows WDM-KS` input devices are
  offered for lowest latency.
