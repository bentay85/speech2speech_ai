# Real-time speech-to-speech avatar (Windows)

Talk to a lip-synced AI avatar: Silero VAD gates the microphone, Gemma 4
(audio via llama.cpp) answers, Kokoro TTS speaks the reply, and the FlashHead
avatar animates in real time. 

Tested on Windows 11, NVIDIA RTX 4060 Ti (16 GB), Python 3.10,
torch 2.7.1 + CUDA 12.8. Run all commands from the repository root.

## 1. Prerequisites

- NVIDIA driver with CUDA 12.x support; Python 3.10; [uv](https://docs.astral.sh/uv/).
- **eSpeak NG 1.52.0** (required by Kokoro TTS for English fallback):
  install `espeak-ng-1.52.0-x64.msi` from
  <https://github.com/espeak-ng/espeak-ng/releases>, then verify in a **new**
  terminal: `espeak-ng --version`.
- Optional, recommended for `torch.compile`: enable long paths
  (`LongPathsEnabled` = 1 under `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`).

## 2. Install

```powershell
uv venv .venv --python 3.10
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

`requirements.txt` covers everything: CUDA 12.8 PyTorch (extra index is in the
file), `triton-windows`, and the prebuilt SageAttention wheel. Comment out the
SageAttention line if you don't want it; the model falls back to PyTorch SDPA.

## 3. Models

Download these into `models\` in the repository root (all paths are resolved
from the repo, not the working directory):

```text
models\
  SoulX-FlashHead-1_3B\   avatar DiT + VAE (only Model_Lite\ and VAE_LTX\ are used)
  wav2vec2-base-960h\     audio encoder for the avatar
  Gemma4-E4B\             LLM: gemma-4-E4B-it-UD-Q4_K_XL.gguf + mmproj-BF16.gguf
  Kokoro-82M\             TTS: config.json, kokoro-v1_0.pth, voices\*.pt
  silero-vad\             silero_vad.onnx
```

| Folder | Download from |
| :--- | :--- |
| `SoulX-FlashHead-1_3B` | [Soul-AILab/SoulX-FlashHead-1_3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B) |
| `wav2vec2-base-960h` | [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) |
| `Gemma4-E4B` | [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) |
| `Kokoro-82M` | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| `silero-vad` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) |

```powershell
uv pip install --python .venv\Scripts\python.exe "huggingface_hub[cli]"
.venv\Scripts\hf.exe download Soul-AILab/SoulX-FlashHead-1_3B --local-dir models\SoulX-FlashHead-1_3B
.venv\Scripts\hf.exe download facebook/wav2vec2-base-960h --local-dir models\wav2vec2-base-960h
.venv\Scripts\hf.exe download unsloth/gemma-4-E4B-it-GGUF --include "gemma-4-E4B-it-UD-Q4_K_XL.gguf" --include "mmproj-BF16.gguf" --local-dir models\Gemma4-E4B
.venv\Scripts\hf.exe download hexgrad/Kokoro-82M --include "config.json" --include "kokoro-v1_0.pth" --include "voices/*.pt" --local-dir models\Kokoro-82M
New-Item -ItemType Directory -Force models\silero-vad | Out-Null
Copy-Item .venv\Lib\site-packages\silero_vad\data\silero_vad.onnx models\silero-vad\
```

`mmproj-BF16.gguf` is required: it is what gives the LLM audio input. In
mainland China run `$env:HF_ENDPOINT = "https://hf-mirror.com"` first.
`Model_Pro\` and `VAE_Wan\` inside `SoulX-FlashHead-1_3B` can be deleted.

`bin\` holds the tested `llama-server.exe` (llama.cpp CUDA build b11748) plus
its CUDA runtime DLLs; `src/app.py` starts and stops it automatically.

## 4. Voice assistant (`src/app.py`)

```powershell
.venv\Scripts\python.exe src\app.py     # window with all models loading at startup
```

Use `pythonw.exe` instead of `python.exe` for a console-free desktop app; logs
then go to `logs\app.log`. Wait for the four status dots to turn green, then
press **Start conversation**. Pick a WDM-KS microphone, speak, and the avatar
replies live; interrupt any time by talking over it.

In-app controls: avatar condition image (`avatar_images\`, default
`avatar_images\female.jpg`), avatar name (replaces `<Name>` in the system
prompt), avatar resolution, voice (28 US/UK Kokoro voices), desktop/webcam
capture (the current screen or a webcam frame is attached to the LLM on each
turn; a region can be given by pixel coordinates), an editable system prompt,
Llama.cpp command, sampling, Silero, Kokoro and FlashHead settings, and a text
box for typing messages or dropping images instead of speaking. All settings
are stored in `settings.json` and reloaded on the next launch.

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--size` | `384` | Initial avatar resolution. |
| `--voice` | `af_heart` | Initial Kokoro voice. |
| `--cond_image` | `avatar_images\female.jpg` | Initial avatar image. |
| `--av_delay_ms` | `0` | Positive delays video, negative advances it. |
| `--input_wav` | off | Debug: stream a WAV through the mic path. |
| `--server` | `http://127.0.0.1:8080` | Reuse an already-running llama-server. |

Changing avatar resolution recompiles the model for the new shape (first use of
a size can take minutes; graphs are cached in `.cache\`).

## 5. Notes

- Single NVIDIA GPU: LLM + projector ~5.7 GB, avatar at 384p ~5 GB, Kokoro on
  GPU (~11.5 GB of 16 GB total).
- Compiled graphs cache in `.cache\`; delete to force a rebuild. `.tmp\` is
  gitignored scratch.
- Voice-loop audio I/O uses PortAudio; only `Windows WDM-KS` input devices are
  offered for lowest latency.
