@echo off
rem ===========================================================================
rem  SoulX FlashHead AI Avatar - one-shot setup for Windows.
rem  Creates the venv, installs all Python packages, downloads every model
rem  into models\, fetches the llama-server binaries into bin\, and installs
rem  eSpeak NG (required by Kokoro TTS). Re-runnable: existing work is skipped.
rem
rem  Needs: Windows 10+, an NVIDIA GPU with recent drivers, and an internet
rem  connection. Run from an elevated prompt ONLY if eSpeak NG should install
rem  silently; otherwise it prints manual steps instead.
rem ===========================================================================
setlocal
cd /d "%~dp0"

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"
set "HF=%VENV%\Scripts\hf.exe"

rem Mainland China: uncomment to download models from the hf-mirror.com mirror.
rem set "HF_ENDPOINT=https://hf-mirror.com"

echo ======================================================================
echo  SoulX FlashHead setup
echo ======================================================================
echo.

mkdir ".tmp" 2>nul

rem --------------------------------------------------------------- 1. venv
if exist "%PY%" (
    echo [venv] Found existing %VENV%, skipping creation.
    goto :packages
)
where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [venv] Creating %VENV% with uv, Python 3.10...
    uv venv "%VENV%" --python 3.10
    if errorlevel 1 goto :fail_venv
) else (
    echo [venv] Creating %VENV% with the Python launcher...
    py -3.10 -m venv "%VENV%"
    if errorlevel 1 python -m venv "%VENV%"
    if errorlevel 1 goto :fail_venv
)

:packages
echo.
echo [deps] Installing Python packages from requirements.txt...
echo        ^(torch+CUDA, PySide6, kokoro, silero-vad, ... - several GB on first run^)
where uv >nul 2>nul
if %errorlevel% equ 0 (
    uv pip install --python "%PY%" -r requirements.txt
    if errorlevel 1 goto :fail_deps
) else (
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :fail_deps
)

rem hf CLI into the venv (harmless if already present).
if not exist "%HF%" (
    where uv >nul 2>nul
    if errorlevel 1 (
        "%PY%" -m pip install "huggingface_hub[cli]"
    ) else (
        uv pip install --python "%PY%" "huggingface_hub[cli]"
    )
    if errorlevel 1 goto :fail_deps
)

rem ------------------------------------------------------------- 2. models
echo.
echo [models] Checking model weights in models\...

rem --- SoulX FlashHead avatar ---
dir /b "models\SoulX-FlashHead-1_3B\*" >nul 2>nul
if %errorlevel% equ 0 (
    echo [skip] SoulX-FlashHead-1_3B already present.
) else (
    echo [get ] Soul-AILab/SoulX-FlashHead-1_3B ...
    "%HF%" download Soul-AILab/SoulX-FlashHead-1_3B --local-dir "models\SoulX-FlashHead-1_3B"
    if errorlevel 1 goto :fail_models
)

rem --- wav2vec2 (avatar audio frontend) ---
dir /b "models\wav2vec2-base-960h\*" >nul 2>nul
if %errorlevel% equ 0 (
    echo [skip] wav2vec2-base-960h already present.
) else (
    echo [get ] facebook/wav2vec2-base-960h ...
    "%HF%" download facebook/wav2vec2-base-960h --local-dir "models\wav2vec2-base-960h"
    if errorlevel 1 goto :fail_models
)

rem --- Gemma 4 E4B GGUF + audio projector ---
if exist "models\Gemma4-E4B\gemma-4-E4B-it-UD-Q4_K_XL.gguf" if exist "models\Gemma4-E4B\mmproj-BF16.gguf" (
    echo [skip] Gemma 4 E4B GGUF files already present.
) else (
    echo [get ] unsloth/gemma-4-E4B-it-GGUF ^(Q4_K_XL + mmproj, ~several GB^)...
    "%HF%" download unsloth/gemma-4-E4B-it-GGUF --include "gemma-4-E4B-it-UD-Q4_K_XL.gguf" --include "mmproj-BF16.gguf" --local-dir "models\Gemma4-E4B"
    if errorlevel 1 goto :fail_models
)

rem --- Kokoro TTS weights + all voice presets ---
if exist "models\Kokoro-82M\kokoro-v1_0.pth" if exist "models\Kokoro-82M\voices\*.pt" (
    echo [skip] Kokoro-82M weights and voices already present.
) else (
    echo [get ] hexgrad/Kokoro-82M ^(model + voices, ~several GB^)...
    "%HF%" download hexgrad/Kokoro-82M --include "config.json" --include "kokoro-v1_0.pth" --include "voices/*.pt" --local-dir "models\Kokoro-82M"
    if errorlevel 1 goto :fail_models
)

rem --- Silero VAD: shipped inside the silero-vad pip package, copy it out.
if exist "models\silero-vad\silero_vad.onnx" (
    echo [skip] silero_vad.onnx already present.
) else (
    if exist ".venv\Lib\site-packages\silero_vad\data\silero_vad.onnx" (
        echo [get ] Copying silero_vad.onnx from the silero-vad package...
        mkdir "models\silero-vad" 2>nul
        copy /y ".venv\Lib\site-packages\silero_vad\data\silero_vad.onnx" "models\silero-vad\" >nul
    ) else (
        echo [warn] silero_vad.onnx not found in site-packages; reinstall requirements.
    )
)

rem ------------------------------------------------------------ 3. llama.cpp
echo.
if exist "bin\llama-server.exe" (
    echo [skip] bin\llama-server.exe already present.
) else (
    echo [get ] llama.cpp b11748 CUDA 12.4 build for Windows x64...
    mkdir "bin" 2>nul
    curl -L -f -o ".tmp\llama-cuda.zip" "https://github.com/ggml-org/llama.cpp/releases/download/b11748/llama-b11748-bin-win-cuda-12.4-x64.zip"
    if errorlevel 1 goto :fail_llama
    tar -xf ".tmp\llama-cuda.zip" -C "bin"
    if errorlevel 1 goto :fail_llama
    del ".tmp\llama-cuda.zip"
)
if exist "bin\cudart64_12.dll" (
    echo [skip] CUDA runtime DLLs already in bin\.
) else (
    echo [get ] llama.cpp b11748 CUDA runtime DLLs...
    curl -L -f -o ".tmp\cudart.zip" "https://github.com/ggml-org/llama.cpp/releases/download/b11748/cudart-llama-bin-win-cuda-12.4-x64.zip"
    if errorlevel 1 (
        echo [warn] CUDA runtime download failed; copy cudart64_12.dll into bin\ manually.
    ) else (
        tar -xf ".tmp\cudart.zip" -C "bin"
        del ".tmp\cudart.zip"
    )
)

rem ---------------------------------------------------------- 4. eSpeak NG
echo.
where espeak-ng >nul 2>nul
if %errorlevel% equ 0 (
    echo [skip] eSpeak NG is on the PATH.
) else if exist "C:\Program Files\eSpeak NG\lib\espeak-ng.dll" (
    echo [skip] eSpeak NG is installed but not on the PATH ^(Kokoro finds it via the registry^).
) else (
    echo [get ] eSpeak NG 1.52.0 ^(required by Kokoro TTS^)...
    curl -L -f -o ".tmp\espeak-ng.msi" "https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng-1.52.0-x64.msi"
    if errorlevel 1 (
        echo [warn] Could not download eSpeak NG. Install it manually from:
        echo        https://github.com/espeak-ng/espeak-ng/releases
    ) else (
        msiexec /i ".tmp\espeak-ng.msi" /qn /norestart
        if errorlevel 1 (
            echo [warn] Silent install failed ^(needs administrator^). Re-run this script
            echo        from an elevated prompt, or open the MSI manually:
            echo        %cd%\.tmp\espeak-ng.msi
        ) else (
            echo [ok  ] eSpeak NG installed. Open a NEW terminal so the PATH applies.
        )
    )
)

echo.
echo ======================================================================
echo  Setup complete.
echo.
echo    Run the app:        run-app.bat    (or: .venv\Scripts\python.exe src\app.py)
echo    Debug logging:      set LOG_LEVEL=DEBUG before launching.
echo    Logs:               logs\app.log  (truncated on every start)
echo                        .tmp\llama-server.log (llama.cpp server output)
echo ======================================================================
pause
exit /b 0

:fail_venv
echo.
echo [FAIL] Could not create the virtual environment.
echo        Install Python 3.10 (https://www.python.org/downloads/) or uv
echo        (https://docs.astral.sh/uv/) and re-run this script.
pause
exit /b 1

:fail_deps
echo.
echo [FAIL] Package installation failed. Read the output above; common causes:
echo        - antivirus blocking the download
echo        - no internet / corporate proxy
echo        - Python version other than 3.10
pause
exit /b 1

:fail_models
echo.
echo [FAIL] Model download failed. Check your connection (or set the
echo        HF_ENDPOINT mirror line at the top of this file) and re-run:
echo        downloads resume where they left off.
pause
exit /b 1

:fail_llama
echo.
echo [FAIL] Could not download llama-server binaries. Download manually:
echo        https://github.com/ggml-org/llama.cpp/releases/tag/b11748
echo        Use the "Windows x64 (CUDA 12)" zip and unpack it into bin\.
pause
exit /b 1
