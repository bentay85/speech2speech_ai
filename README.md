## Models
Speech to Text (Whisper): Automatically downloaded to Huggingface Hub Folder (C:\Users\<YourUsername>\.cache\huggingface\hub\) 
[https://huggingface.co/openai/whisper-base.en](https://huggingface.co/openai/whisper-base.en)

Voice Activity Detection (Silero): Automatically downloaded
[https://github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad)

Vision Language Model: Download Qwen3-VL-2B-Instruct-UD-Q4_K_XL.gguf and mmproj-BF16.gguf, to your LM Studio models folder\unsloth\Qwen3-VL-2B-Instruct-GGUF. 
Load into LM Studio with all layers offloaded to GPU and 16384 context. Adjust context according to your GPU VRAM. Enable developer model and run the local server. 
[https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/tree/main](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/tree/main)

Text to Speech (Piper): Download en_US-lessac-medium.onnx and en_US-lessac-medium.onnx.json and place them the project folder
[https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium)

## Project Folder
Clone the repository and change directory into the project folder.

We uninstall onnxruntime to ensure that we only use onnxruntim-gpu, so that the onnx models run on the GPU.
 
```
uv sync
uv pip uninstall onnxruntime
uv run main.py
```
## Blog Post
[https://bentay85.bearblog.dev/multimodal_voice_agent/](https://bentay85.bearblog.dev/multimodal_voice_agent/)
