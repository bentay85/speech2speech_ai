<iframe 
  src="https://www.youtube.com/embed/YlkMih_hSKc" 
  title="YouTube video player"
  style="width: 100%; aspect-ratio: 16 / 9; border: 0;" 
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
  allowfullscreen
  loading="lazy">
</iframe>
## Introduction
This project implements a pipecat pipeline that runs a speech to speech AI with a vision component. The AI can see your webcam feed and interact based on the image taken right after you end your speech. It runs fully in Windows with [UV](https://github.com/astral-sh/uv/releases) and [LM Studio](https://lmstudio.ai/) installed.  
## Project Folder
Clone the repository and change directory into the project folder.

Download the Piper TTS and Qwen3-VL models. Details in the next section.

Run LM Studio and load the Qwen3-VL model Enable developer mode and run the local server.  
```
git clone https://github.com/bentay85/speech2speech_ai.git
cd speech2speech_ai
uv sync
uv pip uninstall onnxruntime
uv run main.py
```
Note: Uninstall onnxruntime to ensure that we only use onnxruntim-gpu, so that the onnx models run on the GPU.
## Models
Speech to Text (Whisper): [link](https://huggingface.co/openai/whisper-base.en)  
Automatically downloaded to Huggingface Hub folder

Voice Activity Detection (Silero): [link](https://github.com/snakers4/silero-vad)  
Automatically downloaded

Vision Language Model(Qwen3-VL-2B-Instruct): [link](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/tree/main)  
Download Qwen3-VL-2B-Instruct-UD-Q4_K_XL.gguf and mmproj-BF16.gguf, to your LM Studio models folder\unsloth\Qwen3-VL-2B-Instruct-GGUF. If you have 8GB VRAM, you should be able to unload all layers to the GPU and have 16384 context.  

Text to Speech (Piper): [link](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium)  
Download en_US-lessac-medium.onnx and en_US-lessac-medium.onnx.json and place them the project folder  
## Blog Post
[https://bentay85.bearblog.dev/multimodal_voice_agent/](https://bentay85.bearblog.dev/multimodal_voice_agent/)