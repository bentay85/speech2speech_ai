### Speech To Speech AI


https://github.com/user-attachments/assets/43b6f3fd-f94d-4d68-b487-b63e0ba47c4a


This script runs a speech to speech AI assistant with voice and webcam / desktop screenshot input. 

Script tested on Windows 11 with: 

1. Nvidia Graphics Card with 8GB of VRAM and latest drivers installed. You can run with less VRAM if you chose a smaller quantised GGUF model, or even on CPU if you do not mind the latency. 
2. The [uv package manger](https://github.com/astral-sh/uv) installed and added to PATH
3. [llama.cpp](https://github.com/ggml-org/llama.cpp) inference engine to run the Large Language Model. Download the latest Windows release with the CUDA DLLs from the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) page and place in the bin folder.  
4. A webcam, a mic and speakers. 

### Models

1. Silero Voice Activity Detection
Silero is a lightweight voice activity detection model. It is used to detect when you have started and stopped speaking. A clip of your speech will then be sent to the large language model. Download silero\_vad.onnx from the [silero-vad GitHub](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/data/silero_vad.onnx) page and place in the models\\silero folder.

2. Gemma 4 E4B is a multi-modal (text, image, audio) Large Language Model released by Google. Download gemma-4-E4B-it-Q4\_K\_M.gguf and mmproj-F16.gguf from the [unsloth huggingface](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/tree/main) page and place in the models\\gemma-4-E4B-it-gguf folder.

3. Piper Text to Speech is a lightweight text to speech engine. Download en\_US-lessac-medium.onnx and en\_US-lessac-medium.onnx.json from the [Rhasspy Huggingface](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium) page and place in models\\piper\_tts folder.  

### Design Decisions  

1. All models are run locally on your PC for privacy reasons. This also decreases AI response latency if the PC hardware is capable enough to run the models.   

2. Silero VAD and Piper TTS were run on the CPU to reserve the VRAM capacity for the Large Language model. 

3. E4B was chosen as it is the largest model that has native speech input and the quantised gguf can fit within 8GB of VRAM.

4. Silero VAD was suppressed when the AI is speaking as I have an open mic and speakers setup. This means that you and the AI takes turns to speak. 

5. Thinking for Gemma 4 was turned off to reduce the latency between the end of your speech and the AI's speech. This can be changed in the llama.cpp launch command. 

6. An image is taken of the webcam / desktop only after you have finished speaking. Your voice and the image is then sent to the large language model for processing. This reduces the amount of multi modal data that the large language model has to process before providing a response. 

### Running the AI Voice Assistant

Clone the repository and use uv to setup the virtual environment

```
git clone https://github.com/bentay85/speech2speech_ai.git
cd speech2speech_ai
uv sync
```
Download all the models and ensure that all the required files are in their respective folders.
```
speech2speech_ai/  
├── bin  
│   └── Place llama.cpp release files here  
├── main.py  
└── models  
   ├── gemma-4-E4B-it-gguf  
   │   ├── gemma-4-E4B-it-Q4\_K_M.gguf  
   │   └── mmproj-F16.gguf  
   ├── piper_tts  
   │   ├── en\_US-lessac-medium.onnx  
   │   └── en\_US-lessac-medium.onnx.json  
   └── silero  
       └── silero_vad.onnx  
```
You will need to have 2 command prompts open, one to run the llama.cpp LLM server and the second to run the script. Run the llama.cpp server in the first command prompt.  
```
.\bin\llama-server.exe ^
 -m .\models\gemma-4-E4B-it-gguf\gemma-4-E4B-it-Q4_K_M.gguf ^
 --mmproj .\models\gemma-4-E4B-it-gguf\mmproj-F16.gguf ^
 -ngl 99 ^
 -c 32768 ^
 -b 1024 ^
 -ub 1024 ^
 --reasoning off ^
 --reasoning-budget 0 ^
 --host 0.0.0.0 ^
 --port 8080
```  
Run the script in the second command prompt.  
```  
uv run main.py
```  
You will see a preview screen appear with your webcam feed. While on the preview window,   
1. Use the Tab key to toggle between webcam and desktop screen capture.   
2. Use the q key to quit.     

The desktop screen capture is set by default to 1920 x 1080, 150 pixels from the top of the screen. This captures 1/4 of my 4k desktop. I skip the top 150 pixels as this is where my Chrome address and bookmark bars are. You can change this behaviour in  `main.py`. Interaction with the AI should be just via speech.  
