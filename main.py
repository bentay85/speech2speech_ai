import pyaudio
import numpy as np
import wave
import time
import cv2
import queue
import threading
import re
import json
import base64
import requests
import io
import onnxruntime as ort
from datetime import datetime
from piper import PiperVoice
from mss import MSS

# --- Configuration ---
VAD_MODEL_PATH = 'models/silero/silero_vad.onnx'
PIPER_MODEL_PATH = "models/piper_tts/en_US-lessac-medium.onnx"
LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

SAMPLING_RATE = 16000
CHUNK_SIZE = 512
CONTEXT_SIZE = 64
START_THRESHOLD = 0.4   
STOP_THRESHOLD = 0.1    
SILENCE_LIMIT = 0.4     
PRE_ROLL_CHUNKS = 15    
DEBUG_MODE = True       

# Camera Settings
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

# Screen Capture Settings
# On a 4k desktop resolution, capturing 1080p which is 1/4 of the screen. 
# skipping the top 150 pixels as this is normally the search bar of my browser. 
MONITOR = {"top": 150, "left": 0, "width": 1920, "height": 1080} 

# --- System Prompt ---
SYSTEM_PROMPT = (
    "You are a helpful AI Assistant. Keep your answers concise and conversational. Refer to yourself as I and me as you. The image is either a view from my webcam or my desktop screen. "
)

# --- Shared State ---
audio_queue = queue.Queue()
tts_queue = queue.Queue() 
state = {
    'is_recording': False,
    'is_inferring': False,
    'is_speaking': False, 
    'current_confidence': 0.0,
    'flash_trigger': False,
    'running': True,
    'last_frame': None, 
    'view_mode': 'webcam', 
    'history': [{"role": "system", "content": SYSTEM_PROMPT}]
}

# --- Utils ---
def get_ts():
    return f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}]"

def log(message):
    print(f"{get_ts()} {message}")

# --- Initialization ---
log("Loading Silero VAD...")
vad_session = ort.InferenceSession(VAD_MODEL_PATH, providers=['CPUExecutionProvider'])

log("Loading Piper TTS...")
piper_voice = PiperVoice.load(PIPER_MODEL_PATH, use_cuda=False)

def tts_worker():
    """Piper TTS synthesis with correct attribute handling."""
    while state['running']:
        try:
            text = tts_queue.get(timeout=0.1)
            state['is_speaking'] = True
            try:
                for chunk in piper_voice.synthesize(text):
                    if hasattr(chunk, 'audio_int16_bytes'):
                        data = chunk.audio_int16_bytes
                    elif hasattr(chunk, 'audio'):
                        data = chunk.audio
                    else:
                        data = bytes(chunk)
                    
                    tts_stream.write(data)
            except Exception as e:
                log(f"[!] TTS Playback Error: {e}")
        except queue.Empty:
            if not state['is_inferring']:
                state['is_speaking'] = False

def run_gemma_inference(image_np, audio_bytes):
    """Multimodal inference via local llama.cpp server."""
    state['is_inferring'] = True
    try:
        # 1. Encode image to JPEG base64
        ret, buffer = cv2.imencode('.jpg', image_np)
        b64_img = base64.b64encode(buffer).decode('utf-8')
        img_url = f"data:image/jpeg;base64,{b64_img}"

        # 2. Encode audio to base64 directly from memory bytes
        b64_audio = base64.b64encode(audio_bytes).decode('utf-8')

        # 3. Build message payload for API
        messages = [state['history'][0]]
        ai_responses = [turn for turn in state['history'] if turn['role'] == 'assistant']
        messages.extend(ai_responses[-5:])
        
        current_content = [
            {"type": "image_url", "image_url": {"url": img_url}},
            {"type": "input_audio", "input_audio": {"data": b64_audio, "format": "wav"}}
        ]
        messages.append({"role": "user", "content": current_content})

        payload = {
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "top_p": 0.9
        }

        # 4. Stream response from server
        response = requests.post(LLAMA_SERVER_URL, json=payload, stream=True, timeout=60)
        response.raise_for_status()

        print(f"\n{get_ts()} [AI]: ", end="", flush=True)
        full_response = ""
        unspoken_buffer = ""
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        if 'choices' in data_json and len(data_json['choices']) > 0:
                            delta = data_json['choices'][0].get('delta', {})
                            
                            # Handle specific reasoning fields (like DeepSeek-R1 API structure)
                            reasoning_chunk = delta.get('reasoning_content', '')
                            if reasoning_chunk:
                                print(reasoning_chunk, end="", flush=True)
                                full_response += reasoning_chunk
                                continue # Skip adding reasoning text to the TTS buffer
                                
                            chunk = delta.get('content', '')
                            if chunk:
                                full_response += chunk
                                print(chunk, end="", flush=True)
                                unspoken_buffer += chunk
                                
                                # 1. Clean out completed <think>...</think> blocks from the TTS buffer
                                while '<think>' in unspoken_buffer and '</think>' in unspoken_buffer:
                                    start = unspoken_buffer.find('<think>')
                                    end = unspoken_buffer.find('</think>') + len('</think>')
                                    unspoken_buffer = unspoken_buffer[:start] + unspoken_buffer[end:]
                                    
                                # 2. If we are currently inside an unclosed <think> tag, wait for it to close
                                if '<think>' in unspoken_buffer:
                                    continue
                                
                                # 3. Chunking for TTS
                                parts = re.split(r'(?<=[,.!?\n])\s+', unspoken_buffer)
                                if len(parts) > 1:
                                    # Keep the last incomplete sentence in the buffer
                                    unspoken_buffer = parts.pop()
                                    for part in parts:
                                        clean_text = part.replace('*', '').strip()
                                        # Remove any stray unclosed tags just in case
                                        clean_text = re.sub(r'<[^>]+>', '', clean_text)
                                        if clean_text:
                                            tts_queue.put(clean_text)
                    except json.JSONDecodeError:
                        pass
                        
        # Flush any remaining text in the unspoken buffer
        if '<think>' in unspoken_buffer:
            # If the response abruptly ends during a thought, chop the thought off
            unspoken_buffer = unspoken_buffer[:unspoken_buffer.find('<think>')]
            
        final_clean = unspoken_buffer.replace('*', '').strip()
        final_clean = re.sub(r'<[^>]+>', '', final_clean)
        if final_clean:
            tts_queue.put(final_clean)
            
        print("\n")
        state['history'].append({"role": "assistant", "content": full_response})

    except requests.exceptions.RequestException as req_err:
        log(f"[!] Server Error: {req_err}")
        log(f"Ensure your llama-server is running and accessible at {LLAMA_SERVER_URL}")
    except Exception as e:
        log(f"[!] Inference Error: {e}")
    finally:
        state['is_inferring'] = False

def audio_processor():
    recorded_frames = []
    pre_roll_buffer = []
    silence_start_time = None
    
    vad_state = np.zeros((2, 1, 128), dtype=np.float32)
    context_buffer = np.zeros(CONTEXT_SIZE, dtype=np.float32)

    while state['running']:
        try:
            data = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        audio_int16 = np.frombuffer(data, np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        input_tensor = np.concatenate([context_buffer, audio_float32]).reshape(1, -1)

        out = vad_session.run(None, {
            "input": input_tensor,
            "sr": np.array(SAMPLING_RATE, dtype=np.int64),
            "state": vad_state
        })
        
        confidence = out[0].item()
        vad_state = out[1] 
        
        context_buffer = audio_float32[-CONTEXT_SIZE:].copy()
        
        state['current_confidence'] = confidence

        if not state['is_recording']:
            pre_roll_buffer.append(data)
            if len(pre_roll_buffer) > PRE_ROLL_CHUNKS:
                pre_roll_buffer.pop(0)
            
            if confidence > START_THRESHOLD and not state['is_inferring'] and not state['is_speaking']:
                state['is_recording'] = True
                recorded_frames = list(pre_roll_buffer)
                log("Voice detected, listening...")
        else:
            recorded_frames.append(data)
            if confidence < STOP_THRESHOLD:
                if silence_start_time is None:
                    silence_start_time = time.time()
                
                if (time.time() - silence_start_time) > SILENCE_LIMIT:
                    log("End of speech detected. Processing capture...")
                    state['is_recording'] = False
                    silence_start_time = None
                    
                    # Store to in-memory bytes buffer instead of disk
                    wav_io = io.BytesIO()
                    wf = wave.open(wav_io, 'wb')
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLING_RATE)
                    wf.writeframes(b''.join(recorded_frames))
                    wf.close()
                    audio_bytes = wav_io.getvalue()
                    
                    if state['last_frame'] is not None:
                        current_img = state['last_frame'].copy()
                        state['flash_trigger'] = True
                        threading.Thread(target=run_gemma_inference, args=(current_img, audio_bytes)).start()
                    recorded_frames = []
            else:
                silence_start_time = None

def audio_callback(in_data, frame_count, time_info, status):
    if not state['is_inferring'] and not state['is_speaking']:
        audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

# --- Start Hardware ---
p = pyaudio.PyAudio()
audio_stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE,
                input=True, frames_per_buffer=CHUNK_SIZE, stream_callback=audio_callback)
tts_stream = p.open(format=pyaudio.paInt16, channels=1, rate=piper_voice.config.sample_rate,
                output=True)

threading.Thread(target=audio_processor, daemon=True).start()
threading.Thread(target=tts_worker, daemon=True).start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

sct = MSS()
flash_until = 0
log("--- System Active (Press 'q' to exit, 'Tab' to toggle view) ---")

try:
    while True:
        ret, cam_frame = cap.read()
        if not ret: break

        # Select the active view
        if state['view_mode'] == 'webcam':
            current_frame = cam_frame
        else:
            # Desktop Capture Logic (1280x960 capture)
            sct_img = sct.grab(MONITOR)
            desktop_frame = np.array(sct_img)
            desktop_frame = cv2.cvtColor(desktop_frame, cv2.COLOR_BGRA2BGR)
            
            # Resize desktop frame to match webcam dimensions (640x480)
            current_frame = cv2.resize(desktop_frame, (TARGET_WIDTH, TARGET_HEIGHT))

        # We are only saving/showing the current_frame now
        state['last_frame'] = current_frame.copy()
        display_frame = current_frame.copy()
        h, w = display_frame.shape[:2]

        if state['flash_trigger']:
            flash_until = time.time() + 0.4
            state['flash_trigger'] = False

        if time.time() < flash_until:
            cv2.rectangle(display_frame, (0, 0), (w, h), (255, 255, 255), 12)

        # Draw State Labels
        if state['is_recording']:
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display_frame, "LISTENING", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif state['is_speaking']:
            cv2.putText(display_frame, "SPEAKING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        elif state['is_inferring']:
            cv2.putText(display_frame, "THINKING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        # Draw View Mode Indicator
        mode_text = "WEBCAM" if state['view_mode'] == 'webcam' else "DESKTOP"
        cv2.putText(display_frame, f"[{mode_text}]", (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # --- Audio Level Meter (The Bar) ---
        meter_x = 20
        meter_y = h - 30
        meter_w = int(state['current_confidence'] * 150)
        color = (0, 255, 0) if state['current_confidence'] > START_THRESHOLD else (200, 200, 200)
        
        # Background bar
        cv2.rectangle(display_frame, (meter_x, meter_y), (meter_x + 150, meter_y + 10), (50, 50, 50), -1)
        # Active level bar
        cv2.rectangle(display_frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + 10), color, -1)
        cv2.putText(display_frame, "MIC", (meter_x, meter_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Gemma-4 Vision/Voice Assistant', display_frame)
        
        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 9: # ASCII 9 is the Tab key
            state['view_mode'] = 'desktop' if state['view_mode'] == 'webcam' else 'webcam'
            log(f"Switched view mode to: {state['view_mode'].upper()}")

finally:
    log("Shutting down...")
    state['running'] = False
    sct.close()
    audio_stream.stop_stream()
    audio_stream.close()
    tts_stream.stop_stream()
    tts_stream.close()
    p.terminate()
    cap.release()
    cv2.destroyAllWindows()