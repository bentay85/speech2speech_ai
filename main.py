import asyncio
import cv2
import base64

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame, 
    BotStartedSpeakingFrame, 
    BotStoppedSpeakingFrame, 
    UserStoppedSpeakingFrame,
    TTSSpeakFrame
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContext,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.runner import PipelineRunner

class VisionCaptureProcessor(FrameProcessor):
    def __init__(self, context: LLMContext, camera_index=0):
        super().__init__()
        self._context = context
        self._cap = cv2.VideoCapture(camera_index)
        self._window_name = "Pipecat Vision Preview"
        self._window_initialized = False  # Track if window is open

        if not self._cap.isOpened(): 
            print(f"Warning: Could not open camera at index {camera_index}")
        else:
            # Set native resolution to 432x240
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 432)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    def _get_base64_image(self):
        ret, frame = self._cap.read()
        if not ret:
            return None
        
        # Flip the image horizontally (mirror effect)
        #frame = cv2.flip(frame, 1)

        # Lazy-initialize the window only when the first frame is ready
        if not self._window_initialized:
            cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
            self._window_initialized = True
        
        # Display the image
        cv2.imshow(self._window_name, frame)
        cv2.waitKey(1)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(buffer).decode('utf-8')

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame):
            print("DEBUG: 📸 User stopped speaking. Capturing and showing window...")
            b64_image = self._get_base64_image()
            
            if b64_image:
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                }
                
                for msg in reversed(self._context.messages):
                    if msg["role"] == "user":
                        if isinstance(msg["content"], str):
                            msg["content"] = [
                                {"type": "text", "text": msg["content"]},
                                image_content
                            ]
                        elif isinstance(msg["content"], list):
                            msg["content"].append(image_content)
                        
                        print(f"DEBUG: ✅ 432x240 image attached to Context.")
                        break 

        await self.push_frame(frame, direction)

    def __del__(self):
        if hasattr(self, '_cap') and self._cap.isOpened():
            self._cap.release()
        # Only attempt to destroy if we actually created it
        if hasattr(self, '_window_initialized') and self._window_initialized:
            cv2.destroyAllWindows()

async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        )
    )
    
    vad_analyzer = SileroVADAnalyzer(params=VADParams(min_volume=0.05, confidence=0.7, stop_secs=0.2))
    stt = WhisperSTTService(model_path="base.en", device="cuda")
    
    # Define the initial conversation context
    messages = [
        {"role": "system", "content": (
            "You are an AI assistant in a video call with me. Refer to me as 'you' and yourself as 'I'. Reply in 1-2 short sentences. No lists or bullet points."
        )},
    ]
    context = LLMContext(messages)

    # Initialize vision processor with the shared context
    vision_processor = VisionCaptureProcessor(context=context, camera_index=0)
    
    llm = OpenAILLMService(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        settings=OpenAILLMService.Settings(
            model="qwen-3-vl-2b-instruct",
            max_tokens=80,
        )
    )

    tts = PiperTTSService(
        model_path="en_US-lessac-medium.onnx", 
        config_path="en_US-lessac-medium.onnx.json",
        settings=PiperTTSService.Settings(
            voice="en_US-lessac-medium" 
        )
    )
    
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyzer,
            # Set this to a low value to eliminate that long wait.
            # 0.5s is a "sweet spot" for natural conversation.
            user_turn_stop_timeout=0.25,
            user_mute_strategies=[AlwaysUserMuteStrategy()]
        )
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        vision_processor, # Catching the UserStoppedSpeakingFrame here
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @task.event_handler("on_pipeline_started")
    async def on_pipeline_started(task, frame):
        await asyncio.sleep(0.8)
        await task.queue_frames([TTSSpeakFrame(text="Hello! How can I help you today?")])

    print("\n--- ᓚᘏᗢ Local Bot with Piper TTS & Vision ---")
    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())