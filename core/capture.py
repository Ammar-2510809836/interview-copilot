import asyncio
import logging
import pyaudiowpatch as pyaudio

logger = logging.getLogger(__name__)

class CaptureEngine:
    """
    Handles dual-channel audio capture (System WASAPI + Mic).
    Captures Windows system audio (Channel A - Interviewer) and Mic (Channel B - User).
    """
    def __init__(self):
        self.p = None
        self.system_stream = None
        self.mic_stream = None
        self.is_running = False

    def _get_default_wasapi_loopback(self):
        try:
            wasapi_info = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            logger.error("WASAPI not available on this system.")
            return None

        default_speakers = self.p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        
        if not default_speakers["isLoopbackDevice"]:
            for loopback in self.p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                logger.error("Default WASAPI loopback device not found.")
                return None
                
        return default_speakers

    def start_capture(self, audio_queue: asyncio.Queue):
        """
        Starts audio streams and pushes audio chunks to the queue with speaker tags.
        Pushes tuples of: (tag, raw_audio_data, sample_rate, channels)
        """
        self.p = pyaudio.PyAudio()
        self.is_running = True
        loop = asyncio.get_running_loop()

        wasapi_loopback = self._get_default_wasapi_loopback()
        if not wasapi_loopback:
            logger.error("Failed to find WASAPI loopback device. Cannot capture system audio.")
            return

        sys_rate = int(wasapi_loopback["defaultSampleRate"])
        sys_channels = wasapi_loopback["maxInputChannels"]

        # Callback for system audio (Interviewer)
        def system_callback(in_data, frame_count, time_info, status):
            if self.is_running and in_data:
                # Push to asyncio queue from pyaudio callback thread
                loop.call_soon_threadsafe(
                    audio_queue.put_nowait, 
                    ("[INTERVIEWER]", in_data, sys_rate, sys_channels)
                )
            return (None, pyaudio.paContinue)

        try:
            self.system_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=sys_channels,
                rate=sys_rate,
                input=True,
                input_device_index=wasapi_loopback["index"],
                stream_callback=system_callback
            )
            logger.info(f"System capture started: {wasapi_loopback['name']} ({sys_rate}Hz, {sys_channels}ch)")
        except Exception as e:
            logger.error(f"Failed to start system audio stream: {e}")

        # Callback for mic audio (User)
        mic_rate = 16000
        mic_channels = 1
        def mic_callback(in_data, frame_count, time_info, status):
            if self.is_running and in_data:
                loop.call_soon_threadsafe(
                    audio_queue.put_nowait, 
                    ("[ME]", in_data, mic_rate, mic_channels)
                )
            return (None, pyaudio.paContinue)

        try:
            default_mic = self.p.get_default_input_device_info()
            self.mic_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=mic_channels,
                rate=mic_rate,
                input=True,
                input_device_index=default_mic["index"],
                stream_callback=mic_callback
            )
            logger.info(f"Mic capture started on default device ({mic_rate}Hz, {mic_channels}ch)")
        except Exception as e:
            logger.error(f"Failed to start mic audio stream: {e}")

    def stop_capture(self):
        """
        Safely stops and closes audio streams.
        """
        self.is_running = False
        
        if self.system_stream:
            self.system_stream.stop_stream()
            self.system_stream.close()
            self.system_stream = None
            
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()
            self.mic_stream = None
            
        if self.p:
            self.p.terminate()
            self.p = None
            
        logger.info("Audio capture stopped.")
