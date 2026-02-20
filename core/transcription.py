import asyncio
import logging
import os
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)

logger = logging.getLogger(__name__)

class TranscriptionEngine:
    """
    Handles asynchronous Speech-To-Text via Deepgram Cloud API.
    """
    def __init__(self):
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            logger.warning("DEEPGRAM_API_KEY not found in environment!")
            
        self.dg_client = DeepgramClient(self.api_key) if self.api_key else None
        self.connections = {}
        self.text_queue = None

    async def _setup_connection(self, tag: str, rate: int, channels: int):
        if not self.dg_client:
            logger.error(f"Cannot setup WS for {tag} - No Deepgram API Key.")
            return None
            
        try:
            dg_connection = self.dg_client.listen.asyncwebsocket.v("1")
            
            # self inside the callback is the dg_connection in SDK 4.x
            engine_self = self

            async def on_message(self, result, **kwargs):
                try:
                    # In SDK 4.8.x and 5.x, result is an object of LiveResultResponse.
                    if getattr(result, "channel", None) is None:
                        return
                    
                    if not getattr(result.channel, "alternatives", None):
                        return
                        
                    sentence = result.channel.alternatives[0].transcript
                    is_final = getattr(result, "is_final", False)

                    if not sentence:
                        return

                    if is_final:
                        if engine_self.text_queue:
                            await engine_self.text_queue.put((tag, sentence))
                        logger.info(f"{tag}: {sentence}")
                except Exception as e:
                    logger.error(f"Error parsing Deepgram message: {e}")

            async def on_error(self, error, **kwargs):
                logger.error(f"Deepgram error on {tag}: {error}")
                
            dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
            dg_connection.on(LiveTranscriptionEvents.Error, on_error)

            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                encoding="linear16",
                channels=channels,
                sample_rate=rate,
                interim_results=False,
                endpointing=300
            )
            
            if await dg_connection.start(options) is False:
                logger.error(f"Failed to start Deepgram WS for {tag}")
                return None
                
            logger.info(f"Deepgram WS connected for {tag} ({rate}Hz, {channels}ch)")
            return dg_connection
            
        except Exception as e:
            logger.error(f"Exception connecting Deepgram for {tag}: {e}")
            return None

    async def transcribe_stream(self, audio_queue: asyncio.Queue, text_queue: asyncio.Queue):
        """
        Consumes audio from queue, sends to API, tags with [INTERVIEWER]/[ME], 
        and pushes to the text queue.
        """
        self.text_queue = text_queue
        logger.info("TranscriptionEngine listening to audio queue...")
        
        # Async helper to send KeepAlive
        async def keep_alive(tag, conn):
            while True:
                await asyncio.sleep(8)
                try:
                    # In SDK 4.x, send empty string or json keepalive
                    await conn.send('{"type": "KeepAlive"}')
                except Exception:
                    break

        self.keep_alive_tasks = {}

        try:
            while True:
                tag, in_data, rate, channels = await audio_queue.get()
                
                # Lazy initialization of WS connection based on exact rate/channels
                if tag not in self.connections:
                    conn = await self._setup_connection(tag, rate, channels)
                    self.connections[tag] = conn
                    if conn:
                        self.keep_alive_tasks[tag] = asyncio.create_task(keep_alive(tag, conn))
                        
                conn = self.connections[tag]
                if conn:
                    try:
                        await conn.send(in_data)
                    except Exception as e:
                        logger.error(f"Failed to send data to {tag} WS: {e}")
                        
        except asyncio.CancelledError:
            logger.info("TranscriptionEngine stream cancelled.")
        finally:
            for task in self.keep_alive_tasks.values():
                task.cancel()
            for tag, conn in self.connections.items():
                if conn:
                    await conn.finish()
            self.connections.clear()
