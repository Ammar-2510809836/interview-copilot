import asyncio
import logging
import os
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)

logger = logging.getLogger(__name__)

# How many times to retry a failed WS connection before giving up
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY_SECONDS = 3


class TranscriptionEngine:
    """
    Handles asynchronous Speech-To-Text via Deepgram Cloud API.
    Includes automatic WebSocket reconnection logic to survive network drops
    or Deepgram disconnects mid-interview.
    """
    def __init__(self):
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            logger.warning("DEEPGRAM_API_KEY not found in environment!")

        self.dg_client = DeepgramClient(self.api_key) if self.api_key else None
        self.connections = {}
        self.connection_meta = {}  # tag -> (rate, channels) for reconnect
        self.text_queue = None
        self.keep_alive_tasks = {}
        self._closed = False

    async def _setup_connection(self, tag: str, rate: int, channels: int):
        if not self.dg_client:
            logger.error(f"Cannot setup WS for {tag} - No Deepgram API Key.")
            return None

        try:
            dg_connection = self.dg_client.listen.asyncwebsocket.v("1")

            engine_self = self

            async def on_message(self, result, **kwargs):
                try:
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
                            try:
                                engine_self.text_queue.put_nowait((tag, sentence))
                            except asyncio.QueueFull:
                                logger.warning(f"text_queue full — dropping transcript from {tag}")
                        logger.info(f"{tag}: {sentence}")
                except Exception as e:
                    logger.error(f"Error parsing Deepgram message: {e}")

            async def on_error(self, error, **kwargs):
                logger.error(f"Deepgram WS error on {tag}: {error}")

            async def on_close(self, close_event, **kwargs):
                """Called when Deepgram closes the WebSocket. Trigger reconnect."""
                if not engine_self._closed:
                    logger.warning(f"Deepgram WS closed for {tag}. Scheduling reconnect...")
                    asyncio.create_task(engine_self._reconnect(tag))

            dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
            dg_connection.on(LiveTranscriptionEvents.Error, on_error)
            dg_connection.on(LiveTranscriptionEvents.Close, on_close)

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

    async def _reconnect(self, tag: str):
        """Automatically reconnect a dropped WebSocket for the given tag."""
        if tag not in self.connection_meta:
            logger.error(f"Cannot reconnect {tag}: no metadata stored.")
            return

        rate, channels = self.connection_meta[tag]

        for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
            if self._closed:
                return
            logger.info(f"Reconnecting {tag} (attempt {attempt}/{MAX_RECONNECT_ATTEMPTS})...")
            await asyncio.sleep(RECONNECT_DELAY_SECONDS)

            conn = await self._setup_connection(tag, rate, channels)
            if conn:
                self.connections[tag] = conn
                # Restart keep-alive for new connection
                if tag in self.keep_alive_tasks:
                    self.keep_alive_tasks[tag].cancel()
                self.keep_alive_tasks[tag] = asyncio.create_task(self._keep_alive(tag, conn))
                logger.info(f"Reconnected {tag} successfully.")
                return

        logger.error(f"All {MAX_RECONNECT_ATTEMPTS} reconnect attempts failed for {tag}. Giving up.")

    async def _keep_alive(self, tag: str, conn):
        """Send periodic KeepAlive pings to prevent Deepgram from timing out."""
        while True:
            await asyncio.sleep(8)
            try:
                await conn.send('{"type": "KeepAlive"}')
            except Exception:
                logger.warning(f"KeepAlive failed for {tag} — connection may be dead.")
                break

    async def transcribe_stream(self, audio_queue: asyncio.Queue, text_queue: asyncio.Queue):
        """
        Consumes audio from queue, sends to Deepgram API, tags with [INTERVIEWER]/[ME],
        and pushes transcripts to the text queue.
        Auto-reconnects on WebSocket failure.
        """
        self.text_queue = text_queue
        self._closed = False
        logger.info("TranscriptionEngine listening to audio queue...")

        try:
            while True:
                tag, in_data, rate, channels = await audio_queue.get()

                # Lazy init — store meta for reconnect
                if tag not in self.connections:
                    conn = await self._setup_connection(tag, rate, channels)
                    self.connections[tag] = conn
                    self.connection_meta[tag] = (rate, channels)
                    if conn:
                        self.keep_alive_tasks[tag] = asyncio.create_task(
                            self._keep_alive(tag, conn)
                        )

                conn = self.connections.get(tag)
                if conn:
                    try:
                        await conn.send(in_data)
                    except Exception as e:
                        logger.error(f"Failed to send audio to {tag} WS: {e}. Will reconnect.")
                        self.connections[tag] = None
                        asyncio.create_task(self._reconnect(tag))

        except asyncio.CancelledError:
            logger.info("TranscriptionEngine stream cancelled.")
        finally:
            self._closed = True
            for task in self.keep_alive_tasks.values():
                task.cancel()
            for tag, conn in self.connections.items():
                if conn:
                    try:
                        await conn.finish()
                    except Exception:
                        pass
            self.connections.clear()
            self.connection_meta.clear()
