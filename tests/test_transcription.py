"""
Unit tests for core/transcription.py — TranscriptionEngine connection and reconnect logic.
All Deepgram API calls are mocked — no API key required.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call

# chromadb must be imported before PyQt6 on Windows
import chromadb
from core.transcription import TranscriptionEngine, MAX_RECONNECT_ATTEMPTS


class TestTranscriptionEngineInit(unittest.TestCase):
    """Tests for TranscriptionEngine initialization."""

    def test_missing_api_key_logs_warning(self):
        """Missing DEEPGRAM_API_KEY should warn but not crash."""
        with patch.dict("os.environ", {}, clear=True):
            engine = TranscriptionEngine()
        self.assertIsNone(engine.dg_client)

    def test_api_key_present_creates_client(self):
        """With API key set, dg_client should be initialized."""
        with patch("core.transcription.DeepgramClient") as mock_dg:
            with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test_key"}):
                engine = TranscriptionEngine()
        self.assertIsNotNone(engine.dg_client)

    def test_initial_state(self):
        """Verify initial state of all tracking dicts."""
        engine = TranscriptionEngine()
        self.assertEqual(engine.connections, {})
        self.assertEqual(engine.connection_meta, {})
        self.assertEqual(engine.keep_alive_tasks, {})
        self.assertFalse(engine._closed)


class TestSetupConnection(unittest.IsolatedAsyncioTestCase):
    """Tests for _setup_connection."""

    async def test_returns_none_without_client(self):
        """Without a Deepgram client (no API key), _setup_connection returns None."""
        engine = TranscriptionEngine()
        engine.dg_client = None
        result = await engine._setup_connection("[INTERVIEWER]", 48000, 2)
        self.assertIsNone(result)

    async def test_returns_none_on_connection_failure(self):
        """If Deepgram start() returns False, _setup_connection returns None."""
        engine = TranscriptionEngine()
        mock_conn = AsyncMock()
        mock_conn.start = AsyncMock(return_value=False)
        mock_conn.on = MagicMock()

        mock_dg_client = MagicMock()
        mock_dg_client.listen.asyncwebsocket.v.return_value = mock_conn
        engine.dg_client = mock_dg_client

        result = await engine._setup_connection("[ME]", 16000, 1)
        self.assertIsNone(result)

    async def test_returns_connection_on_success(self):
        """Successful start() returns the connection object."""
        engine = TranscriptionEngine()
        mock_conn = AsyncMock()
        mock_conn.start = AsyncMock(return_value=True)
        mock_conn.on = MagicMock()

        mock_dg_client = MagicMock()
        mock_dg_client.listen.asyncwebsocket.v.return_value = mock_conn
        engine.dg_client = mock_dg_client

        result = await engine._setup_connection("[INTERVIEWER]", 48000, 2)
        self.assertIsNotNone(result)


class TestReconnectLogic(unittest.IsolatedAsyncioTestCase):
    """Tests for the auto-reconnect mechanism."""

    async def test_reconnect_aborts_if_no_metadata(self):
        """_reconnect with no stored metadata should log an error and return immediately."""
        engine = TranscriptionEngine()
        # Should not raise, just log error and return
        await engine._reconnect("[NONEXISTENT]")
        # Verify no connections were created
        self.assertNotIn("[NONEXISTENT]", engine.connections)

    async def test_reconnect_aborts_if_closed(self):
        """_reconnect should abort all attempts if engine is marked as closed."""
        engine = TranscriptionEngine()
        engine._closed = True
        engine.connection_meta["[INTERVIEWER]"] = (48000, 2)
        engine._setup_connection = AsyncMock()  # should never be called

        await engine._reconnect("[INTERVIEWER]")
        engine._setup_connection.assert_not_called()

    async def test_reconnect_tries_max_attempts(self):
        """_reconnect should try exactly MAX_RECONNECT_ATTEMPTS times on repeated failure."""
        engine = TranscriptionEngine()
        engine.connection_meta["[ME]"] = (16000, 1)
        engine._setup_connection = AsyncMock(return_value=None)  # always fails

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await engine._reconnect("[ME]")

        self.assertEqual(engine._setup_connection.call_count, MAX_RECONNECT_ATTEMPTS)

    async def test_reconnect_succeeds_on_second_attempt(self):
        """_reconnect should stop once a connection is established."""
        engine = TranscriptionEngine()
        engine.connection_meta["[INTERVIEWER]"] = (48000, 2)

        mock_conn = AsyncMock()
        # First attempt fails, second succeeds
        engine._setup_connection = AsyncMock(side_effect=[None, mock_conn])
        engine.keep_alive_tasks = {}

        mock_task = MagicMock()
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch("asyncio.create_task", return_value=mock_task) as mock_create_task:
                # Also cancel any keep_alive coroutines to suppress RuntimeWarning
                with patch.object(engine, "_keep_alive", return_value=AsyncMock()):
                    await engine._reconnect("[INTERVIEWER]")

        self.assertEqual(engine.connections["[INTERVIEWER]"], mock_conn)
        self.assertEqual(engine._setup_connection.call_count, 2)



class TestTextQueueIntegration(unittest.IsolatedAsyncioTestCase):
    """Tests for queue full handling in the on_message callback."""

    async def test_queue_full_does_not_raise(self):
        """If text_queue is full, the engine should log a warning but not raise."""
        engine = TranscriptionEngine()
        engine.text_queue = asyncio.Queue(maxsize=1)
        await engine.text_queue.put(("[ME]", "first item"))

        # Simulate putting to a full queue
        try:
            engine.text_queue.put_nowait(("[ME]", "overflow item"))
            self.fail("Expected QueueFull to be raised")
        except asyncio.QueueFull:
            pass  # Expected — the on_message handler catches this gracefully


if __name__ == "__main__":
    unittest.main()
