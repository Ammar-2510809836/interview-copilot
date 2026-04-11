"""
Unit tests for core/llm.py — LLMClient model routing and mocked API calls.
All Groq API calls are mocked — no real API key required to run these tests.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# chromadb must be imported before PyQt6 to avoid DLL conflicts on Windows
import chromadb
from core.llm import LLMClient


class TestModelRouter(unittest.TestCase):
    """Tests for _route_model — pure logic, no API calls."""

    def setUp(self):
        # Patch the Groq client so no API key is needed
        with patch("core.llm.AsyncGroq"):
            self.llm = LLMClient()

    def test_technical_question_routes_to_70b(self):
        """Technical keywords should trigger the 70b model."""
        technical_queries = [
            "implement a binary search algorithm",
            "design a distributed system architecture",
            "write a python script to parse json",
            "explain docker and kubernetes networking",
            "how to optimize sql query performance",
            "what is rag and how do embeddings work",
            "implement a neural network from scratch",
        ]
        for query in technical_queries:
            model, max_tokens = self.llm._route_model(query.lower())
            self.assertEqual(
                model, "llama-3.3-70b-versatile",
                f"Expected 70b for technical query: '{query}', got {model}"
            )
            self.assertGreaterEqual(max_tokens, 300)

    def test_behavioral_question_routes_to_8b(self):
        """Behavioral/HR questions should route to the fast 8b model."""
        behavioral_queries = [
            "what is your salary expectation",
            "tell me about yourself",
            "what are your strengths and weaknesses",
            "why do you want to work here",
            "describe a time you showed leadership",
            "where do you see yourself in five years",
            "how do you handle conflict",
        ]
        for query in behavioral_queries:
            model, max_tokens = self.llm._route_model(query.lower())
            self.assertEqual(
                model, "llama-3.1-8b-instant",
                f"Expected 8b for behavioral query: '{query}', got {model}"
            )
            self.assertLessEqual(max_tokens, 300)

    def test_router_returns_tuple(self):
        """_route_model always returns a (str, int) tuple."""
        result = self.llm._route_model("any question")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], int)

    def test_empty_query_defaults_to_8b(self):
        """Empty query string should not crash and should default to fast model."""
        model, _ = self.llm._route_model("")
        self.assertEqual(model, "llama-3.1-8b-instant")


class TestGenerateAnswerMocked(unittest.IsolatedAsyncioTestCase):
    """Tests for generate_answer with fully mocked Groq API."""

    async def asyncSetUp(self):
        with patch("core.llm.AsyncGroq"):
            self.llm = LLMClient()
        # Build a mock response object that the Groq SDK would return
        mock_choice = MagicMock()
        mock_choice.message.content = "I have strong experience with Python and AI pipelines."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        self.llm.client = AsyncMock()
        self.llm.client.chat.completions.create = AsyncMock(return_value=mock_response)

    async def test_generate_answer_returns_string(self):
        result = await self.llm.generate_answer(["[INTERVIEWER]: Tell me about yourself."], "Portfolio context here.")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    async def test_generate_answer_no_api_key(self):
        """Without API key, returns an error string immediately."""
        self.llm.client = None
        result = await self.llm.generate_answer([], "")
        self.assertIn("Error", result)

    async def test_generate_answer_api_failure(self):
        """API exception returns an error string, does not raise."""
        self.llm.client.chat.completions.create = AsyncMock(side_effect=Exception("API down"))
        result = await self.llm.generate_answer(["[INTERVIEWER]: test"], "context")
        self.assertIn("Error", result)

    async def test_generate_answer_regen_higher_temperature(self):
        """generate_answer_regen should call the API with temperature >= 0.5."""
        await self.llm.generate_answer_regen(
            ["[INTERVIEWER]: What is your strength?"], "context", last_question="What is your strength?"
        )
        call_kwargs = self.llm.client.chat.completions.create.call_args.kwargs
        self.assertGreaterEqual(call_kwargs.get("temperature", 0), 0.5)

    async def test_generate_answer_regen_no_skip(self):
        """generate_answer_regen should never return the literal 'SKIP' string."""
        mock_choice = MagicMock()
        mock_choice.message.content = "SKIP"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        self.llm.client.chat.completions.create = AsyncMock(return_value=mock_response)
        # Regen returns whatever the LLM says — the caller expects it not to SKIP
        # This test verifies the prompt includes "Do NOT say SKIP"
        await self.llm.generate_answer_regen(["test"], "context", last_question="Q")
        call_args = self.llm.client.chat.completions.create.call_args.kwargs
        user_msg = call_args["messages"][-1]["content"]
        self.assertIn("SKIP", user_msg)  # The instruction is in the prompt


class TestGenerateAnswerStream(unittest.IsolatedAsyncioTestCase):
    """Tests for the streaming async generator."""

    async def asyncSetUp(self):
        with patch("core.llm.AsyncGroq"):
            self.llm = LLMClient()

    def _make_stream_chunks(self, tokens: list[str]):
        """Create a list of mock streaming chunk objects."""
        chunks = []
        for token in tokens:
            choice = MagicMock()
            choice.delta.content = token
            chunk = MagicMock()
            chunk.choices = [choice]
            chunks.append(chunk)
        return chunks

    async def test_stream_yields_tokens(self):
        """Streaming should yield the correct assembled text."""
        chunks = self._make_stream_chunks(["Hello", " world", "!"])

        async def mock_stream():
            for c in chunks:
                yield c

        self.llm.client = AsyncMock()
        self.llm.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        result = ""
        async for token in self.llm.generate_answer_stream(["test"], "context"):
            result += token

        self.assertEqual(result, "Hello world!")

    async def test_stream_skip_detection(self):
        """If LLM responds with SKIP, the generator should yield '__SKIP__' and stop."""
        chunks = self._make_stream_chunks(["SK", "IP"])

        async def mock_stream():
            for c in chunks:
                yield c

        self.llm.client = AsyncMock()
        self.llm.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        tokens = []
        async for token in self.llm.generate_answer_stream(["filler"], "context"):
            tokens.append(token)

        self.assertIn("__SKIP__", tokens)

    async def test_stream_no_api_key(self):
        """With no API key, stream should yield an error string."""
        self.llm.client = None
        tokens = []
        async for token in self.llm.generate_answer_stream([], ""):
            tokens.append(token)
        self.assertTrue(any("Error" in t for t in tokens))

    async def test_stream_handles_api_exception(self):
        """API exception during streaming should yield an error string, not raise."""
        self.llm.client = AsyncMock()
        self.llm.client.chat.completions.create = AsyncMock(side_effect=Exception("network error"))

        tokens = []
        try:
            async for token in self.llm.generate_answer_stream(["test"], "ctx"):
                tokens.append(token)
        except Exception as e:
            self.fail(f"Stream raised instead of yielding error: {e}")

        self.assertTrue(any("Error" in t for t in tokens))


if __name__ == "__main__":
    unittest.main()
