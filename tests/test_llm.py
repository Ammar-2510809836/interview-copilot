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

    def test_groq_backup_key_adds_backup_attempts(self):
        """A second Groq key can be used after the primary Groq fallback chain."""
        backup_client = AsyncMock()
        self.llm.groq_backup_client = backup_client

        attempts = self.llm._model_attempts("llama-3.3-70b-versatile")

        self.assertIn((backup_client, "groq_backup", "llama-3.1-8b-instant"), attempts)

    def test_cerebras_provider_routes_to_cerebras_models(self):
        """Cerebras can be selected as a primary provider."""
        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "cerebras",
                "CEREBRAS_API_KEY": "test-key",
                "CEREBRAS_TECHNICAL_MODEL": "gpt-oss-120b",
                "CEREBRAS_BEHAVIORAL_MODEL": "zai-glm-4.7",
            },
            clear=False,
        ):
            with patch("core.llm.Cerebras") as mock_cerebras, patch("core.llm.AsyncGroq"):
                mock_cerebras.return_value = MagicMock()
                llm = LLMClient()

        tech_model, tech_tokens = llm._route_model("explain aws vpc networking", "technical")
        behavior_model, behavior_tokens = llm._route_model("tell me about yourself", "behavioral")

        self.assertEqual(tech_model, "gpt-oss-120b")
        self.assertGreaterEqual(tech_tokens, 400)
        self.assertEqual(behavior_model, "zai-glm-4.7")
        self.assertLessEqual(behavior_tokens, 300)

    def test_cerebras_client_disables_sdk_retries_by_default(self):
        """Cerebras 429s should reach our fallback logic immediately."""
        with patch.dict(
            "os.environ",
            {"LLM_PROVIDER": "cerebras", "CEREBRAS_API_KEY": "test-key"},
            clear=False,
        ):
            with patch("core.llm.Cerebras") as mock_cerebras, patch("core.llm.AsyncGroq"):
                mock_cerebras.return_value = MagicMock()
                LLMClient()

        kwargs = mock_cerebras.call_args.kwargs
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertEqual(kwargs["timeout"], 12.0)

    def test_cerebras_primary_adds_same_provider_fallback(self):
        """Cerebras primary attempts should try zai-glm-4.7 before external fallback."""
        self.llm.provider = "cerebras"
        self.llm.client = MagicMock()
        self.llm.cerebras_tech_model = "gpt-oss-120b"
        self.llm.cerebras_behavior_model = "zai-glm-4.7"

        attempts = self.llm._model_attempts("gpt-oss-120b")

        self.assertEqual(attempts[0][1:], ("cerebras", "gpt-oss-120b"))
        self.assertIn((self.llm.client, "cerebras", "zai-glm-4.7"), attempts)

    def test_cerebras_create_uses_max_completion_tokens(self):
        """Cerebras SDK uses max_completion_tokens instead of max_tokens."""
        client = MagicMock()
        response = MagicMock()
        client.chat.completions.create.return_value = response

        result = asyncio.run(self.llm._chat_completion_create(
            client,
            "cerebras",
            "gpt-oss-120b",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=123,
            temperature=0.2,
            stream=True,
        ))

        self.assertIs(result, response)
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-oss-120b")
        self.assertEqual(kwargs["max_completion_tokens"], 123)
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["reasoning_effort"], "medium")

    def test_spoken_answer_style_is_enabled_by_default(self):
        """Default prompt should be optimized for live spoken delivery."""
        self.assertIn("SPOKEN LIVE INTERVIEW MODE", self.llm.system_prompt)
        self.assertIn("Opening:", self.llm.system_prompt)
        self.assertIn("Say:", self.llm.system_prompt)
        self.assertIn("Close:", self.llm.system_prompt)

    def test_system_prompt_keeps_role_adaptation_generic(self):
        """Base prompt should not hardcode one job family or vendor."""
        lower_prompt = self.llm.system_prompt.lower()

        self.assertIn("match the target role", lower_prompt)
        self.assertNotIn("zendesk", lower_prompt)
        self.assertNotIn("intercom", lower_prompt)
        self.assertNotIn("gorgias", lower_prompt)
        self.assertNotIn("servicenow", lower_prompt)
        self.assertNotIn("customer success/support", lower_prompt)

    def test_standard_answer_style_can_disable_spoken_prompt(self):
        """ANSWER_STYLE=standard keeps the old prompt shape available."""
        with patch.dict("os.environ", {"ANSWER_STYLE": "standard"}, clear=False):
            with patch("core.llm.AsyncGroq"):
                llm = LLMClient()

        self.assertNotIn("SPOKEN LIVE INTERVIEW MODE", llm.system_prompt)


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

    async def test_generate_answer_rate_limit_uses_fallback_model(self):
        """A Groq 429 should retry on the configured fallback model."""
        fallback_choice = MagicMock()
        fallback_choice.message.content = "Fallback answer."
        fallback_response = MagicMock()
        fallback_response.choices = [fallback_choice]
        self.llm.client.chat.completions.create = AsyncMock(
            side_effect=[Exception("429 rate_limit_exceeded tokens per day"), fallback_response]
        )

        result = await self.llm.generate_answer(
            ["[INTERVIEWER]: Explain AWS VPC design."],
            "context",
            question_type="technical"
        )

        self.assertEqual(result, "Fallback answer.")
        calls = self.llm.client.chat.completions.create.call_args_list
        self.assertEqual(calls[0].kwargs["model"], "llama-3.3-70b-versatile")
        self.assertEqual(calls[1].kwargs["model"], "llama-3.1-8b-instant")

    async def test_generate_answer_nvidia_failure_uses_groq_fallback_client(self):
        """Non-Groq primary providers should fall back to the Groq fallback client."""
        primary_client = AsyncMock()
        fallback_client = AsyncMock()

        fallback_choice = MagicMock()
        fallback_choice.message.content = "Groq fallback answer."
        fallback_response = MagicMock()
        fallback_response.choices = [fallback_choice]

        primary_client.chat.completions.create = AsyncMock(side_effect=Exception("timeout"))
        fallback_client.chat.completions.create = AsyncMock(return_value=fallback_response)

        self.llm.provider = "nvidia"
        self.llm.client = primary_client
        self.llm.fallback_client = fallback_client

        result = await self.llm.generate_answer(
            ["[INTERVIEWER]: Explain AWS VPC design."],
            "context",
            question_type="technical"
        )

        self.assertEqual(result, "Groq fallback answer.")
        primary_client.chat.completions.create.assert_awaited_once()
        fallback_client.chat.completions.create.assert_awaited_once()
        self.assertEqual(
            fallback_client.chat.completions.create.call_args.kwargs["model"],
            "llama-3.1-8b-instant"
        )

    async def test_generate_answer_cerebras_failure_uses_cerebras_fallback_model(self):
        """Cerebras primary should retry zai-glm-4.7 before Groq fallback."""
        client = MagicMock()
        fallback_choice = MagicMock()
        fallback_choice.message.content = "Cerebras fallback answer."
        fallback_response = MagicMock()
        fallback_response.choices = [fallback_choice]
        client.chat.completions.create.side_effect = [
            Exception("429 Too Many Requests"),
            fallback_response,
        ]

        self.llm.provider = "cerebras"
        self.llm.client = client
        self.llm.cerebras_tech_model = "gpt-oss-120b"
        self.llm.cerebras_behavior_model = "zai-glm-4.7"

        result = await self.llm.generate_answer(
            ["[INTERVIEWER]: Explain AWS VPC design."],
            "context",
            question_type="technical"
        )

        self.assertEqual(result, "Cerebras fallback answer.")
        calls = client.chat.completions.create.call_args_list
        self.assertEqual(calls[0].kwargs["model"], "gpt-oss-120b")
        self.assertEqual(calls[1].kwargs["model"], "zai-glm-4.7")

    async def test_generate_answer_empty_content_uses_fallback_model(self):
        """Provider responses with content=None should not leave the overlay blank."""
        client = MagicMock()
        empty_choice = MagicMock()
        empty_choice.message.content = None
        empty_response = MagicMock()
        empty_response.choices = [empty_choice]

        fallback_choice = MagicMock()
        fallback_choice.message.content = "Fallback after empty."
        fallback_response = MagicMock()
        fallback_response.choices = [fallback_choice]

        client.chat.completions.create.side_effect = [empty_response, fallback_response]

        self.llm.provider = "cerebras"
        self.llm.client = client
        self.llm.cerebras_tech_model = "gpt-oss-120b"
        self.llm.cerebras_behavior_model = "zai-glm-4.7"

        result = await self.llm.generate_answer(
            ["[INTERVIEWER]: Please introduce yourself briefly."],
            "context",
            question_type="technical"
        )

        self.assertEqual(result, "Fallback after empty.")

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


class TestTurnClassification(unittest.IsolatedAsyncioTestCase):
    """Tests for conversational turn-taking classification."""

    async def asyncSetUp(self):
        with patch("core.llm.AsyncGroq"):
            self.llm = LLMClient()
        self.llm.client = None

    async def test_turn_classifier_skips_filler_without_api(self):
        result = await self.llm.classify_turn("okay")
        self.assertEqual(result["intent"], "filler")
        self.assertFalse(result["should_answer_now"])

    async def test_turn_classifier_waits_for_incomplete_question_without_api(self):
        result = await self.llm.classify_turn("Can you explain AWS deployment with")
        self.assertEqual(result["intent"], "incomplete")
        self.assertFalse(result["should_answer_now"])

    async def test_turn_classifier_detects_followup_without_api(self):
        result = await self.llm.classify_turn(
            "What if the deployment fails halfway?",
            previous_question="How do you design a CI/CD pipeline?"
        )
        self.assertEqual(result["intent"], "followup")
        self.assertTrue(result["should_answer_now"])

    async def test_turn_classifier_parses_json_api_response(self):
        mock_choice = MagicMock()
        mock_choice.message.content = (
            '{"intent":"interruption","should_answer_now":true,'
            '"confidence":0.92,"clean_question":"Why Terraform instead of CloudFormation?"}'
        )
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        self.llm.client = AsyncMock()
        self.llm.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await self.llm.classify_turn(
            "Why Terraform instead of CloudFormation?",
            previous_question="Explain your AWS IaC approach.",
            was_answering=True
        )

        self.assertEqual(result["intent"], "interruption")
        self.assertTrue(result["should_answer_now"])
        self.assertEqual(result["clean_question"], "Why Terraform instead of CloudFormation?")


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

    async def test_stream_rate_limit_uses_fallback_model(self):
        """Streaming retries on fallback model when the primary model hits a 429."""
        chunks = self._make_stream_chunks(["Fallback", " stream"])

        async def mock_stream():
            for c in chunks:
                yield c

        self.llm.client = AsyncMock()
        self.llm.client.chat.completions.create = AsyncMock(
            side_effect=[Exception("429 rate_limit_exceeded tokens per day"), mock_stream()]
        )

        result = ""
        async for token in self.llm.generate_answer_stream(
            ["[INTERVIEWER]: Explain AWS VPC design."],
            "context",
            question_type="technical"
        ):
            result += token

        self.assertEqual(result, "Fallback stream")
        calls = self.llm.client.chat.completions.create.call_args_list
        self.assertEqual(calls[0].kwargs["model"], "llama-3.3-70b-versatile")
        self.assertEqual(calls[1].kwargs["model"], "llama-3.1-8b-instant")

    async def test_stream_nvidia_failure_uses_groq_backup_client(self):
        """If primary and first Groq fallback fail, stream retries with the backup Groq key."""
        chunks = self._make_stream_chunks(["Groq", " fallback"])

        async def mock_stream():
            for c in chunks:
                yield c

        primary_client = AsyncMock()
        fallback_client = AsyncMock()
        backup_client = AsyncMock()
        primary_client.chat.completions.create = AsyncMock(side_effect=Exception("connection timeout"))
        fallback_client.chat.completions.create = AsyncMock(side_effect=Exception("429 rate_limit_exceeded"))
        backup_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        self.llm.provider = "nvidia"
        self.llm.client = primary_client
        self.llm.fallback_client = fallback_client
        self.llm.groq_backup_client = backup_client

        result = ""
        async for token in self.llm.generate_answer_stream(
            ["[INTERVIEWER]: Explain AWS VPC design."],
            "context",
            question_type="technical"
        ):
            result += token

        self.assertEqual(result, "Groq fallback")
        primary_client.chat.completions.create.assert_awaited_once()
        fallback_client.chat.completions.create.assert_awaited_once()
        backup_client.chat.completions.create.assert_awaited_once()
        self.assertEqual(
            backup_client.chat.completions.create.call_args.kwargs["model"],
            "llama-3.3-70b-versatile"
        )

    async def test_stream_empty_cerebras_response_uses_cerebras_fallback_model(self):
        """Empty Cerebras streams should retry the same-provider fallback model."""
        empty_chunk = self._make_stream_chunks([None])
        fallback_chunks = self._make_stream_chunks(["Recovered"])

        client = MagicMock()
        client.chat.completions.create.side_effect = [empty_chunk, fallback_chunks]

        self.llm.provider = "cerebras"
        self.llm.client = client
        self.llm.cerebras_tech_model = "gpt-oss-120b"
        self.llm.cerebras_behavior_model = "zai-glm-4.7"

        result = ""
        async for token in self.llm.generate_answer_stream(
            ["[INTERVIEWER]: Explain AWS VPC design."],
            "context",
            question_type="technical"
        ):
            result += token

        self.assertEqual(result, "Recovered")
        calls = client.chat.completions.create.call_args_list
        self.assertEqual(calls[0].kwargs["model"], "gpt-oss-120b")
        self.assertEqual(calls[1].kwargs["model"], "zai-glm-4.7")


if __name__ == "__main__":
    unittest.main()
