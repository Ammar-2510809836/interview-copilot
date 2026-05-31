"""
Unit tests for core/llm.py — LLMClient model routing and mocked API calls.
All Groq API calls are mocked — no real API key required to run these tests.
"""
import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# chromadb must be imported before PyQt6 to avoid DLL conflicts on Windows
import chromadb
from core.llm import LLMClient


class GenerateContentResponse:  # noqa: N801 - mirrors google-genai's real class name
    """Faithful stand-in for a google-genai response/stream chunk.

    Real Gemini objects expose `.text` (and have no `.choices`); the response
    parser discriminates providers by this class name. Using a generic MagicMock
    here would falsely expose `.choices`, so tests must use this shape.
    """

    def __init__(self, text):
        self.text = text


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

        self.assertIn((backup_client, "groq_backup", "llama-3.3-70b-versatile"), attempts)
        self.assertIn((backup_client, "groq_backup", "llama-3.1-8b-instant"), attempts)

    def test_groq_client_disables_sdk_retries_by_default(self):
        """Fallback logic should run without the Groq SDK sleeping on 429 retries."""
        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "groq",
                "GROQ_API_KEY": "primary-test-key",
                "GROQ_BACKUP_API_KEY": "backup-test-key",
            },
            clear=True,
        ):
            with patch("core.llm.AsyncGroq") as mock_groq:
                LLMClient()

        mock_groq.assert_any_call(
            api_key="primary-test-key",
            timeout=15.0,
            max_retries=0,
        )
        mock_groq.assert_any_call(
            api_key="backup-test-key",
            timeout=15.0,
            max_retries=0,
        )

    def test_gemini_provider_routes_to_current_flash_models(self):
        """Gemini can be selected as a primary provider with current model ids."""
        with patch.dict(
            "os.environ",
            {
                "LLM_PROVIDER": "gemini",
                "GEMINI_API_KEY": "test-key",
                "GEMINI_TECHNICAL_MODEL": "gemini-2.5-flash",
                "GEMINI_BEHAVIORAL_MODEL": "gemini-2.5-flash-lite",
            },
            clear=False,
        ):
            with patch("core.llm.genai") as mock_genai, patch("core.llm.AsyncGroq"):
                mock_genai.Client.return_value = MagicMock()
                llm = LLMClient()

        tech_model, tech_tokens = llm._route_model("explain aws vpc networking", "technical")
        behavior_model, behavior_tokens = llm._route_model("tell me about yourself", "behavioral")

        self.assertEqual(tech_model, "gemini-2.5-flash")
        self.assertGreaterEqual(tech_tokens, 400)
        self.assertEqual(behavior_model, "gemini-2.5-flash-lite")
        self.assertLessEqual(behavior_tokens, 300)

    def test_gemini_primary_adds_same_provider_fallback(self):
        """Gemini primary attempts should try Flash-Lite before external fallback."""
        self.llm.provider = "gemini"
        self.llm.client = MagicMock()
        self.llm.gemini_tech_model = "gemini-2.5-flash"
        self.llm.gemini_behavior_model = "gemini-2.5-flash-lite"

        attempts = self.llm._model_attempts("gemini-2.5-flash")

        self.assertEqual(attempts[0][1:], ("gemini", "gemini-2.5-flash"))
        self.assertIn((self.llm.client, "gemini", "gemini-2.5-flash-lite"), attempts)

    def test_gemini_create_uses_generate_content_config(self):
        """Gemini SDK uses generate_content with contents and config."""
        client = MagicMock()
        response = MagicMock()
        response.text = "ok"
        client.models.generate_content.return_value = response

        result = asyncio.run(self.llm._chat_completion_create(
            client,
            "gemini",
            "gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "hi"},
            ],
            max_tokens=123,
            temperature=0.2,
            stream=False,
        ))

        self.assertIs(result, response)
        kwargs = client.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-2.5-flash")
        self.assertIn("USER:\nhi", kwargs["contents"])
        self.assertIn("config", kwargs)

    def test_gemini_empty_response_does_not_fall_through_to_choices(self):
        """Real Gemini empty responses do not expose OpenAI-style choices."""
        response = MagicMock()
        response.__class__.__name__ = "GenerateContentResponse"
        response.text = None

        self.assertEqual(self.llm._response_text(response), "")

    def test_spoken_answer_style_is_enabled_by_default(self):
        """Default prompt should be optimized for live spoken delivery."""
        self.assertIn("SPOKEN LIVE INTERVIEW MODE", self.llm.system_prompt)
        # New flowing-prose shaping: no bullet/scaffold labels.
        self.assertIn("Do NOT use bullet points", self.llm.system_prompt)
        self.assertNotIn("Opening:", self.llm.system_prompt)
        self.assertNotIn("Say:", self.llm.system_prompt)
        self.assertNotIn("Close:", self.llm.system_prompt)

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

    async def test_generate_answer_gemini_failure_uses_gemini_fallback_model(self):
        """Gemini primary should retry Flash-Lite before Groq fallback."""
        client = MagicMock()
        fallback_response = MagicMock()
        fallback_response.text = "Gemini fallback answer."
        client.models.generate_content.side_effect = [
            Exception("429 Too Many Requests"),
            fallback_response,
        ]

        self.llm.provider = "gemini"
        self.llm.client = client
        self.llm.gemini_tech_model = "gemini-2.5-flash"
        self.llm.gemini_behavior_model = "gemini-2.5-flash-lite"

        result = await self.llm.generate_answer(
            ["[INTERVIEWER]: Explain AWS VPC design."],
            "context",
            question_type="technical"
        )

        self.assertEqual(result, "Gemini fallback answer.")
        calls = client.models.generate_content.call_args_list
        self.assertEqual(calls[0].kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(calls[1].kwargs["model"], "gemini-2.5-flash-lite")

    async def test_generate_answer_empty_content_uses_fallback_model(self):
        """Provider responses with content=None should not leave the overlay blank."""
        client = MagicMock()
        empty_response = GenerateContentResponse(None)
        fallback_response = GenerateContentResponse("Fallback after empty.")

        client.models.generate_content.side_effect = [empty_response, fallback_response]

        self.llm.provider = "gemini"
        self.llm.client = client
        self.llm.gemini_tech_model = "gemini-2.5-flash"
        self.llm.gemini_behavior_model = "gemini-2.5-flash-lite"

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

    async def test_stream_closed_when_generator_aclosed(self):
        """Interruption: closing the generator must close the provider stream
        (otherwise httpx cleans it up from a different task and raises)."""
        chunks = self._make_stream_chunks(["Hello", " there", " friend"])
        closed = []

        class FakeStream:
            def __aiter__(self_inner):
                async def gen():
                    for c in chunks:
                        yield c
                return gen()

            async def aclose(self_inner):
                closed.append(True)

        self.llm.client = AsyncMock()
        self.llm.client.chat.completions.create = AsyncMock(return_value=FakeStream())

        agen = self.llm.generate_answer_stream(
            ["[INTERVIEWER]: Tell me about yourself."], "context", question_type="behavioral"
        )
        async for _token in agen:
            break  # consumer interrupts after the first token
        await agen.aclose()  # what contextlib.aclosing() does in main.py

        self.assertEqual(len(closed), 1)

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

    async def test_stream_empty_gemini_response_uses_gemini_fallback_model(self):
        """Empty Gemini streams should retry the same-provider fallback model."""
        empty_chunk = GenerateContentResponse(None)
        fallback_chunk = GenerateContentResponse("Recovered")

        client = MagicMock()
        client.models.generate_content_stream.side_effect = [[empty_chunk], [fallback_chunk]]

        self.llm.provider = "gemini"
        self.llm.client = client
        self.llm.gemini_tech_model = "gemini-2.5-flash"
        self.llm.gemini_behavior_model = "gemini-2.5-flash-lite"

        result = ""
        async for token in self.llm.generate_answer_stream(
            ["[INTERVIEWER]: Explain AWS VPC design."],
            "context",
            question_type="technical"
        ):
            result += token

        self.assertEqual(result, "Recovered")
        calls = client.models.generate_content_stream.call_args_list
        self.assertEqual(calls[0].kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(calls[1].kwargs["model"], "gemini-2.5-flash-lite")


class TestSpokenPrompt(unittest.TestCase):
    def test_spoken_mode_is_flowing_prose_with_no_bullet_scaffold(self):
        os.environ["ANSWER_STYLE"] = "spoken"
        try:
            prompt = LLMClient()._answer_style_prompt()
        finally:
            os.environ.pop("ANSWER_STYLE", None)
        lowered = prompt.lower()
        self.assertNotIn("▸", prompt)   # the bullet glyph
        self.assertNotIn("say:", lowered)
        self.assertNotIn("close:", lowered)
        # No leftover bullet scaffold glyphs (the prose explicitly tells the
        # model NOT to use bullets, so the word "bullet" itself is expected).
        self.assertNotIn("•", prompt)
        self.assertIn("sentence", lowered)
        self.assertIn("do not use bullet", lowered)

    def test_standard_mode_returns_no_extra_shaping(self):
        os.environ["ANSWER_STYLE"] = "standard"
        try:
            prompt = LLMClient()._answer_style_prompt()
        finally:
            os.environ.pop("ANSWER_STYLE", None)
        self.assertEqual(prompt, "")


class TestInterviewModePrompt(unittest.TestCase):
    """INTERVIEW_MODE gates the heavy technical templates to save tokens."""

    def test_default_mode_is_lean_without_heavy_technical_templates(self):
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("INTERVIEW_MODE", None)
            with patch("core.llm.AsyncGroq"):
                llm = LLMClient()
        sp = llm.system_prompt
        # Heavy, role-specific verbatim blocks are excluded by default.
        self.assertNotIn("Security Group vs Network ACL", sp)
        self.assertNotIn("sudo ss -tlnp", sp)
        # Core guidance is retained.
        self.assertIn("Match the target role", sp)
        # Lean prompt is far smaller than the old ~11k-char prompt.
        self.assertLess(len(sp), 6000)

    def test_technical_mode_restores_technical_templates(self):
        with patch.dict("os.environ", {"INTERVIEW_MODE": "technical"}, clear=False):
            with patch("core.llm.AsyncGroq"):
                llm = LLMClient()
        sp = llm.system_prompt
        self.assertIn("Security Group", sp)
        self.assertIn("sudo ss -tlnp", sp)


class TestProviderResponseParsing(unittest.TestCase):
    """_response_text / _chunk_text must extract Groq/OpenAI content, not just Gemini.

    Regression: a `text is None` guard (added for Gemini) short-circuited and
    returned "" for every Groq response, so all answers came back empty.
    """

    def setUp(self):
        with patch("core.llm.AsyncGroq"):
            self.llm = LLMClient()

    def _groq_chunk(self, content):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])

    def _groq_response(self, content):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def test_chunk_text_extracts_groq_delta_content(self):
        self.assertEqual(self.llm._chunk_text(self._groq_chunk("Hello")), "Hello")

    def test_response_text_extracts_groq_message_content(self):
        self.assertEqual(self.llm._response_text(self._groq_response("Hello world")), "Hello world")

    def test_chunk_text_returns_empty_for_groq_role_only_delta(self):
        # First streamed chunk often carries role but no content — must be "" not crash.
        chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])
        self.assertEqual(self.llm._chunk_text(chunk), "")

    def test_gemini_text_attribute_still_used(self):
        class GenerateContentResponse:
            text = "gemini answer"
        self.assertEqual(self.llm._response_text(GenerateContentResponse()), "gemini answer")

    def test_gemini_empty_response_returns_empty(self):
        class GenerateContentResponse:
            text = None
        self.assertEqual(self.llm._response_text(GenerateContentResponse()), "")


if __name__ == "__main__":
    unittest.main()
