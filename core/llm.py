import os
import logging
import json
import re
import inspect
from groq import AsyncGroq
from openai import AsyncOpenAI

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

logger = logging.getLogger(__name__)

classification_prompt = """Classify this interview transcript fragment.

Return only one word from this set:
behavioral, technical, coding, followup, filler

Important:
- Cloud, Linux, DevOps, infrastructure, security, networking, databases, programming, debugging, troubleshooting, system design, tools, and command questions are technical.
- Questions asking "why" about a technical design, tool choice, trade-off, incident, command, architecture, or implementation are technical, not behavioral.
- Only use behavioral for HR/personal experience prompts like "tell me about a time", "conflict", "strength", "weakness", "motivation", "salary", or "why this role".

Question:
{question}
"""

turn_classification_prompt = """You classify real-time AI interview transcript turns.

The interviewer may speak conversationally, pause mid-question, interrupt the candidate, or ask short follow-ups.

Return ONLY valid JSON with these keys:
- intent: one of "new_question", "followup", "continuation", "filler", "interruption", "incomplete"
- should_answer_now: boolean
- confidence: number from 0 to 1
- clean_question: string

Rules:
- Use "filler" for acknowledgements like okay, right, thanks, good, yes with no question.
- Use "incomplete" when the text ends mid-thought or needs more words to make sense.
- Use "followup" for short contextual questions like "why?", "how so?", "what about rollback?", or references to prior answers.
- Use "continuation" when the interviewer resumed and the new text completes the same question.
- Use "interruption" when the interviewer cuts in while the candidate/copilot was answering and asks a different point.
- should_answer_now is false for filler and incomplete.

Recent conversation:
{history}

Previous answered question:
{previous_question}

Current accumulated interviewer text:
{current_text}
"""

class LLMClient:
    """
    Anti-hallucination LLM brain strictly generating 30-word bullet points.
    Supports multiple providers: Groq, NVIDIA NIM, and Gemini.
    """
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "groq").lower()
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.groq_backup_key = os.getenv("GROQ_BACKUP_API_KEY")
        self.nvidia_key = os.getenv("NVIDIA_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None
        self.fallback_client = None
        self.groq_backup_client = None

        # Dynamic Model Selection from .env
        self.groq_tech_model = os.getenv("GROQ_TECHNICAL_MODEL", "llama-3.3-70b-versatile")
        self.groq_behavior_model = os.getenv("GROQ_BEHAVIORAL_MODEL", "llama-3.1-8b-instant")
        self.groq_backup_tech_model = os.getenv("GROQ_BACKUP_TECHNICAL_MODEL", self.groq_tech_model)
        self.groq_backup_behavior_model = os.getenv("GROQ_BACKUP_BEHAVIORAL_MODEL", self.groq_behavior_model)

        self.nvidia_tech_model = os.getenv("NVIDIA_TECHNICAL_MODEL", "meta/llama-3.3-70b-instruct")
        self.nvidia_behavior_model = os.getenv("NVIDIA_BEHAVIORAL_MODEL", "meta/llama-3.1-8b-instruct")

        self.gemini_tech_model = os.getenv("GEMINI_TECHNICAL_MODEL", "gemini-2.5-flash")
        self.gemini_behavior_model = os.getenv("GEMINI_BEHAVIORAL_MODEL", "gemini-2.5-flash-lite")

        self.answer_style = os.getenv("ANSWER_STYLE", "spoken").strip().lower()
        # Lean prompt by default; INTERVIEW_MODE=technical restores the detailed
        # AWS/Linux/Terraform/coding templates (saves ~1k tokens/call otherwise).
        self.interview_mode = os.getenv("INTERVIEW_MODE", "general").strip().lower()
        # answer_max_bullets / answer_include_close: retained for env-compat; not used
        # by the current flowing-prose prompt (see _answer_style_prompt).
        self.answer_max_bullets = os.getenv("ANSWER_MAX_BULLETS", "4").strip()
        self.answer_include_close = os.getenv("ANSWER_INCLUDE_CLOSE", "true").strip().lower() not in {"0", "false", "no"}
        self.groq_timeout = self._env_float("GROQ_TIMEOUT", 15.0)
        self.groq_max_retries = self._env_int("GROQ_MAX_RETRIES", 0)

        if self.provider == "nvidia":
            if not self.nvidia_key:
                logger.warning("NVIDIA_API_KEY not found in environment! Falling back to Groq.")
                self.provider = "groq"
            else:
                logger.info(f"LLM Provider: NVIDIA NIM (Tech: {self.nvidia_tech_model})")
                self.client = AsyncOpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=self.nvidia_key
                )

        if self.provider == "gemini":
            if not self.gemini_key:
                logger.warning("GEMINI_API_KEY not found in environment! Falling back to Groq.")
                self.provider = "groq"
            elif genai is None:
                logger.warning("google-genai not installed! Falling back to Groq.")
                self.provider = "groq"
            else:
                logger.info(f"LLM Provider: Gemini (Tech: {self.gemini_tech_model})")
                self.client = genai.Client(api_key=self.gemini_key)

        if self.provider == "groq":
            if not self.groq_key:
                logger.warning("GROQ_API_KEY not found in environment!")
            logger.info(f"LLM Provider: Groq (Tech: {self.groq_tech_model})")
            self.client = self._make_groq_client(self.groq_key) if self.groq_key else None

        if self.provider not in {"groq", "nvidia", "gemini"}:
            logger.warning("Unsupported LLM_PROVIDER=%s; falling back to Groq.", self.provider)
            self.provider = "groq"
            self.client = self._make_groq_client(self.groq_key) if self.groq_key else None

        if self.provider != "groq" and self.groq_key:
            self.fallback_client = self._make_groq_client(self.groq_key)
        if self.groq_backup_key:
            self.groq_backup_client = self._make_groq_client(self.groq_backup_key)
        
        self.system_prompt = self._build_system_prompt()





    # Keywords that indicate a question needs the powerful large model
    TECHNICAL_KEYWORDS = {
        'algorithm', 'implement', 'code', 'design', 'architecture', 'system',
        'database', 'sql', 'nosql', 'docker', 'kubernetes', 'k8s', 'aws',
        'azure', 'gcp', 'cloud', 'linux', 'bash', 'python', 'api', 'rest',
        'graphql', 'microservice', 'latency', 'throughput', 'scalab', 'deploy',
        'devops', 'cicd', 'pipeline', 'terraform', 'debug', 'memory', 'cpu',
        'kernel', 'thread', 'async', 'concurrent', 'network', 'tcp', 'http',
        'rag', 'llm', 'vector', 'embedding', 'model', 'train', 'neural',
        'complexity', 'optimize', 'cache', 'load balanc', 'distributed',
        'security', 'encrypt', 'auth', 'token', 'jwt', 'oauth',
        'iam', 'role', 'user', 'policy', 's3', 'ec2', 'rds', 'alb',
        'vpc', 'subnet', 'cidr', 'route table', 'nat', 'nacl', 'acl',
        'security group', 'cloudformation', 'parameter', 'mapping',
        'linux', 'systemctl', 'systemd', 'netstat', 'ss', 'lsof', 'port',
        'server', 'process', 'service', 'socket', 'firewall', 'load balancer',
        'dns', 'tls', 'ssl', 'certificate', 'nginx', 'apache', 'database',
        'postgres', 'mysql', 'index', 'query', 'replica', 'backup', 'restore',
        'redis', 'queue', 'kafka', 'sqs', 'sns', 'event', 'message',
        'terraform', 'ansible', 'cloudformation', 'cdk', 'jenkins',
        'github actions', 'gitlab', 'prometheus', 'grafana', 'elk',
        'observability', 'monitoring', 'logging', 'alerting', 'incident',
        'kubernetes', 'pod', 'deployment', 'service', 'ingress', 'helm',
    }

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning("Invalid %s=%s; using %s.", name, value, default)
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            logger.warning("Invalid %s=%s; using %s.", name, value, default)
            return default

    def _make_groq_client(self, api_key: str) -> AsyncGroq:
        return AsyncGroq(
            api_key=api_key,
            timeout=self.groq_timeout,
            max_retries=self.groq_max_retries,
        )

    # Lean core — always included. Vendor/role-agnostic so it suits any interview.
    _LEAN_SYSTEM_PROMPT = """You are acting AS the candidate in a live interview. Speak in the first person ("I", "my", "I've"). You are NOT a coach giving tips.

## CRITICAL RULES
1. If the interviewer says only filler (yes/no/okay/right/good/thanks/hello/nice) with NO question, reply exactly with "SKIP".
2. If the transcript is incomplete, interrupted, or only conversational filler, reply exactly with "SKIP".
3. Never say "you should", "candidates should", or "it's important to". Use only "I", "my", "I've".
4. Answer the CURRENT question. Do not answer an earlier question the interviewer has already moved past. For a follow-up, answer the new question directly, building on my previous answer.
5. Match the target role from the retrieved context: be specific and practical, and never overstate seniority or claim experience that is not in the portfolio or interview context.
6. Keep it concise: a few natural spoken sentences. Lead with the direct answer, then one concrete example or proof point from my experience.

## BEHAVIORAL / HR QUESTIONS
For "tell me about a time", conflict, strengths/weaknesses, motivation, "why this role", salary, and availability: tell it as a short spoken story using STAR internally — a brief situation, what I personally did ("I", not "we"), and the result — woven together without section labels. For salary, be direct with a range. For "why this role", connect their work to my specific skills.

## VOICE
Good: "I architected a pipeline that cut deploy time from 45 minutes to 8." / "The biggest challenge was concurrent WebSocket connections, so I used asyncio with a connection pool." / "In my experience, the trade-off usually favors..."
Avoid: "I am familiar with...", "One should consider...", "Here are some key aspects...", "As a professional...".

## CONTEXT
Weave portfolio projects and experience in naturally when they fit. If asked about something not in my background, answer confidently from general knowledge without fabricating personal experience. Never say "I haven't done this, but...". If genuinely unsure of intent, give a brief confident answer rather than asking for clarification. Use `backticks` for code or tool names.
"""

    # Heavy technical templates — only added when INTERVIEW_MODE=technical.
    _TECHNICAL_ADDENDUM = """
---

## TECHNICAL QUESTIONS (concepts, architecture, trade-offs, troubleshooting, IaC, commands)
Lead with the exact answer first, then 2-4 precise points. Include at least one concrete command, configuration field, service behavior, trade-off, or example. Never use STAR for technical questions — answer them directly.

**Comparison questions:**
- One-sentence direct contrast
- 2-4 points covering behavior, scope, trade-off, and when I would use each
- If the distinction affects operations, mention the failure mode or practical consequence

**Troubleshooting questions:**
- Start with the first exact command/check I would run
- Then the next 2-3 checks in order
- End with the likely fix and how I would verify it

**Architecture/system design questions:**
- State the baseline design in one sentence
- Cover compute, data/storage, networking, security, observability, scaling, and cost only as relevant
- Mention one risk/trade-off and one mitigation

**Infrastructure-as-Code questions:**
- reusable modules/components, variables/parameters, environment separation, state/backend strategy, plan/apply flow, and drift prevention

**Command/script questions:**
- Give the exact command or minimal script before explanation
- Prefer modern commands, then mention legacy fallback only if useful
- Include a validation command or expected output cue

**Security Group vs Network ACL:**
- Security Groups are stateful, attached to ENIs/instances, allow-only rules with implicit deny, and return traffic is automatically allowed
- Network ACLs are stateless, subnet-level, ordered numbered rules, support allow and explicit deny, and inbound/outbound return paths must both be allowed
- Use Security Groups for workload-level access; use NACLs for coarse subnet guardrails or explicit subnet-level blocks

**Linux port/process troubleshooting (command-first):**
- `sudo ss -tlnp | grep ':443'` as the modern first command
- fallback: `sudo lsof -iTCP:443 -sTCP:LISTEN -P -n`
- map PID to service with `systemctl status <PID>` or `ps -fp <PID>`, then restart with `sudo systemctl restart <service>`
- verify with `sudo systemctl status <service>` and rerun `ss`

**Terraform/IaC structure:**
- reusable modules for VPC/EC2/RDS, per-environment `*.tfvars`, isolated state/backends, clear variable names like `vpc_cidr`, `instance_type`, `environment`
- inline security-group rules are simple but harder to manage at scale; separate `aws_security_group_rule` resources improve change tracking but need careful lifecycle ownership

**CloudFormation:**
- `Parameters` for deploy-time values, `Mappings` for environment lookups, `Conditions` for optional resources; example names like `VpcCidr`, `InstanceType`

## CODING / ALGORITHM QUESTIONS
Brief approach in one sentence, then a clean commented code block (Python unless asked otherwise), then time/space complexity and edge cases.
"""

    def _build_system_prompt(self) -> str:
        """Assemble the system prompt: lean core, optional technical detail, voice.

        Default is the lean core to minimize tokens per call. INTERVIEW_MODE=technical
        appends the detailed technical/coding templates.
        """
        prompt = self._LEAN_SYSTEM_PROMPT
        if self.interview_mode in {"technical", "tech", "swe", "devops", "engineering"}:
            prompt += self._TECHNICAL_ADDENDUM
        return prompt + self._answer_style_prompt()

    def _answer_style_prompt(self) -> str:
        """Spoken-delivery shaping for live interviews — flowing prose, no bullets.

        The app shows an instant thinking-bridge opener (see core/bridge_lines.py),
        so the model goes straight into substance with no opener of its own.
        """
        if self.answer_style not in {"spoken", "natural", "live"}:
            return ""

        return """

---

## SPOKEN LIVE INTERVIEW MODE
I am speaking these words out loud in a live interview, so they must sound like natural speech, not a written document.

- Speak in first person with contractions ("I'd", "I've", "that's"). Confident, specific, practical.
- Keep it to 3-4 flowing sentences. Lead with the actual answer, then one concrete proof point from my experience.
- Do NOT use bullet points, numbered lists, markdown bold, headings, or STAR/section labels.
- Do NOT open with a bridge or filler line — start directly on the substance (the app already supplied my opening words).
- For technical questions, say the concrete answer or exact command first, then briefly why.
- For behavioral questions, tell it as a short spoken story: the situation, what I did, and the result — woven together, not labelled.
- Leave room for a follow-up; do not cram in every detail.
- If the interviewer only said filler with no question, reply exactly with "SKIP".
"""

    def _has_technical_keyword(self, text_lower: str) -> bool:
        """Match technical routing terms without short-token false positives."""
        for kw in self.TECHNICAL_KEYWORDS:
            if len(kw) <= 3 or kw.replace("-", "").isalnum():
                if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", text_lower):
                    return True
            elif kw in text_lower:
                return True
        return False

    def _route_model(self, question_text: str, question_type: str = None) -> tuple[str, int]:
        """
        Route to fast or powerful model based on question content and provider.
        Returns (model_name, max_tokens).
        """
        if self.provider == "nvidia":
            # NVIDIA NIM model mapping
            if question_type:
                if question_type in ["coding", "technical"]:
                    return self.nvidia_tech_model, 500
                else:
                    return self.nvidia_behavior_model, 300

            text_lower = question_text.lower()
            is_technical = self._has_technical_keyword(text_lower)
            if is_technical:
                return self.nvidia_tech_model, 500
            else:
                return self.nvidia_behavior_model, 300
        elif self.provider == "gemini":
            if question_type:
                if question_type in ["coding", "technical"]:
                    return self.gemini_tech_model, 500
                elif question_type in ["behavioral", "followup"]:
                    return self.gemini_behavior_model, 300
                else:
                    return self.gemini_behavior_model, 250

            text_lower = question_text.lower()
            is_technical = self._has_technical_keyword(text_lower)
            if is_technical:
                logger.info(f"Router: Technical question → {self.gemini_tech_model}")
                return self.gemini_tech_model, 500
            else:
                logger.info(f"Router: Behavioral/HR question → {self.gemini_behavior_model}")
                return self.gemini_behavior_model, 250
        else:
            # Groq model mapping
            if question_type:
                if question_type == "coding":
                    return self.groq_tech_model, 600
                elif question_type == "technical":
                    return self.groq_tech_model, 400
                elif question_type in ["behavioral", "followup"]:
                    return self.groq_behavior_model, 300
                else:
                    return self.groq_behavior_model, 250

            text_lower = question_text.lower()
            is_technical = self._has_technical_keyword(text_lower)
            if is_technical:
                logger.info(f"Router: Technical question → {self.groq_tech_model}")
                return self.groq_tech_model, 400
            else:
                logger.info(f"Router: Behavioral/HR question → {self.groq_behavior_model}")
                return self.groq_behavior_model, 250

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Detect provider quota/rate-limit failures from SDK exceptions."""
        error_text = str(error).lower()
        return any(
            marker in error_text
            for marker in (
                "429",
                "too many requests",
                "rate limit",
                "rate_limit",
                "rate_limit_exceeded",
                "tokens per day",
                "tokens per minute",
            )
        )

    def _is_retryable_llm_error(self, error: Exception) -> bool:
        """Detect failures worth retrying on the emergency fallback provider."""
        error_text = str(error).lower()
        return self._is_rate_limit_error(error) or any(
            marker in error_text
            for marker in (
                "timeout",
                "timed out",
                "connection",
                "connecterror",
                "service unavailable",
                "503",
                "502",
                "500",
                "empty content",
            )
        )

    def _fallback_models(self) -> list[str]:
        """Return primary Groq fallback models used when any provider fails."""
        configured = os.getenv("GROQ_FALLBACK_MODELS", "")
        fallback_models = [m.strip() for m in configured.split(",") if m.strip()]
        if self.groq_behavior_model:
            fallback_models.append(self.groq_behavior_model)

        models = []
        for model in fallback_models:
            if model not in models:
                models.append(model)
        return models

    def _backup_groq_models(self, primary_model: str) -> list[str]:
        """Return fallback models for the second Groq key."""
        configured = os.getenv("GROQ_BACKUP_FALLBACK_MODELS", "")
        fallback_models = [m.strip() for m in configured.split(",") if m.strip()]
        fallback_models.extend([
            self.groq_backup_tech_model,
            self.groq_backup_behavior_model,
        ])

        models = []
        for model in fallback_models:
            if model and model not in models:
                models.append(model)
        return models

    def _gemini_fallback_models(self, primary_model: str) -> list[str]:
        """Return same-provider Gemini fallback models."""
        configured = os.getenv("GEMINI_FALLBACK_MODELS", "")
        fallback_models = [m.strip() for m in configured.split(",") if m.strip()]
        fallback_models.append(self.gemini_behavior_model)

        models = []
        for model in fallback_models:
            if model and model != primary_model and model not in models:
                models.append(model)
        return models

    def _model_attempts(self, primary_model: str) -> list[tuple[object, str, str]]:
        """
        Return primary provider attempt plus Groq fallback attempts.
        Each item is (client, provider_name, model).
        """
        attempts = [(self.client, self.provider, primary_model)]

        if self.provider == "gemini" and self.client:
            for model in self._gemini_fallback_models(primary_model):
                attempts.append((self.client, "gemini", model))

        groq_client = self.client if self.provider == "groq" else self.fallback_client
        if groq_client:
            for model in self._fallback_models():
                if self.provider == "groq" and model == primary_model:
                    continue
                attempts.append((groq_client, "groq", model))

        if self.groq_backup_client:
            for model in self._backup_groq_models(primary_model):
                attempts.append((self.groq_backup_client, "groq_backup", model))
        return attempts

    def _fallback_max_tokens(self, max_tokens: int) -> int:
        """Use a shorter answer on fallback to reduce token pressure during interviews."""
        configured = os.getenv("GROQ_FALLBACK_MAX_TOKENS")
        if configured:
            try:
                return max(80, min(max_tokens, int(configured)))
            except ValueError:
                logger.warning("Invalid GROQ_FALLBACK_MAX_TOKENS=%s; using default.", configured)
        return min(max_tokens, 350)

    def _gemini_request(self, messages: list, max_tokens: int, temperature: float) -> tuple[str, object]:
        """Translate chat-style messages into Gemini contents and config."""
        system_parts = []
        content_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                content_parts.append(f"{role.upper()}:\n{content}")

        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)

        config = genai_types.GenerateContentConfig(**config_kwargs) if genai_types else config_kwargs
        return "\n\n".join(content_parts), config

    async def _chat_completion_create(
        self,
        client,
        provider: str,
        model: str,
        messages: list,
        max_tokens: int,
        temperature: float,
        stream: bool = False,
    ):
        """Create a chat completion across supported provider SDKs."""
        if provider == "gemini":
            contents, config = self._gemini_request(messages, max_tokens, temperature)
            if stream:
                return client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config,
                )
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

        return await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

    async def _stream_chunks(self, stream):
        """Yield chunks from either async or sync provider streams."""
        if hasattr(stream, "__aiter__"):
            async for chunk in stream:
                yield chunk
            return

        for chunk in stream:
            yield chunk

    async def _aclose_stream(self, stream) -> None:
        """Best-effort close of a provider stream.

        When the consumer breaks out early (interviewer interruption) the async
        generator gets GeneratorExit; if the underlying httpx stream isn't closed
        here it gets cleaned up later from a different task, raising
        'Attempted to exit cancel scope in a different task'. Closing it in this
        task avoids that noise.
        """
        for name in ("aclose", "close"):
            closer = getattr(stream, name, None)
            if closer is None:
                continue
            try:
                result = closer()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass
            return

    def _response_text(self, response) -> str:
        """Extract assistant text defensively across provider response shapes."""
        text = getattr(response, "text", None)
        if isinstance(text, str) and text:
            return str(text).strip()
        # Gemini responses expose `.text`; an empty/None one means no content.
        # Groq/OpenAI responses have no `.text` (None) — fall through to .choices.
        if response.__class__.__name__ == "GenerateContentResponse":
            return ""

        try:
            message = response.choices[0].message
        except (AttributeError, IndexError, TypeError):
            return ""

        content = getattr(message, "content", None)
        if content is None:
            return ""
        return str(content).strip()

    def _chunk_text(self, chunk) -> str:
        """Extract stream delta text defensively across provider chunk shapes."""
        text = getattr(chunk, "text", None)
        if isinstance(text, str) and text:
            return str(text)
        # Gemini stream chunks expose `.text`; Groq/OpenAI chunks have no `.text`
        # (None) and carry content in .choices[0].delta — fall through to it.
        if chunk.__class__.__name__ == "GenerateContentResponse":
            return ""

        try:
            delta = chunk.choices[0].delta
        except (AttributeError, IndexError, TypeError):
            return ""

        content = getattr(delta, "content", None)
        if content is None:
            return ""
        return str(content)

    async def classify_question(self, question_text: str) -> str:
        """
        Classify interview question into type: behavioral, technical, coding, followup, or filler.
        Uses the fast 8b model for quick classification.
        """
        if not self.client or len(question_text.strip()) < 3:
            return "behavioral"

        if self.provider == "nvidia":
            classification_model = self.nvidia_behavior_model
        elif self.provider == "gemini":
            classification_model = self.gemini_behavior_model
        else:
            classification_model = self.groq_behavior_model

        try:
            response = await self._chat_completion_create(
                self.client,
                self.provider,
                classification_model,
                messages=[
                    {"role": "system", "content": "You are a question classifier. Reply with only one word."},
                    {"role": "user", "content": classification_prompt.format(question=question_text[:200])}
                ],
                max_tokens=10,
                temperature=0.0,
            )
            result = self._response_text(response).lower()
            if not result:
                logger.warning("Question classification returned empty content.")
                return "behavioral"

            valid_types = {"behavioral", "technical", "coding", "followup", "filler"}
            for vtype in valid_types:
                if vtype in result:
                    logger.info(f"Question classified as: {vtype}")
                    return vtype

            return "behavioral"  # Default fallback
        except Exception as e:
            logger.error(f"Question classification failed: {e}")
            return "behavioral"  # Safe fallback

    def _heuristic_turn_classification(self, current_text: str, previous_question: str = "", was_answering: bool = False) -> dict:
        """Cheap local fallback for turn-taking when the LLM classifier is unavailable."""
        text = (current_text or "").strip()
        text_lower = text.lower()
        words = text_lower.split()

        filler_phrases = {
            "yes", "yeah", "yep", "no", "okay", "ok", "right", "good",
            "great", "thanks", "thank you", "hello", "hi", "nice"
        }
        if not text or text_lower in filler_phrases:
            return {
                "intent": "filler",
                "should_answer_now": False,
                "confidence": 0.85,
                "clean_question": text,
            }

        dangling_words = {
            "and", "or", "but", "because", "with", "using", "for", "to", "from",
            "the", "a", "an", "that", "which", "when", "where", "if", "so",
            "like", "about", "regarding", "into", "by", "of"
        }
        last_word = words[-1].rstrip(".,!?;:") if words else ""
        if len(words) < 4 or last_word in dangling_words:
            return {
                "intent": "incomplete",
                "should_answer_now": False,
                "confidence": 0.75,
                "clean_question": text,
            }

        followup_starters = (
            "why", "how so", "what about", "what if", "but", "and why",
            "can you elaborate", "could you explain", "you mentioned",
            "that approach", "that solution", "how would", "would you"
        )
        if previous_question and any(text_lower.startswith(s) for s in followup_starters):
            return {
                "intent": "interruption" if was_answering else "followup",
                "should_answer_now": True,
                "confidence": 0.78,
                "clean_question": text,
            }

        return {
            "intent": "new_question",
            "should_answer_now": True,
            "confidence": 0.7,
            "clean_question": text,
        }

    async def classify_turn(
        self,
        current_text: str,
        transcript_history: list = None,
        previous_question: str = "",
        was_answering: bool = False
    ) -> dict:
        """
        Classify whether the accumulated interviewer text is ready to answer.
        This is separate from question type classification; it protects against
        conversational pauses, filler, follow-ups, and interruptions.
        """
        fallback = self._heuristic_turn_classification(current_text, previous_question, was_answering)

        if not self.client or len((current_text or "").strip()) < 3:
            return fallback

        if self.provider == "nvidia":
            model = self.nvidia_behavior_model
        elif self.provider == "gemini":
            model = self.gemini_behavior_model
        else:
            model = self.groq_behavior_model
        history = "\n".join((transcript_history or [])[-8:])

        try:
            response = await self._chat_completion_create(
                self.client,
                self.provider,
                model,
                messages=[
                    {"role": "system", "content": "You are a strict JSON classifier for real-time interview turn-taking."},
                    {"role": "user", "content": turn_classification_prompt.format(
                        history=history[:1200],
                        previous_question=(previous_question or "")[:400],
                        current_text=(current_text or "")[:800],
                    )}
                ],
                max_tokens=120,
                temperature=0.0,
            )
            raw = self._response_text(response)
            if not raw:
                logger.warning("Turn classification returned empty content.")
                return fallback
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            data = json.loads(match.group(0) if match else raw)

            intent = str(data.get("intent", fallback["intent"])).lower()
            valid_intents = {"new_question", "followup", "continuation", "filler", "interruption", "incomplete"}
            if intent not in valid_intents:
                intent = fallback["intent"]

            return {
                "intent": intent,
                "should_answer_now": bool(data.get("should_answer_now", fallback["should_answer_now"])),
                "confidence": float(data.get("confidence", fallback["confidence"])),
                "clean_question": str(data.get("clean_question") or current_text or "").strip(),
            }
        except Exception as e:
            logger.error(f"Turn classification failed: {e}")
            return fallback

    async def generate_answer(self, transcript_history: list, rag_context: str, question_type: str = None, is_followup: bool = False, conversation_summary: str = "", max_tokens_override: int = None) -> str:
        """
        Non-streaming fallback LLM call.
        """
        if not self.client:
            return f"Error: {self.provider.upper()} API Key missing."

        # Auto-classify if not provided
        if not question_type and transcript_history:
            last_interviewer = ""
            for msg in reversed(transcript_history):
                if msg.startswith("[INTERVIEWER]"):
                    last_interviewer = msg.replace("[INTERVIEWER]: ", "").strip()
                    break
            question_type = await self.classify_question(last_interviewer)

        # Combine last N transcripts to give the LLM conversation context
        recent_history = "\n".join(transcript_history[-15:])

        # Build enhanced prompt with conversation context
        context_parts = [f"Portfolio/Resume Context:\n{rag_context}"]
        if conversation_summary:
            context_parts.append(f"Conversation Summary:\n{conversation_summary}")
        if is_followup:
            context_parts.append("This is a FOLLOW-UP question. Build on your previous answer.")

        context_parts.append(f"Recent Conversation:\n{recent_history}")
        context_parts.append("Generate your response now. If no [INTERVIEWER] question is present, reply exactly with 'SKIP'.")

        # Route to the right model based on question complexity
        model, max_tokens = self._route_model(recent_history.lower(), question_type)
        if max_tokens_override:
            max_tokens = max_tokens_override

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n\n".join(context_parts)}
        ]

        attempts = self._model_attempts(model)
        for index, (attempt_client, attempt_provider, attempt_model) in enumerate(attempts):
            attempt_max_tokens = max_tokens if index == 0 else self._fallback_max_tokens(max_tokens)
            try:
                if index > 0:
                    logger.warning("Retrying LLM generation with %s fallback model: %s", attempt_provider, attempt_model)
                response = await self._chat_completion_create(
                    attempt_client,
                    attempt_provider,
                    attempt_model,
                    messages=messages,
                    max_tokens=attempt_max_tokens,
                    temperature=0.2,
                )

                output = self._response_text(response)
                if not output:
                    raise ValueError("LLM returned empty content")
                return output

            except Exception as e:
                if self._is_retryable_llm_error(e) and index < len(attempts) - 1:
                    logger.warning("LLM %s model %s failed; trying fallback. Error: %s", attempt_provider, attempt_model, e)
                    continue
                logger.error(f"LLM Generation failed: {e}")
                return "Error generating response."

        return "Error generating response."

    async def _generate_answer_stream_without_fallback(self, transcript_history: list, rag_context: str, question_type: str = None, is_followup: bool = False, conversation_summary: str = "", max_tokens_override: int = None):
        """
        Streaming version of generate_answer.
        """
        if not self.client:
            yield f"Error: {self.provider.upper()} API Key missing."
            return

        # Auto-classify if not provided
        if not question_type and transcript_history:
            last_interviewer = ""
            for msg in reversed(transcript_history):
                if msg.startswith("[INTERVIEWER]"):
                    last_interviewer = msg.replace("[INTERVIEWER]: ", "").strip()
                    break
            question_type = await self.classify_question(last_interviewer)

        recent_history = "\n".join(transcript_history[-15:])

        # Build enhanced prompt with conversation context
        context_parts = [f"Portfolio/Resume Context:\n{rag_context}"]
        if conversation_summary:
            context_parts.append(f"Conversation Summary:\n{conversation_summary}")
        if is_followup:
            context_parts.append("This is a FOLLOW-UP question. Build on your previous answer.")

        context_parts.append(f"Recent Conversation:\n{recent_history}")
        context_parts.append("Generate your response now. If no [INTERVIEWER] question is present, reply exactly with 'SKIP'.")

        model, max_tokens = self._route_model(recent_history.lower(), question_type)
        if max_tokens_override:
            max_tokens = max_tokens_override

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n\n".join(context_parts)}
        ]

        try:
            accumulated = ""
            stream = await self._chat_completion_create(
                self.client,
                self.provider,
                model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                stream=True,
            )
            try:
                async for chunk in self._stream_chunks(stream):
                    # Skip Groq/OpenAI empty-choices chunks; Gemini chunks have no
                    # .choices and must fall through to _chunk_text (which reads .text).
                    if getattr(chunk, "choices", None) is not None and not chunk.choices:
                        continue
                    delta = self._chunk_text(chunk)
                    if delta:
                        accumulated += delta
                        # Early SKIP detection — if first tokens spell SKIP, abort
                        if len(accumulated) <= 8 and "SKIP" in accumulated.strip().upper():
                            yield "__SKIP__"
                            return
                        yield delta
            finally:
                await self._aclose_stream(stream)

        except Exception as e:
            logger.error(f"LLM Stream failed: {e}")
            yield f"Error streaming response: {e}"

    async def generate_answer_stream(self, transcript_history: list, rag_context: str, question_type: str = None, is_followup: bool = False, conversation_summary: str = "", max_tokens_override: int = None):
        """
        Streaming version of generate_answer with automatic Groq fallback on rate limits.
        """
        if not self.client:
            yield f"Error: {self.provider.upper()} API Key missing."
            return

        if not question_type and transcript_history:
            last_interviewer = ""
            for msg in reversed(transcript_history):
                if msg.startswith("[INTERVIEWER]"):
                    last_interviewer = msg.replace("[INTERVIEWER]: ", "").strip()
                    break
            question_type = await self.classify_question(last_interviewer)

        recent_history = "\n".join(transcript_history[-15:])

        context_parts = [f"Portfolio/Resume Context:\n{rag_context}"]
        if conversation_summary:
            context_parts.append(f"Conversation Summary:\n{conversation_summary}")
        if is_followup:
            context_parts.append("This is a FOLLOW-UP question. Build on your previous answer.")

        context_parts.append(f"Recent Conversation:\n{recent_history}")
        context_parts.append("Generate your response now. If no [INTERVIEWER] question is present, reply exactly with 'SKIP'.")

        model, max_tokens = self._route_model(recent_history.lower(), question_type)
        if max_tokens_override:
            max_tokens = max_tokens_override

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n\n".join(context_parts)}
        ]

        attempts = self._model_attempts(model)
        for index, (attempt_client, attempt_provider, attempt_model) in enumerate(attempts):
            attempt_max_tokens = max_tokens if index == 0 else self._fallback_max_tokens(max_tokens)
            try:
                accumulated = ""
                if index > 0:
                    logger.warning("Retrying LLM stream with %s fallback model: %s", attempt_provider, attempt_model)
                stream = await self._chat_completion_create(
                    attempt_client,
                    attempt_provider,
                    attempt_model,
                    messages=messages,
                    max_tokens=attempt_max_tokens,
                    temperature=0.2,
                    stream=True,
                )
                try:
                    async for chunk in self._stream_chunks(stream):
                        # Skip Groq/OpenAI empty-choices chunks; Gemini chunks have no
                        # .choices and must fall through to _chunk_text (reads .text).
                        if getattr(chunk, "choices", None) is not None and not chunk.choices:
                            continue
                        delta = self._chunk_text(chunk)
                        if delta:
                            accumulated += delta
                            if len(accumulated) <= 8 and "SKIP" in accumulated.strip().upper():
                                yield "__SKIP__"
                                return
                            yield delta
                finally:
                    await self._aclose_stream(stream)
                if not accumulated.strip():
                    raise ValueError("LLM stream returned empty content")
                return

            except Exception as e:
                if self._is_retryable_llm_error(e) and index < len(attempts) - 1:
                    logger.warning("LLM stream %s model %s failed; trying fallback. Error: %s", attempt_provider, attempt_model, e)
                    continue
                logger.error(f"LLM Stream failed: {e}")
                yield f"Error streaming response: {e}"
                return

    async def generate_answer_regen(self, transcript_history: list, rag_context: str, last_question: str = "", question_type: str = None, conversation_summary: str = "", max_tokens_override: int = None) -> str:
        """Same as generate_answer but with higher temperature for a fresh, different take."""
        if not self.client:
            return f"Error: {self.provider.upper()} API Key missing."

        # Auto-classify if not provided
        if not question_type:
            question_type = await self.classify_question(last_question)

        recent_history = "\n".join(transcript_history[-15:])
        model, max_tokens = self._route_model((last_question + " " + recent_history).lower(), question_type)
        if max_tokens_override:
            max_tokens = max_tokens_override

        question_prompt = last_question if last_question else "the most recent interview question in the conversation"

        # Build enhanced prompt
        context_parts = [f"Portfolio/Resume Context:\n{rag_context}"]
        if conversation_summary:
            context_parts.append(f"Conversation Summary:\n{conversation_summary}")

        context_parts.append(f"Recent Conversation:\n{recent_history}")
        context_parts.append(f"The user pressed regenerate for this question: \"{question_prompt}\"\nGenerate a DIFFERENT, FRESH answer. Do NOT say SKIP.")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n\n".join(context_parts)}
        ]
        attempts = self._model_attempts(model)
        for index, (attempt_client, attempt_provider, attempt_model) in enumerate(attempts):
            attempt_max_tokens = max_tokens if index == 0 else self._fallback_max_tokens(max_tokens)
            try:
                if index > 0:
                    logger.warning("Retrying LLM regen with %s fallback model: %s", attempt_provider, attempt_model)
                response = await self._chat_completion_create(
                    attempt_client,
                    attempt_provider,
                    attempt_model,
                    messages=messages,
                    max_tokens=attempt_max_tokens,
                    temperature=0.6,  # Higher temperature = different answer each regen
                )
                output = self._response_text(response)
                if not output:
                    raise ValueError("LLM returned empty content")
                return output
            except Exception as e:
                if self._is_retryable_llm_error(e) and index < len(attempts) - 1:
                    logger.warning("LLM regen %s model %s failed; trying fallback. Error: %s", attempt_provider, attempt_model, e)
                    continue
                logger.error(f"LLM Regen failed: {e}")
                return "Error regenerating response."

        return "Error regenerating response."


    async def is_question_complete(self, text: str) -> bool:
        """
        Fast LLM call to determine if the speaker has finished their thought/question,
        or if they simply paused mid-sentence.
        """
        if not self.client or len(text.strip()) < 10:
            return False # Too short to evaluate or missing API key
            
        system_prompt = '''Evaluate the provided speech transcript fragment.
Determine if the speaker is resting at a natural endpoint, or if they are in the exact middle of a sentence and clearly intend to say more.

Reply YES if:
- It is a complete thought, statement, or question.
- It is a natural endpoint where someone might stop speaking.

Reply NO if:
- They are cut off in the exact middle of a sentence.
- The phrase ends with a dangling conjunction or preposition (e.g. 'and', 'or', 'because', 'using', 'with', 'the', 'we', 'is', 'it').
- It clearly requires an object or subject to make sense.

Reply ONLY YES or NO.

Examples:
Text: We built it using Python and
Output: NO

Text: Can you explain your reasoning?
Output: YES

Text: So regarding the AWS infrastructure
Output: NO

Text: The best approach is to use AWS Lambda.
Output: YES

Text: Tell me about yourself.
Output: YES'''

        if self.provider == "nvidia":
            gating_model = self.nvidia_behavior_model
        elif self.provider == "gemini":
            gating_model = self.gemini_behavior_model
        else:
            gating_model = self.groq_behavior_model

        try:
            response = await self._chat_completion_create(
                self.client,
                self.provider,
                gating_model, # Ultra-fast model for gating
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text: {text}\nOutput:"}
                ],
                max_tokens=3, # We only need YES or NO
                temperature=0.0, # Deterministic
            )
            
            output = self._response_text(response).upper()
            return "YES" in output
            
        except Exception as e:
            logger.error(f"Semantic evaluation failed: {e}")
            return True # Fallback to True so we don't hang the system on API error
