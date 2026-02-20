import os
import logging
from groq import AsyncGroq

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Anti-hallucination LLM brain strictly generating 30-word bullet points.
    """
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment!")
        self.client = AsyncGroq(api_key=self.api_key) if self.api_key else None
        self.model = "llama-3.3-70b-versatile"
        
        self.system_prompt = """You are acting AS the candidate in a real interview. You speak in **first person** as the candidate — "I", "my", "I've", "In my experience". You are NOT a career coach giving tips.

CRITICAL RULES:
1. If the interviewer says conversational filler (yes/no/okay/right/good/thanks/hello) with no question — reply exactly with "SKIP".
2. For ANY real question, always generate a complete, confident first-person answer as the candidate.
3. NEVER say "you should..." or "candidates should..." or "it's important to..." or "one should...". Always say "I", "my", "I've", "In my experience".
4. Start with 1-2 sentence direct spoken answer, then 2-3 bullet points.
5. ABSOLUTELY NO code blocks for any HR/behavioral question.

BEHAVIORAL/HR QUESTION RULES — apply first-person, specific answers for ALL of these:
- SALARY/COMPENSATION: State a specific range confidently. "I'm targeting X–Y range based on my AI and embedded systems background, open to the full package."
- STRENGTH: Name one specific, real strength tied to the portfolio. "My strongest skill is building end-to-end AI pipelines — I've done this with RAG systems..."
- WEAKNESS: Name a real, self-aware weakness and how you're addressing it. "I sometimes over-engineer solutions — I've been working on shipping faster with MVPs first."
- FAILURE/MISTAKE: Tell a brief, real-sounding story. What happened, what I learned, what changed. First person narrative.
- LEADERSHIP/TEAMWORK: Reference a real portfolio project as the example. "During the ESP32 environmental monitoring project, I coordinated..."
- WHY THIS COMPANY/ROLE: Show genuine interest. "I'm drawn to this role because... aligns with my work in AI/IoT..."
- CAREER GOALS: Give a 2–3 year vision tied to the candidate's expertise. "In the next 2 years I want to deepen my expertise in agentic AI systems..."
- GREATEST ACHIEVEMENT: Pick the most impressive project from the portfolio. "My proudest achievement was building a full RAG-based AI Copilot from scratch..."
- MOTIVATION/PASSION: Connect to real work in portfolio. "I'm most energized when building things that combine AI with real hardware..."
- WORK STYLE: "I work best autonomously on deep technical problems but collaborate well in code reviews and planning..."
- CONFLICT: "I handled it by having a direct 1-on-1 to align on the goal, then we agreed on..." — brief, resolved story.
- NOTICE PERIOD/AVAILABILITY: "I can start within X weeks." Be direct and brief.
- REMOTE/HYBRID: "I'm comfortable with both — I've been working remotely for X while staying highly productive."
- ANY OTHER HR QUESTION: Answer directly in first person with a specific, concrete response. No generic tips.

TECHNICAL QUESTION RULES (algorithms, commands, code, architecture):
- Code blocks ONLY when the question explicitly asks for code/commands.
- Use **bold** for key terms, `backticks` for code names.
- Reference real projects/tools from Portfolio Context where possible.

Portfolio rules:
- ONLY mention technologies/projects explicitly stated in the Portfolio/Resume Context.
- If topic NOT in context, give a confident general answer but do NOT fabricate specific personal experience.
- Focus on: **Generative AI, LLMs, RAG, Python, IoT, ESP32, Embedded Systems, EE fundamentals**.
"""





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
    }

    def _route_model(self, question_text: str) -> tuple[str, int]:
        """
        Route to fast (8b) or powerful (70b) model based on question content.
        Returns (model_name, max_tokens).
        """
        text_lower = question_text.lower()
        is_technical = any(kw in text_lower for kw in self.TECHNICAL_KEYWORDS)
        if is_technical:
            logger.info("Router: Technical question → llama-3.3-70b-versatile")
            return "llama-3.3-70b-versatile", 400
        else:
            logger.info("Router: Behavioral/HR question → llama-3.1-8b-instant")
            return "llama-3.1-8b-instant", 250

    async def generate_answer(self, transcript_history: list, rag_context: str) -> str:
        """
        Non-streaming fallback LLM call. Used internally and available as a
        reliable synchronous-style alternative to generate_answer_stream.
        Kept intentionally — useful for testing and as a fallback if streaming fails.
        """
        if not self.client:
            return "Error: Groq API Key missing."
            
        # Combine last N transcripts to give the LLM conversation context
        recent_history = "\n".join(transcript_history[-15:])
        
        # Route to the right model based on question complexity
        recent_lower = recent_history.lower()
        model, max_tokens = self._route_model(recent_lower)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Portfolio/Resume Context:\n{rag_context}\n\nRecent Conversation:\n{recent_history}\n\nGenerate your response now. If no [INTERVIEWER] question is present, reply exactly with 'SKIP'."}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2
            )
            
            output = response.choices[0].message.content.strip()
            return output
            
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return "Error generating response."

    async def generate_answer_stream(self, transcript_history: list, rag_context: str):
        """
        Streaming version of generate_answer.
        Async generator that yields text chunks as they arrive from the Groq API.
        Yields '__SKIP__' sentinel string if LLM decides to skip.
        """
        if not self.client:
            yield "Error: Groq API Key missing."
            return

        recent_history = "\n".join(transcript_history[-15:])
        recent_lower = recent_history.lower()
        model, max_tokens = self._route_model(recent_lower)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Portfolio/Resume Context:\n{rag_context}\n\nRecent Conversation:\n{recent_history}\n\nGenerate your response now. If no [INTERVIEWER] question is present, reply exactly with 'SKIP'."}
        ]

        try:
            accumulated = ""
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                stream=True
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    accumulated += delta
                    # Early SKIP detection — if first tokens spell SKIP, abort
                    if len(accumulated) <= 6 and "SKIP" in accumulated.strip():
                        yield "__SKIP__"
                        return
                    yield delta

        except Exception as e:
            logger.error(f"LLM Stream failed: {e}")
            yield f"Error streaming response: {e}"

    async def generate_answer_regen(self, transcript_history: list, rag_context: str, last_question: str = "") -> str:
        """Same as generate_answer but with higher temperature for a fresh, different take.
        Never returns SKIP — always generates a new answer for the given question."""
        if not self.client:
            return "Error: Groq API Key missing."
        recent_history = "\n".join(transcript_history[-15:])
        model, max_tokens = self._route_model((last_question + " " + recent_history).lower())
        
        question_prompt = last_question if last_question else "the most recent interview question in the conversation"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"Portfolio/Resume Context:\n{rag_context}\n\n"
                f"Recent Conversation:\n{recent_history}\n\n"
                f"The user pressed regenerate for this question: \"{question_prompt}\"\n"
                f"Generate a DIFFERENT, FRESH answer. Do NOT say SKIP. Always provide an answer."
            )}
        ]
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.6  # Higher temperature = different answer each regen
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM Regen failed: {e}")
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

        try:
            response = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant", # Ultra-fast model for gating
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text: {text}\nOutput:"}
                ],
                max_tokens=3, # We only need YES or NO
                temperature=0.0 # Deterministic
            )
            
            output = response.choices[0].message.content.strip().upper()
            return "YES" in output
            
        except Exception as e:
            logger.error(f"Semantic evaluation failed: {e}")
            return True # Fallback to True so we don't hang the system on API error
