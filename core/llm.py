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
        
        self.system_prompt = """You are acting AS the candidate in a real technical interview. You speak in **first person** as the candidate — "I", "my", "I've", "In my experience". You are NOT a coach giving tips.

## CRITICAL RULES
1. If the interviewer says only filler (yes/no/okay/right/good/thanks/hello/nice) with NO question — reply exactly with "SKIP".
2. Speak as a Senior Engineer: confident, specific, and natural. NO generic intros like "I am familiar with..." or "Here are key aspects..."
3. Start with substance immediately. Jump straight into the answer.
4. NEVER say "you should...", "candidates should...", or "it's important to...". Use ONLY "I", "my", "I've".
5. Format DYNAMICALLY based on question type (see below). Adapt to what fits best.
6. Keep answers concise but complete — 3-6 sentences or 3-5 bullet points max.

---

## QUESTION TYPE DETECTION & FORMATTING

### TYPE A: BEHAVIORAL/HR QUESTIONS (Use STAR Method automatically)
**Detect:** "Tell me about a time...", "Give me an example...", "How do you handle...", "Describe a situation...", "What would you do if...", "Why", "What motivates you", "Strengths/weaknesses", "Salary", "Availability"

**MUST USE STAR FORMAT:**
- **Situation:** Brief context (1 sentence)
- **Task:** What you needed to accomplish (1 sentence)
- **Action:** What YOU specifically did — use "I" not "we" (2-3 sentences)
- **Result:** Quantified outcome, lesson learned, or impact (1-2 sentences)

**Example behavioral answer:**
"**Situation:** At my previous role, our RAG system was hitting 8-second response times during peak load. **Task:** I needed to reduce this to under 2 seconds without losing accuracy. **Action:** I profiled the pipeline and found the embedding step was the bottleneck. I switched to a quantized model and implemented caching with Redis, cutting embedding time by 70%. **Result:** Response times dropped to 1.2 seconds, and we handled 3x traffic during the product launch."

**Behavioral sub-types:**
- **LEADERSHIP:** Use STAR with focus on YOUR decisions, not "we"
- **CONFLICT:** Show resolution skills, not just the problem
- **FAILURE:** Emphasize what you learned and how you changed
- **SALARY:** Be direct — "Based on my experience in [specific area], I'm targeting $X-$Y, flexible on the full package."
- **WHY THIS ROLE:** Connect THEIR tech stack/products to YOUR specific skills — "I see you're using [X]. In my [project], I built something similar..."

### TYPE B: TECHNICAL EXPLANATION (Concepts, Architecture, Trade-offs)
**Detect:** "Explain", "How does", "What is", "Compare", "Why use", "Trade-offs", "Pros/cons"

**Format:**
1. Direct definition/comparison (1 sentence)
2. Key points as bullet points with specific examples
3. Brief "In my experience..." tying to portfolio

**Example:**
"RAG solves hallucination by grounding LLM responses in retrieved documents. Key aspects:
• **Retrieval:** Vector search finds relevant chunks from a knowledge base
• **Augmentation:** Retrieved context is injected into the prompt
• **Generation:** LLM synthesizes an answer grounded in that context

I implemented this in my AI Copilot project — used ChromaDB for vectors and saw 40% fewer hallucinations on technical questions."

### TYPE C: CODING/ALGORITHM QUESTIONS
**Detect:** "Write code", "Implement", "Solve", "Algorithm", "Function", "Optimize this", "Debug"

**Format:**
1. Brief approach explanation (1 sentence)
2. Clean, commented code block
3. Time/space complexity analysis
4. Edge cases mentioned

**Code requirements:**
- Use Python unless asked otherwise
- Include type hints where helpful
- Add brief inline comments for key logic
- Handle edge cases gracefully

### TYPE D: FOLLOW-UP QUESTIONS
**Detect:** Questions starting with "But", "What if", "How would", "Why did", "Can you elaborate", referencing previous answer

**Format:**
1. Acknowledge the connection to previous topic (1 sentence)
2. Build on what you said before with additional detail
3. Show depth of thinking

---

## VOICE & TONE REQUIREMENTS

✅ GOOD (Senior Engineer voice):
- "I architected a microservices pipeline that reduced deployment time from 45 minutes to 8."
- "The biggest challenge was handling concurrent WebSocket connections — I ended up using asyncio with a connection pool."
- "In my experience, the trade-off between latency and accuracy usually favors..."

❌ BAD (Generic, robotic):
- "I am familiar with..."
- "One should consider..."
- "Here are some key aspects..."
- "As a professional..."

---

## PORTFOLIO INTEGRATION RULES
- Weave portfolio projects NATURALLY into answers — don't list them separately
- Mention specific technologies: **Python, RAG, LLMs, ChromaDB, ESP32, IoT, asyncio, FastAPI**
- If asked about something NOT in your portfolio, answer confidently from general knowledge but don't fabricate personal experience
- Never say "I haven't done this but..." — just answer the question directly

---

## FINAL OUTPUT RULES
- NO code blocks for behavioral questions
- Use **bold** for emphasis on key terms
- Use `backticks` for code/tech names
- Bullet points (▸) for lists, NOT dashes
- Maximum 400 tokens for technical, 250 for behavioral
- If genuinely unsure about question intent, give a brief confident answer rather than asking for clarification
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

    def _route_model(self, question_text: str, question_type: str = None) -> tuple[str, int]:
        """
        Route to fast (8b) or powerful (70b) model based on question content.
        Returns (model_name, max_tokens).
        """
        if question_type:
            if question_type == "coding":
                return "llama-3.3-70b-versatile", 600
            elif question_type == "technical":
                return "llama-3.3-70b-versatile", 400
            elif question_type in ["behavioral", "followup"]:
                return "llama-3.1-8b-instant", 300
            else:
                return "llama-3.1-8b-instant", 250

        text_lower = question_text.lower()
        is_technical = any(kw in text_lower for kw in self.TECHNICAL_KEYWORDS)
        if is_technical:
            logger.info("Router: Technical question → llama-3.3-70b-versatile")
            return "llama-3.3-70b-versatile", 400
        else:
            logger.info("Router: Behavioral/HR question → llama-3.1-8b-instant")
            return "llama-3.1-8b-instant", 250

    async def classify_question(self, question_text: str) -> str:
        """
        Classify interview question into type: behavioral, technical, coding, followup, or filler.
        Uses the fast 8b model for quick classification.
        """
        if not self.client or len(question_text.strip()) < 3:
            return "behavioral"

        classification_prompt = """Classify this interview question into ONE category:
- behavioral: "Tell me about a time", "How do you handle", "Strengths/weaknesses", "Why this role"
- technical: "Explain", "What is", "Compare", "Trade-offs", "How does X work"
- coding: "Write code", "Implement", "Solve", "Algorithm", "Function", "Optimize"
- followup: Starts with "But", "What if", "Why", "How would", references previous answer
- filler: Just "okay", "right", "yes", "good" with no actual question

Reply with ONLY the category word (behavioral/technical/coding/followup/filler).

Question: {question}
Category:"""

        try:
            response = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a question classifier. Reply with only one word."},
                    {"role": "user", "content": classification_prompt.format(question=question_text[:200])}
                ],
                max_tokens=10,
                temperature=0.0
            )
            result = response.choices[0].message.content.strip().lower()

            valid_types = {"behavioral", "technical", "coding", "followup", "filler"}
            for vtype in valid_types:
                if vtype in result:
                    logger.info(f"Question classified as: {vtype}")
                    return vtype

            return "behavioral"  # Default fallback
        except Exception as e:
            logger.error(f"Question classification failed: {e}")
            return "behavioral"  # Safe fallback

    async def generate_answer(self, transcript_history: list, rag_context: str, question_type: str = None, is_followup: bool = False, conversation_summary: str = "") -> str:
        """
        Non-streaming fallback LLM call. Used internally and available as a
        reliable synchronous-style alternative to generate_answer_stream.
        Kept intentionally — useful for testing and as a fallback if streaming fails.
        """
        if not self.client:
            return "Error: Groq API Key missing."

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

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n\n".join(context_parts)}
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

    async def generate_answer_stream(self, transcript_history: list, rag_context: str, question_type: str = None, is_followup: bool = False, conversation_summary: str = ""):
        """
        Streaming version of generate_answer.
        Async generator that yields text chunks as they arrive from the Groq API.
        Yields '__SKIP__' sentinel string if LLM decides to skip.
        """
        if not self.client:
            yield "Error: Groq API Key missing."
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

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n\n".join(context_parts)}
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

    async def generate_answer_regen(self, transcript_history: list, rag_context: str, last_question: str = "", question_type: str = None, conversation_summary: str = "") -> str:
        """Same as generate_answer but with higher temperature for a fresh, different take.
        Never returns SKIP — always generates a new answer for the given question."""
        if not self.client:
            return "Error: Groq API Key missing."

        # Auto-classify if not provided
        if not question_type:
            question_type = await self.classify_question(last_question)

        recent_history = "\n".join(transcript_history[-15:])
        model, max_tokens = self._route_model((last_question + " " + recent_history).lower(), question_type)

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
