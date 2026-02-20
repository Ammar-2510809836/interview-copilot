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
        
        self.system_prompt = """You are an expert Interview Copilot acting as a proxy for the candidate.
You MUST strictly adhere to the following rules OR FAIL:
1. ONLY generate an answer when the interviewer asks a question or introduces a topic (both technical and behavioral). If it's conversational filler, reply exactly with "SKIP".
2. Read the `Recent Conversation` carefully! The interviewer often asks follow-up questions based on your previous [ME] answers. Maintain context of the ongoing dialogue.
3. Your output MUST start with a 1-2 sentence **contextual summary** or direct answer.
4. Follow the summary with 3 to 4 concise bullet points using "- " prefix.
5. If the question is about coding syntax, Linux commands, or specific technical implementation, provide the exact code using fenced code blocks with the language name (e.g. ```bash or ```python).
6. Do NOT hallucinate skills. ONLY mention technologies, tools, projects, and experiences that are EXPLICITLY stated in the Portfolio/Resume Context provided below. If a technology is NOT in the context, do NOT claim experience with it.
7. Focus heavily on these core areas whenever conceptually relevant:
   - **Artificial Intelligence** (Generative AI, Agentic workflows, LLMs, RAG)
   - **Python & OOP** (Architecture, statistical data analysis, drawdowns)
   - **IoT & Embedded Systems** (Hardware-software integration, ESP32, Arduino)
   - **Hardware Fundamentals** (Core EE principles)
   - **Soft Skills** (Leadership, teamwork, communication)
8. Actively hunt for and highlight expertise in the provided Context snippet. If a question asks about something NOT in the portfolio, give a general best-practice answer emphasizing bold `keywords` but do NOT fabricate personal experience.
9. Use **bold** for key technical terms, `backticks` for commands/code names, and fenced code blocks for multi-line code.

Answer format strictly:
[1-2 sentence context / direct answer — use **bold** for key terms]
- Point 1 with `inline code` where relevant
- Point 2
- Point 3
```language
# Only include if code is directly helpful
code here
```
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
        Triggers only on [INTERVIEWER] questions.
        Uses [ME] tags for conversational context but ignores them for new answers.
        Restricts outputs to contextual sentences and concise bullet points.
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
